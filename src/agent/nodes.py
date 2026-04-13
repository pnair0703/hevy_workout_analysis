"""Agent nodes — each function is a step in the LangGraph pipeline.

Each node takes AgentState and returns a partial state update.
The graph orchestrates the order and conditional routing.
"""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from openai import OpenAI

from src.agent.router import classify_query
from src.agent.state import AgentState
from src.models.program import TrainingPhase, UserProfile
from src.tools.hevy import HevyClient
from src.tools.nutrition import adjust_volume_for_phase, check_nutrition_constraints
from src.tools.rag import format_context, retrieve
from src.tools.anomaly import detect_anomalies
from src.tools.volume_calc import (
    compute_e1rm_history,
    compute_training_analysis,
    compute_weekly_volume,
)


# ── Default user profile (update with your stats) ──────────────────

DEFAULT_PROFILE = UserProfile(
    name="Pranav",
    bodyweight_lbs=171,
    calories=2500,
    protein_g=200,
    carbs_g=280,
    fat_g=60,
    phase=TrainingPhase.CUT,
    training_days_per_week=5,
    injuries=[],
)


# ── Node 1: Router ─────────────────────────────────────────────────

def router_node(state: AgentState) -> dict:
    """Classify the query and assign a model."""
    result = classify_query(state["user_query"])
    return {
        "query_type": result["query_type"],
        "model": result["model"],
    }


# ── Node 2: Fetch Hevy data ────────────────────────────────────────

def hevy_node(state: AgentState) -> dict:
    """Pull recent workout data from Hevy API."""
    with HevyClient() as client:
        workouts = client.get_recent_workouts(days=28)
        templates = client.build_template_lookup()

        # Serialize workouts for the LLM context
        workout_summaries = []
        for w in workouts[:10]:  # last 10 workouts
            exercises = []
            for ex in w.exercises:
                top = ex.top_set
                exercises.append({
                    "name": ex.title,
                    "working_sets": ex.num_working_sets,
                    "top_weight_lbs": top.weight_lbs if top else None,
                    "top_reps": top.reps if top else None,
                    "e1rm_lbs": round(top.estimated_1rm * 2.20462, 1) if top and top.estimated_1rm else None,
                })
            workout_summaries.append({
                "date": w.start_time.strftime("%Y-%m-%d"),
                "title": w.title,
                "duration_min": w.duration_minutes,
                "exercises": exercises,
            })

    return {
        "hevy_data": {
            "workouts": workout_summaries,
            "templates": {tid: t.model_dump() for tid, t in list(templates.items())[:50]},
            "_raw_workouts": workouts,
            "_raw_templates": templates,
        }
    }


# ── Node 3: Volume analysis ────────────────────────────────────────

def volume_node(state: AgentState) -> dict:
    """Compute volume stats and identify gaps."""
    hevy = state.get("hevy_data", {})
    raw_workouts = hevy.get("_raw_workouts", [])
    raw_templates = hevy.get("_raw_templates", {})

    analysis = compute_training_analysis(raw_workouts, raw_templates, period_days=28)

    # Phase-adjusted targets
    adjusted = adjust_volume_for_phase(analysis.volume_by_muscle, DEFAULT_PROFILE)

    # Anomaly detection (z-score + Isolation Forest)
    anomalies = detect_anomalies(raw_workouts, raw_templates, n_weeks=12)

    return {
        "volume_analysis": {
            "period_days": analysis.period_days,
            "total_workouts": analysis.total_workouts,
            "avg_duration": analysis.avg_duration_minutes,
            "volume_by_muscle": [v.model_dump() for v in analysis.volume_by_muscle],
            "gaps": analysis.gaps,
            "overreaching_signals": analysis.overreaching_signals,
            "adjusted_targets": adjusted,
            "anomalies": anomalies,
        }
    }


# ── Node 4: RAG retrieval ──────────────────────────────────────────

def rag_node(state: AgentState) -> dict:
    """Retrieve relevant sports science context."""
    query = state["user_query"]

    # Also search for gap-related info if gaps exist
    volume = state.get("volume_analysis", {})
    gaps = volume.get("gaps", [])

    results = retrieve(query, n_results=3)

    # If there are gaps, do a second retrieval for gap-specific advice
    if gaps:
        gap_query = f"How to increase training volume for {', '.join(gaps[:3])}"
        gap_results = retrieve(gap_query, n_results=2)
        results.extend(gap_results)

    context = format_context(results)
    return {"rag_context": context}


# ── Node 5: Nutrition constraints ───────────────────────────────────

def nutrition_node(state: AgentState) -> dict:
    """Check nutrition constraints for the current phase."""
    constraints = check_nutrition_constraints(DEFAULT_PROFILE)
    return {"nutrition_constraints": constraints}


# ── Node 6: Synthesizer (generates the final response) ─────────────

SYNTH_SYSTEM_PROMPT = """You are IronAgent, an evidence-based AI training coach.
You have access to the user's real workout data from Hevy, volume analysis,
personalized anomaly detection, sports science literature, and nutrition context.

Your job is to give specific, actionable training advice backed by evidence.

RULES:
- Always reference the user's actual data (exercises, weights, sets, volume)
- Cite sports science sources when making recommendations
- Adjust recommendations based on the user's training phase (cut/bulk/maintain)
- Flag any training gaps proactively, even if the user didn't ask about them
- Use anomaly detection results to highlight muscles that are unusually low
  compared to the user's OWN history (z-score flags), not just population thresholds
- If the Isolation Forest flags an abnormal training pattern, mention it
- Give specific exercise, set, rep, and RPE recommendations — not vague advice
- If the user is cutting, prioritize intensity over volume
- Be concise and direct

RESPONSE FORMAT:
Start with a direct answer to the question. Then provide:
1. Specific recommendations with sets/reps/RPE
2. Evidence from the retrieved literature
3. Any gaps or anomalies the user should know about
"""


def synthesizer_node(state: AgentState) -> dict:
    """Generate the final response using the routed model."""
    query_type = state.get("query_type", "programming")
    model = state.get("model", "gpt-4o-mini")

    # For lookups, skip the LLM and format data directly
    if query_type == "lookup":
        return _handle_lookup(state)

    # Build context for the LLM
    context_parts = []

    # Add workout data
    hevy = state.get("hevy_data", {})
    workouts = hevy.get("workouts", [])
    if workouts:
        context_parts.append(
            f"## Recent Workouts (last 28 days)\n{json.dumps(workouts[:5], indent=2)}"
        )

    # Add volume analysis
    volume = state.get("volume_analysis", {})
    if volume:
        # Clean version without raw objects
        clean_volume = {
            k: v for k, v in volume.items()
            if not k.startswith("_") and k not in ("adjusted_targets", "anomalies")
        }
        context_parts.append(
            f"## Volume Analysis\n{json.dumps(clean_volume, indent=2)}"
        )

        adjusted = volume.get("adjusted_targets", [])
        if adjusted:
            context_parts.append(
                f"## Phase-Adjusted Volume Targets ({DEFAULT_PROFILE.phase.value})\n"
                f"{json.dumps(adjusted, indent=2)}"
            )

        # Add anomaly detection results
        anomalies = volume.get("anomalies", {})
        if anomalies:
            context_parts.append(
                f"## Anomaly Detection (personalized)\n"
                f"Summary: {anomalies.get('summary', 'No anomalies')}\n"
                f"Individual muscle anomalies (z-score vs personal baseline): "
                f"{json.dumps(anomalies.get('individual_anomalies', []), indent=2)}\n"
                f"Overall pattern: {json.dumps(anomalies.get('pattern_anomaly', {}), indent=2)}"
            )

    # Add RAG context
    rag = state.get("rag_context", "")
    if rag:
        context_parts.append(f"## Sports Science Literature\n{rag}")

    # Add nutrition
    nutrition = state.get("nutrition_constraints", {})
    if nutrition:
        context_parts.append(f"## Nutrition Context\n{json.dumps(nutrition, indent=2)}")

    full_context = "\n\n---\n\n".join(context_parts)

    # Call the routed model
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYNTH_SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{full_context}\n\nQUESTION: {state['user_query']}"},
        ],
        temperature=0.3,
        max_tokens=1500,
    )

    answer = response.choices[0].message.content

    return {
        "recommendation": {
            "answer": answer,
            "model_used": model,
            "query_type": query_type,
        },
        "messages": [AIMessage(content=answer)],
    }


def _handle_lookup(state: AgentState) -> dict:
    """Handle simple data lookups without calling an LLM."""
    hevy = state.get("hevy_data", {})
    workouts = hevy.get("workouts", [])
    volume = state.get("volume_analysis", {})

    query = state["user_query"].lower()

    # Simple heuristics for common lookups
    if any(k in query for k in ("last workout", "previous workout", "today")):
        if workouts:
            w = workouts[0]
            lines = [f"**{w['title']}** ({w['date']}, {w['duration_min']}min)"]
            for ex in w["exercises"]:
                weight = f" — {ex['top_weight_lbs']} lbs × {ex['top_reps']}" if ex["top_weight_lbs"] else ""
                lines.append(f"  • {ex['name']}: {ex['working_sets']} sets{weight}")
            answer = "\n".join(lines)
        else:
            answer = "No recent workouts found."

    elif any(k in query for k in ("how many", "count", "total")):
        n = len(workouts)
        answer = f"You've done {n} workouts in the last 28 days."

    elif any(k in query for k in ("gap", "missing", "lacking", "weak")):
        gaps = volume.get("gaps", [])
        if gaps:
            answer = f"Training gaps (below minimum effective volume): {', '.join(gaps)}"
        else:
            answer = "No major training gaps detected."

    else:
        # Fallback — summarize recent data
        answer = (
            f"Last 28 days: {volume.get('total_workouts', '?')} workouts, "
            f"avg {volume.get('avg_duration', '?')} min. "
            f"Gaps: {', '.join(volume.get('gaps', ['none']))}"
        )

    return {
        "recommendation": {
            "answer": answer,
            "model_used": "none (direct lookup)",
            "query_type": "lookup",
        },
        "messages": [AIMessage(content=answer)],
    }