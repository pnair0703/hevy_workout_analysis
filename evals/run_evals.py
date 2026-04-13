"""Eval runner — pytest-based test suite for IronAgent.

Runs each scenario through the agent, then grades the response
using an LLM-as-judge approach.

Usage:
    pytest evals/run_evals.py -v
    pytest evals/run_evals.py -v -k "chest_gap"     # run one scenario
    pytest evals/run_evals.py -v -s                   # show print output
"""

import json
import os
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

from src.agent.graph import run_agent
from evals.judges.judge_prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE


# ── Load scenarios ──────────────────────────────────────────────────

SCENARIOS_PATH = Path(__file__).parent / "scenarios" / "test_scenarios.json"

with open(SCENARIOS_PATH) as f:
    SCENARIOS = json.load(f)


# ── LLM Judge ──────────────────────────────────────────────────────

def judge_response(query: str, context: str, response: str, criteria: list[str]) -> dict:
    """Send the agent's response to an LLM judge for grading.

    Uses gpt-4o for judging since it needs strong reasoning.
    Returns dict with criteria_results, overall_score, summary.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    criteria_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(criteria))

    user_message = JUDGE_USER_TEMPLATE.format(
        query=query,
        context=context,
        response=response,
        criteria=criteria_text,
    )

    result = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        max_tokens=1000,
        response_format={"type": "json_object"},
    )

    raw = result.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "criteria_results": [],
            "overall_score": 0.0,
            "summary": f"Judge returned invalid JSON: {raw[:200]}",
        }


# ── Pytest fixtures ─────────────────────────────────────────────────

@pytest.fixture(scope="session")
def eval_results():
    """Shared dict to accumulate results across all tests."""
    return {}


# ── Parametrized tests ──────────────────────────────────────────────

@pytest.mark.parametrize(
    "scenario",
    SCENARIOS,
    ids=[s["id"] for s in SCENARIOS],
)
def test_scenario(scenario, eval_results):
    """Run a scenario through the agent and grade it."""
    query = scenario["query"]
    context = scenario["context"]
    criteria = scenario["expected_criteria"]

    # Run the agent
    print(f"\n🔄 Running: {scenario['name']}")
    print(f"   Query: {query}")

    agent_result = run_agent(query)
    response = agent_result.get("answer", "")
    model_used = agent_result.get("model_used", "unknown")
    query_type = agent_result.get("query_type", "unknown")

    print(f"   Route: {query_type} → {model_used}")
    print(f"   Response length: {len(response)} chars")

    # Judge the response
    print(f"   Judging...")
    judgment = judge_response(query, context, response, criteria)

    overall_score = judgment.get("overall_score", 0.0)
    summary = judgment.get("summary", "No summary")
    criteria_results = judgment.get("criteria_results", [])

    # Print results
    print(f"   Score: {overall_score:.2f}")
    for cr in criteria_results:
        icon = {"PASS": "✅", "FAIL": "❌", "PARTIAL": "🟡"}.get(cr["result"], "?")
        print(f"   {icon} {cr['criterion'][:60]}")
        if cr["result"] != "PASS":
            print(f"      └─ {cr['reasoning'][:80]}")

    print(f"   Summary: {summary}")

    # Store result
    eval_results[scenario["id"]] = {
        "name": scenario["name"],
        "score": overall_score,
        "model_used": model_used,
        "query_type": query_type,
        "criteria_results": criteria_results,
    }

    # Assert minimum score threshold
    assert overall_score >= 0.5, (
        f"Scenario '{scenario['name']}' scored {overall_score:.2f} "
        f"(minimum: 0.50). Summary: {summary}"
    )


# ── Summary report ──────────────────────────────────────────────────

def pytest_sessionfinish(session, exitstatus):
    """Print a summary report after all tests complete."""
    # This only works when run with pytest directly
    pass


# ── Standalone runner ───────────────────────────────────────────────

def run_all_evals():
    """Run all evals outside of pytest (for quick testing)."""
    print("=" * 60)
    print("IronAgent — Eval Suite")
    print("=" * 60)

    results = []

    for scenario in SCENARIOS:
        print(f"\n{'─' * 60}")
        print(f"📋 {scenario['name']}")
        print(f"   Query: {scenario['query']}")

        agent_result = run_agent(scenario["query"])
        response = agent_result.get("answer", "")
        model_used = agent_result.get("model_used", "unknown")
        query_type = agent_result.get("query_type", "unknown")

        print(f"   Route: {query_type} → {model_used}")

        judgment = judge_response(
            scenario["query"],
            scenario["context"],
            response,
            scenario["expected_criteria"],
        )

        score = judgment.get("overall_score", 0.0)
        criteria_results = judgment.get("criteria_results", [])

        print(f"   Score: {score:.2f}")
        for cr in criteria_results:
            icon = {"PASS": "✅", "FAIL": "❌", "PARTIAL": "🟡"}.get(cr["result"], "?")
            print(f"   {icon} {cr['criterion'][:70]}")

        results.append({
            "id": scenario["id"],
            "name": scenario["name"],
            "score": score,
            "route": f"{query_type} → {model_used}",
        })

    # Summary table
    print(f"\n{'=' * 60}")
    print("📊 EVAL SUMMARY")
    print(f"{'=' * 60}")

    total = 0
    passed = 0
    for r in results:
        total += 1
        status = "✅ PASS" if r["score"] >= 0.5 else "❌ FAIL"
        if r["score"] >= 0.5:
            passed += 1
        print(f"   {status}  {r['score']:.2f}  {r['name']}")
        print(f"          Route: {r['route']}")

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0
    print(f"\n   Passed: {passed}/{total}")
    print(f"   Average score: {avg_score:.2f}")
    print(f"{'=' * 60}")

    # Save results
    output_path = Path(__file__).parent / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n   Results saved to {output_path}")


if __name__ == "__main__":
    run_all_evals()