"""MoM Router — classifies query type and selects the optimal model.

This is the Mixture-of-Models implementation. Instead of routing every
query to the same LLM, the router classifies the query and picks the
best model for the job — optimizing for cost, quality, and latency.

Routes:
    lookup     → skip LLM entirely, direct Hevy API answer ($0)
    programming → gpt-4o-mini (structured, pattern-based)
    coaching    → gpt-4o (needs deeper reasoning)
    research    → gpt-4o-mini (summarization from RAG)
"""

from __future__ import annotations

import os

from openai import OpenAI

# Model assignments per query type
MODEL_MAP: dict[str, str] = {
    "lookup": "none",            # no LLM needed
    "programming": "gpt-4o-mini",
    "coaching": "gpt-4o",
    "research": "gpt-4o-mini",
}

ROUTER_PROMPT = """You are a query classifier for a fitness training AI agent.
Classify the user's query into exactly one of these categories:

1. "lookup" — Simple factual questions about the user's training data.
   Examples: "What was my bench PR?", "How many times did I train legs this week?",
   "What did I do last workout?"

2. "programming" — Questions about exercise selection, sets, reps, or programming.
   Examples: "What should I do for chest?", "Give me a back workout",
   "How many sets of biceps should I add?", "What exercises for rear delts?"

3. "coaching" — Complex questions requiring judgment, tradeoffs, or personalized reasoning.
   Examples: "My bench has stalled and I'm cutting, what should I change?",
   "I'm feeling burnt out, should I deload?", "I tweaked my shoulder, how should I adjust?",
   "Am I overtraining?"

4. "research" — Questions about sports science, evidence, or general training principles.
   Examples: "What does the research say about volume during a cut?",
   "Is training to failure necessary?", "What rep range is best for hypertrophy?"

Respond with ONLY the category name, nothing else."""


def classify_query(query: str) -> dict[str, str]:
    """Classify a query and return the type + model assignment.

    Returns:
        {"query_type": "...", "model": "..."}
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0,
        max_tokens=10,
    )

    query_type = response.choices[0].message.content.strip().lower()

    # Validate — default to programming if the model returns something unexpected
    if query_type not in MODEL_MAP:
        query_type = "programming"

    return {
        "query_type": query_type,
        "model": MODEL_MAP[query_type],
    }