"""Step 3 test — run queries through the full agent pipeline.

Tests all 4 query types to verify MoM routing works:
    1. lookup    → skips LLM, direct data answer
    2. programming → gpt-4o-mini
    3. coaching  → gpt-4o (deeper reasoning)
    4. research  → gpt-4o-mini (RAG summarization)

Usage:
    pip install langgraph langchain-core langchain-openai
    python test_agent.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

from src.agent.graph import run_agent


def main() -> None:
    print("=" * 60)
    print("IronAgent — Step 3: Full Agent + MoM Router Test")
    print("=" * 60)

    test_queries = [
        # Type 1: Lookup (should skip LLM → $0)
        "What did I do in my last workout?",

        # Type 2: Programming (should route to gpt-4o-mini)
        "My chest volume is way too low. What exercises and sets should I add?",

        # Type 3: Coaching (should route to gpt-4o)
        "My incline bench has stalled at 200 lbs and I'm cutting at 2500 cal. What should I adjust?",

        # Type 4: Research (should route to gpt-4o-mini + RAG)
        "What does the research say about training volume during a caloric deficit?",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─' * 60}")
        print(f"Query {i}: {query}")
        print(f"{'─' * 60}")

        result = run_agent(query)

        print(f"\n🏷️  Route: {result.get('query_type', '?')} → {result.get('model_used', '?')}")
        print(f"\n💬 Response:\n{result.get('answer', 'No answer')}")

    print(f"\n{'=' * 60}")
    print("Step 3 complete! Agent + MoM router working. ✅")
    print("Next: eval harness (DeepEval).")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()