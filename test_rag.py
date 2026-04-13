"""Step 2 test — ingest knowledge docs, test retrieval, and run nutrition checker.

Usage:
    pip install chromadb openai
    python test_rag.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

from src.tools.rag import ingest_directory, retrieve, format_context
from src.tools.nutrition import check_nutrition_constraints, adjust_volume_for_phase
from src.tools.hevy import HevyClient
from src.tools.volume_calc import compute_weekly_volume
from src.models.program import UserProfile, TrainingPhase


def main() -> None:
    print("=" * 60)
    print("IronAgent — Step 2: RAG + Nutrition Test")
    print("=" * 60)

    # ── 1. Ingest knowledge docs ───────────────────────────────────
    knowledge_dir = Path(__file__).parent / "src" / "data" / "knowledge"
    print(f"\n📚 Ingesting docs from {knowledge_dir}...")
    results = ingest_directory(knowledge_dir)
    for filename, n_chunks in results.items():
        print(f"   ✅ {filename}: {n_chunks} chunks embedded")

    # ── 2. Test retrieval ──────────────────────────────────────────
    test_queries = [
        "How much volume should I do for chest per week?",
        "Should I reduce training volume during a cut?",
        "My bench press has stalled, what should I do?",
        "What rep range is best for hypertrophy?",
    ]

    print("\n🔍 Testing retrieval:")
    for query in test_queries:
        print(f"\n   Q: {query}")
        results = retrieve(query, n_results=2)
        for r in results:
            preview = r["text"][:120].replace("\n", " ")
            print(f"   → [{r['source']}] (score: {r['score']}) {preview}...")

    # ── 3. Test nutrition checker ──────────────────────────────────
    print("\n" + "=" * 60)
    print("🍽️  Nutrition Constraint Check")
    print("=" * 60)

    # Your current profile
    profile = UserProfile(
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

    print(f"\n   Phase: {profile.phase.value}")
    print(f"   Calories: {profile.calories}")
    print(f"   Protein: {profile.protein_g}g ({profile.protein_g / profile.bodyweight_lbs:.2f}g/lb)")

    constraints = check_nutrition_constraints(profile)
    print(f"\n   Volume modifier: {constraints['volume_modifier']:+.0%}")
    print(f"   Intensity: {constraints['intensity_note']}")
    print(f"   Priority: {constraints['priority']}")

    if constraints["warnings"]:
        print("\n   ⚠️  Warnings:")
        for w in constraints["warnings"]:
            print(f"      • {w}")

    if constraints["recommendations"]:
        print("\n   ✅ Good:")
        for r in constraints["recommendations"]:
            print(f"      • {r}")

    # ── 4. Phase-adjusted volume targets ───────────────────────────
    print("\n" + "=" * 60)
    print("📊 Phase-Adjusted Volume Targets (cutting)")
    print("=" * 60)

    with HevyClient() as client:
        templates = client.build_template_lookup()
        recent = client.get_recent_workouts(days=28)
        volume_stats = compute_weekly_volume(recent, templates, period_days=28)

    adjusted = adjust_volume_for_phase(volume_stats, profile)
    print()
    for a in adjusted:
        icon = {"under": "🔴", "over": "🟡", "on_target": "🟢"}[a["status"]]
        print(f"   {icon} {a['muscle_group']:12s}  {a['current_sets']:5.1f} → target: {a['target_range']} sets/wk")
        if a["status"] != "on_target":
            print(f"      └─ {a['action']}")

    print("\n" + "=" * 60)
    print("Step 2 complete! RAG + Nutrition working. ✅")
    print("Next: LangGraph agent with MoM router.")
    print("=" * 60)


if __name__ == "__main__":
    main()