"""Quick test — run this first to verify your Hevy API connection.

Usage:
    1. Copy .env.example to .env and fill in your HEVY_API_KEY
    2. pip install httpx pydantic python-dotenv
    3. python test_hevy.py
"""

import sys
from pathlib import Path

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

from src.tools.hevy import HevyClient
from src.tools.volume_calc import compute_weekly_volume, compute_training_analysis


def main() -> None:
    print("=" * 60)
    print("IronAgent — Hevy Connection Test")
    print("=" * 60)

    with HevyClient() as client:
        # 1. Check connection
        count = client.get_workout_count()
        print(f"\n✅ Connected! Total workouts logged: {count}")

        # 2. Pull recent workouts
        print("\n📋 Last 5 workouts:")
        workouts = client.get_workouts(page=1, page_size=5)
        for w in workouts:
            dur = f" ({w.duration_minutes}min)" if w.duration_minutes else ""
            print(f"  • {w.start_time.strftime('%Y-%m-%d')} — {w.title}{dur}")
            for ex in w.exercises:
                top = ex.top_set
                if top and top.weight_kg:
                    print(
                        f"      {ex.title}: {ex.num_working_sets} sets, "
                        f"top set {top.weight_lbs} lbs × {top.reps}"
                    )
                else:
                    print(f"      {ex.title}: {ex.num_working_sets} sets")

        # 3. Build template lookup and run volume analysis
        print("\n🔍 Building exercise template lookup...")
        templates = client.build_template_lookup()
        print(f"   Found {len(templates)} exercise templates")

        print("\n📊 Last 28 days — weekly volume by muscle group:")
        recent = client.get_recent_workouts(days=28)
        analysis = compute_training_analysis(recent, templates, period_days=28)

        print(f"   Workouts: {analysis.total_workouts}")
        if analysis.avg_duration_minutes:
            print(f"   Avg duration: {analysis.avg_duration_minutes} min")

        print()
        for vs in analysis.volume_by_muscle:
            bar = "█" * int(vs.weekly_sets)
            status = "✅" if vs.meets_minimum else "⚠️"
            print(f"   {status} {vs.muscle_group:12s} {vs.weekly_sets:5.1f} sets/wk {bar}")
            if vs.top_exercises:
                print(f"      └─ {', '.join(vs.top_exercises[:3])}")

        if analysis.gaps:
            print(f"\n🚨 Gaps (below minimum effective volume):")
            for g in analysis.gaps:
                print(f"   • {g}")

        if analysis.overreaching_signals:
            print(f"\n⚠️  Overreaching signals:")
            for s in analysis.overreaching_signals:
                print(f"   • {s}")

        print("\n" + "=" * 60)
        print("Step 1 complete! Your Hevy data is flowing. ✅")
        print("Next: we'll wire this into the LangGraph agent.")
        print("=" * 60)


if __name__ == "__main__":
    main()