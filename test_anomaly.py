"""Test anomaly detection — z-score + Isolation Forest on your Hevy data.

Usage:
    pip install scikit-learn numpy
    python test_anomaly.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

from src.tools.hevy import HevyClient
from src.tools.anomaly import (
    build_weekly_volume_matrix,
    zscore_anomalies,
    isolation_forest_anomalies,
    detect_anomalies,
)


def main() -> None:
    print("=" * 60)
    print("IronAgent — Anomaly Detection Test")
    print("=" * 60)

    with HevyClient() as client:
        workouts = client.get_all_workouts(max_pages=10)
        templates = client.build_template_lookup()

    print(f"\n📦 Loaded {len(workouts)} total workouts")

    # ── 1. Show the weekly volume matrix ───────────────────────────
    matrix, weeks, muscles = build_weekly_volume_matrix(
        workouts, templates, n_weeks=12
    )

    print(f"\n📊 Weekly volume matrix ({len(weeks)} weeks × {len(muscles)} muscles):")
    print(f"   {'Week':<12s}", end="")
    for m in muscles:
        print(f"{m[:6]:>7s}", end="")
    print()

    for i, week in enumerate(weeks):
        print(f"   {week:<12s}", end="")
        for j in range(len(muscles)):
            val = matrix[i, j]
            print(f"{val:7.0f}", end="")
        print()

    # ── 2. Z-score anomalies ───────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("🔬 Z-Score Anomalies (per muscle vs your personal baseline)")
    print(f"{'─' * 60}")

    z_results = zscore_anomalies(workouts, templates, n_weeks=12)

    if z_results:
        for a in z_results:
            icon = {"high": "🔴", "moderate": "🟡", "low": "🟠"}[a["severity"]]
            print(f"   {icon} {a['description']}")
    else:
        print("   ✅ No individual muscle anomalies detected.")

    # ── 3. Isolation Forest ────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("🌲 Isolation Forest (overall training pattern)")
    print(f"{'─' * 60}")

    iso_result = isolation_forest_anomalies(workouts, templates, n_weeks=12)

    icon = "🔴" if iso_result["is_anomalous"] else "✅"
    print(f"   {icon} {iso_result['description']}")

    if iso_result["contributing_muscles"]:
        print(f"\n   Top contributing muscles:")
        for c in iso_result["contributing_muscles"]:
            arrow = "↓" if c["direction"] == "below" else "↑"
            print(
                f"      {arrow} {c['muscle']}: {c['current']:.0f} sets "
                f"(avg: {c['average']:.1f})"
            )

    # ── 4. Combined report ─────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("📋 Combined Anomaly Report")
    print(f"{'─' * 60}")

    combined = detect_anomalies(workouts, templates, n_weeks=12)
    print(f"   {combined['summary']}")

    print(f"\n{'=' * 60}")
    print("Anomaly detection working. ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()