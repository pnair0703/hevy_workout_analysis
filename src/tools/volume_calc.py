"""Volume calculator tool — analyzes workout data for training insights."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.models.workout import ExerciseTemplate, Workout
from src.models.program import VolumeStats, TrainingAnalysis


# ── Muscle group mapping ────────────────────────────────────────────
# Maps Hevy's primary_muscle_group strings to canonical groups.
# Expand this as you see new values from your template library.

MUSCLE_GROUP_MAP: dict[str, str] = {
    "chest": "chest",
    "back": "back",
    "lats": "back",
    "upper_back": "back",
    "traps": "back",
    "shoulders": "shoulders",
    "front_delt": "shoulders",
    "side_delt": "shoulders",
    "rear_delt": "shoulders",
    "biceps": "biceps",
    "triceps": "triceps",
    "forearms": "forearms",
    "quadriceps": "quads",
    "quads": "quads",
    "hamstrings": "hamstrings",
    "glutes": "glutes",
    "calves": "calves",
    "abs": "abs",
    "core": "abs",
    "abdominals": "abs",
    "lower_back": "back",
    "abductors": "glutes",
    "adductors": "glutes",
    "cardio": "other",
    "other": "other"
}

# Approximate minimum effective volume and max recoverable volume
# (working sets per muscle group per week).
# Based on Schoenfeld/Israetel recommendations.

VOLUME_LANDMARKS: dict[str, dict[str, int]] = {
    "chest":       {"mev": 10, "mrv": 22},
    "back":        {"mev": 10, "mrv": 23},
    "shoulders":   {"mev": 8,  "mrv": 22},
    "biceps":      {"mev": 8,  "mrv": 20},
    "triceps":     {"mev": 6,  "mrv": 18},
    "quads":       {"mev": 8,  "mrv": 20},
    "hamstrings":  {"mev": 6,  "mrv": 16},
    "glutes":      {"mev": 4,  "mrv": 16},
    "calves":      {"mev": 8,  "mrv": 16},
    "abs":         {"mev": 0,  "mrv": 16},
    "forearms":    {"mev": 2,  "mrv": 12},
}


def normalize_muscle_group(raw: str) -> str:
    """Map a Hevy muscle group string to a canonical name."""
    return MUSCLE_GROUP_MAP.get(raw.lower().strip(), raw.lower().strip())


def compute_weekly_volume(
    workouts: list[Workout],
    template_lookup: dict[str, ExerciseTemplate],
    period_days: int = 28,
) -> list[VolumeStats]:
    """Compute average weekly sets per muscle group over the given period.

    Counts working sets only (excludes warmup sets). Uses the exercise
    template's primary_muscle_group for classification.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)
    recent = [w for w in workouts if w.start_time >= cutoff]
    weeks = max(period_days / 7, 1)

    # Accumulate sets per muscle group
    sets_by_muscle: dict[str, int] = defaultdict(int)
    exercises_by_muscle: dict[str, list[str]] = defaultdict(list)

    for workout in recent:
        for exercise in workout.exercises:
            template = template_lookup.get(exercise.exercise_template_id)
            if template and template.primary_muscle_group:
                muscle = normalize_muscle_group(template.primary_muscle_group)
            else:
                # Fallback: try to infer from exercise title
                muscle = _infer_muscle_group(exercise.title)

            n_working = exercise.num_working_sets
            sets_by_muscle[muscle] += n_working

            if exercise.title not in exercises_by_muscle[muscle]:
                exercises_by_muscle[muscle].append(exercise.title)

    # Build VolumeStats for each muscle group
    results: list[VolumeStats] = []
    all_muscles = set(list(VOLUME_LANDMARKS.keys()) + list(sets_by_muscle.keys()))

    for muscle in sorted(all_muscles):
        if muscle in ("other",):
            continue
        total_sets = sets_by_muscle.get(muscle, 0)
        weekly = round(total_sets / weeks, 1)
        landmarks = VOLUME_LANDMARKS.get(muscle, {"mev": 8, "mrv": 20})

        results.append(
            VolumeStats(
                muscle_group=muscle,
                weekly_sets=weekly,
                trend=_compute_trend(workouts, template_lookup, muscle, period_days),
                meets_minimum=weekly >= landmarks["mev"],
                meets_maximum=weekly >= landmarks["mrv"],
                top_exercises=exercises_by_muscle.get(muscle, [])[:5],
            )
        )

    return results


def compute_training_analysis(
    workouts: list[Workout],
    template_lookup: dict[str, ExerciseTemplate],
    period_days: int = 28,
) -> TrainingAnalysis:
    """Full training analysis over the given period."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)
    recent = [w for w in workouts if w.start_time >= cutoff]

    durations = [w.duration_minutes for w in recent if w.duration_minutes]
    avg_dur = round(sum(durations) / len(durations), 1) if durations else None

    volume_stats = compute_weekly_volume(workouts, template_lookup, period_days)

    gaps = [
        v.muscle_group for v in volume_stats
        if not v.meets_minimum and v.muscle_group in VOLUME_LANDMARKS
    ]

    overreaching = _detect_overreaching(workouts, template_lookup, period_days)

    return TrainingAnalysis(
        period_days=period_days,
        total_workouts=len(recent),
        avg_duration_minutes=avg_dur,
        volume_by_muscle=volume_stats,
        gaps=gaps,
        overreaching_signals=overreaching,
    )


def compute_e1rm_history(
    workouts: list[Workout],
    exercise_title: str,
    period_days: int = 90,
) -> list[dict]:
    """Return a time series of estimated 1RM for a given exercise.

    Returns list of {"date": ..., "e1rm_kg": ..., "e1rm_lbs": ...}
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)
    history: list[dict] = []

    for workout in sorted(workouts, key=lambda w: w.start_time):
        if workout.start_time < cutoff:
            continue
        for exercise in workout.exercises:
            if exercise.title.lower() != exercise_title.lower():
                continue
            top = exercise.top_set
            if top and top.estimated_1rm:
                history.append({
                    "date": workout.start_time.strftime("%Y-%m-%d"),
                    "e1rm_kg": top.estimated_1rm,
                    "e1rm_lbs": round(top.estimated_1rm * 2.20462, 1),
                })
    return history


def compute_tonnage(
    workouts: list[Workout],
    period_days: int = 7,
) -> dict[str, float]:
    """Compute total tonnage (kg) per workout over the period."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)
    result: dict[str, float] = {}
    for w in workouts:
        if w.start_time >= cutoff:
            label = f"{w.start_time.strftime('%Y-%m-%d')} {w.title}"
            result[label] = round(w.total_volume_kg, 1)
    return result


# ── Private helpers ─────────────────────────────────────────────────

def _compute_trend(
    workouts: list[Workout],
    template_lookup: dict[str, ExerciseTemplate],
    muscle: str,
    period_days: int,
) -> str:
    """Compare volume in first half vs second half of the period."""
    now = datetime.now(timezone.utc)
    mid = now - timedelta(days=period_days / 2)
    cutoff = now - timedelta(days=period_days)

    first_half = 0
    second_half = 0

    for w in workouts:
        if w.start_time < cutoff:
            continue
        for ex in w.exercises:
            tmpl = template_lookup.get(ex.exercise_template_id)
            if tmpl:
                m = normalize_muscle_group(tmpl.primary_muscle_group)
            else:
                m = _infer_muscle_group(ex.title)
            if m != muscle:
                continue
            sets = ex.num_working_sets
            if w.start_time < mid:
                first_half += sets
            else:
                second_half += sets

    if first_half == 0 and second_half == 0:
        return "insufficient_data"
    if first_half == 0:
        return "increasing"
    ratio = second_half / first_half
    if ratio > 1.15:
        return "increasing"
    elif ratio < 0.85:
        return "decreasing"
    return "stable"


def _detect_overreaching(
    workouts: list[Workout],
    template_lookup: dict[str, ExerciseTemplate],
    period_days: int,
) -> list[str]:
    """Simple heuristic: volume up but performance (e1RM) flat/down."""
    signals: list[str] = []

    volume_stats = compute_weekly_volume(workouts, template_lookup, period_days)
    for vs in volume_stats:
        if vs.trend == "increasing" and vs.meets_maximum:
            signals.append(
                f"{vs.muscle_group}: volume increasing and at/above MRV — consider a deload"
            )

    return signals


def _infer_muscle_group(title: str) -> str:
    """Best-effort muscle group inference from exercise name."""
    t = title.lower()
    if any(k in t for k in ("bench", "chest", "fly", "pec")):
        return "chest"
    if any(k in t for k in ("row", "pull-up", "pullup", "lat ", "pulldown")):
        return "back"
    if any(k in t for k in ("squat", "leg press", "lunge", "leg ext")):
        return "quads"
    if any(k in t for k in ("deadlift", "rdl", "leg curl", "hamstring")):
        return "hamstrings"
    if any(k in t for k in ("shoulder", "ohp", "press", "lateral raise", "delt")):
        return "shoulders"
    if any(k in t for k in ("curl", "bicep")):
        return "biceps"
    if any(k in t for k in ("tricep", "pushdown", "skullcrusher", "dip")):
        return "triceps"
    if any(k in t for k in ("calf", "calves")):
        return "calves"
    if any(k in t for k in ("glute", "hip thrust")):
        return "glutes"
    if any(k in t for k in ("ab ", "crunch", "plank", "core")):
        return "abs"
    return "other"