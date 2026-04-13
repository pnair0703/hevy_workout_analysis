"""Anomaly detection — personalized gap detection using training history.

Two complementary approaches:
    1. Z-score per muscle group: flags individual muscles that deviate
       from the user's personal baseline.
    2. Isolation Forest on the full weekly profile: detects abnormal
       overall training patterns across all muscle groups simultaneously.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest

from src.models.workout import ExerciseTemplate, Workout
from src.tools.volume_calc import normalize_muscle_group, VOLUME_LANDMARKS


# Canonical muscle groups we track
TRACKED_MUSCLES = sorted(VOLUME_LANDMARKS.keys())


def build_weekly_volume_matrix(
    workouts: list[Workout],
    template_lookup: dict[str, ExerciseTemplate],
    n_weeks: int = 20,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Convert workout history into a (n_weeks × n_muscles) matrix.

    Each row is one week. Each column is weekly working sets for a muscle group.

    Returns:
        matrix: numpy array of shape (n_weeks, n_muscles)
        week_labels: list of "YYYY-Www" labels for each row
        muscle_labels: list of muscle group names for each column
    """
    now = datetime.now(timezone.utc)
    muscle_labels = TRACKED_MUSCLES

    # Bucket workouts into weeks
    weekly_sets: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for workout in workouts:
        # Calculate which week this workout belongs to
        days_ago = (now - workout.start_time).days
        if days_ago < 0 or days_ago >= n_weeks * 7:
            continue
        week_idx = days_ago // 7
        week_label = (now - timedelta(days=week_idx * 7)).strftime("%Y-W%W")

        for exercise in workout.exercises:
            template = template_lookup.get(exercise.exercise_template_id)
            if template and template.primary_muscle_group:
                muscle = normalize_muscle_group(template.primary_muscle_group)
            else:
                from src.tools.volume_calc import _infer_muscle_group
                muscle = _infer_muscle_group(exercise.title)

            if muscle in muscle_labels:
                weekly_sets[week_label][muscle] += exercise.num_working_sets

    # Build sorted week labels (most recent first)
    all_weeks = sorted(weekly_sets.keys(), reverse=True)[:n_weeks]

    # Build matrix
    matrix = np.zeros((len(all_weeks), len(muscle_labels)))
    for i, week in enumerate(all_weeks):
        for j, muscle in enumerate(muscle_labels):
            matrix[i, j] = weekly_sets[week].get(muscle, 0)

    return matrix, all_weeks, muscle_labels


# ── Z-Score Anomaly Detection ───────────────────────────────────────

def zscore_anomalies(
    workouts: list[Workout],
    template_lookup: dict[str, ExerciseTemplate],
    n_weeks: int = 20,
    threshold: float = 1.5,
) -> list[dict]:
    """Detect per-muscle anomalies using z-scores against personal history.

    Compares the most recent week against the user's rolling baseline.
    A z-score below -threshold means the muscle is significantly undertrained
    relative to the user's own patterns.

    Returns list of:
        {"muscle": ..., "current_sets": ..., "mean": ..., "std": ...,
         "z_score": ..., "severity": "low|moderate|high"}
    """
    matrix, week_labels, muscle_labels = build_weekly_volume_matrix(
        workouts, template_lookup, n_weeks
    )

    if len(matrix) < 3:
        return []  # not enough history

    # Current week = row 0 (most recent)
    current = matrix[0]

    # Baseline = rows 1 onward (historical)
    history = matrix[1:]

    anomalies: list[dict] = []

    for j, muscle in enumerate(muscle_labels):
        hist_values = history[:, j]
        mean = float(np.mean(hist_values))
        std = float(np.std(hist_values))

        if std < 0.5:
            # Very consistent history — use a minimum std to avoid
            # division issues and false positives from tiny variations
            std = max(std, 1.0)

        current_val = float(current[j])
        z = (current_val - mean) / std

        if z < -threshold:
            if z < -2.5:
                severity = "high"
            elif z < -2.0:
                severity = "moderate"
            else:
                severity = "low"

            anomalies.append({
                "muscle": muscle,
                "current_sets": round(current_val, 1),
                "personal_mean": round(mean, 1),
                "personal_std": round(std, 1),
                "z_score": round(z, 2),
                "severity": severity,
                "description": (
                    f"{muscle}: {current_val:.0f} sets this week vs "
                    f"your average of {mean:.1f} ± {std:.1f} "
                    f"(z = {z:.2f}, {severity} deviation)"
                ),
            })

    # Sort by severity (most anomalous first)
    anomalies.sort(key=lambda x: x["z_score"])
    return anomalies


# ── Isolation Forest Anomaly Detection ──────────────────────────────

def isolation_forest_anomalies(
    workouts: list[Workout],
    template_lookup: dict[str, ExerciseTemplate],
    n_weeks: int = 20,
    contamination: float = 0.15,
) -> dict:
    """Detect abnormal overall training patterns using Isolation Forest.

    Trains on the full weekly volume profile (all muscle groups as features).
    Checks whether the most recent week's pattern is an outlier.

    Returns:
        {
            "is_anomalous": bool,
            "anomaly_score": float (-1 to 0, more negative = more anomalous),
            "description": str,
            "contributing_muscles": list of muscles most different from centroid
        }
    """
    matrix, week_labels, muscle_labels = build_weekly_volume_matrix(
        workouts, template_lookup, n_weeks
    )

    if len(matrix) < 5:
        return {
            "is_anomalous": False,
            "anomaly_score": 0.0,
            "description": "Not enough training history for pattern detection (need 5+ weeks).",
            "contributing_muscles": [],
        }

    # Fit Isolation Forest on all weeks
    iso = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
    )
    iso.fit(matrix)

    # Score the most recent week
    current = matrix[0].reshape(1, -1)
    prediction = iso.predict(current)[0]  # 1 = normal, -1 = anomaly
    score = iso.score_samples(current)[0]  # more negative = more anomalous

    is_anomalous = prediction == -1

    # Find which muscles contribute most to the anomaly
    # Compare current week to the centroid (mean of all weeks)
    centroid = np.mean(matrix, axis=0)
    deviations = current[0] - centroid
    abs_deviations = np.abs(deviations)

    # Top 3 muscles driving the anomaly
    top_indices = np.argsort(abs_deviations)[::-1][:3]
    contributing = []
    for idx in top_indices:
        muscle = muscle_labels[idx]
        diff = deviations[idx]
        direction = "below" if diff < 0 else "above"
        contributing.append({
            "muscle": muscle,
            "current": float(current[0][idx]),
            "average": float(centroid[idx]),
            "direction": direction,
        })

    if is_anomalous:
        desc = (
            f"This week's training pattern is abnormal (score: {score:.3f}). "
            f"Biggest deviations: "
            + ", ".join(
                f"{c['muscle']} ({c['current']:.0f} vs avg {c['average']:.1f} sets)"
                for c in contributing
            )
        )
    else:
        desc = f"Training pattern looks normal (score: {score:.3f})."

    return {
        "is_anomalous": is_anomalous,
        "anomaly_score": round(float(score), 4),
        "description": desc,
        "contributing_muscles": contributing,
    }


# ── Combined Detection ──────────────────────────────────────────────

def detect_anomalies(
    workouts: list[Workout],
    template_lookup: dict[str, ExerciseTemplate],
    n_weeks: int = 20,
) -> dict:
    """Run both z-score and Isolation Forest detection.

    Returns a combined report with individual muscle flags
    and overall pattern assessment.
    """
    z_results = zscore_anomalies(workouts, template_lookup, n_weeks)
    iso_result = isolation_forest_anomalies(workouts, template_lookup, n_weeks)

    return {
        "individual_anomalies": z_results,
        "pattern_anomaly": iso_result,
        "summary": _build_summary(z_results, iso_result),
    }


def _build_summary(z_results: list[dict], iso_result: dict) -> str:
    """Build a human-readable summary of all anomaly findings."""
    parts: list[str] = []

    if z_results:
        high = [a for a in z_results if a["severity"] == "high"]
        moderate = [a for a in z_results if a["severity"] == "moderate"]

        if high:
            muscles = ", ".join(a["muscle"] for a in high)
            parts.append(f"Critically low volume: {muscles}")
        if moderate:
            muscles = ", ".join(a["muscle"] for a in moderate)
            parts.append(f"Below personal baseline: {muscles}")

    if iso_result.get("is_anomalous"):
        parts.append(f"Overall training pattern is abnormal this week")

    if not parts:
        return "Training looks consistent with your personal history."

    return ". ".join(parts) + "."