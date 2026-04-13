"""Nutrition constraint checker — adjusts training recommendations based on diet phase."""

from __future__ import annotations

from src.models.program import TrainingPhase, UserProfile, VolumeStats


# Evidence-based volume adjustment factors by training phase.
# During a deficit, recovery capacity drops → reduce volume toward MEV.
# During a surplus, recovery is enhanced → can push toward MRV.

PHASE_ADJUSTMENTS: dict[TrainingPhase, dict] = {
    TrainingPhase.CUT: {
        "volume_modifier": -0.20,  # reduce volume ~20%
        "intensity_note": "Maintain intensity (weight on bar) to preserve muscle. Cut volume, not load.",
        "frequency_note": "Can reduce frequency by 1 session/week if recovery is impaired.",
        "priority": "Prioritize compound lifts. Drop isolation volume first.",
        "protein_minimum_per_lb": 1.0,  # g/lb bodyweight
        "max_deficit_kcal": 750,
    },
    TrainingPhase.BULK: {
        "volume_modifier": 0.15,  # can push volume ~15% above baseline
        "intensity_note": "Progressive overload is the priority. Push for PRs.",
        "frequency_note": "Higher frequency (2x/muscle/week) supports growth.",
        "priority": "Add volume to lagging muscle groups first.",
        "protein_minimum_per_lb": 0.8,
        "max_deficit_kcal": 0,
    },
    TrainingPhase.MAINTAIN: {
        "volume_modifier": 0.0,
        "intensity_note": "Maintain current intensity and volume.",
        "frequency_note": "Current frequency is fine if recovery is adequate.",
        "priority": "Address any gaps in muscle group coverage.",
        "protein_minimum_per_lb": 0.8,
        "max_deficit_kcal": 0,
    },
    TrainingPhase.PEAK: {
        "volume_modifier": -0.30,  # taper volume significantly
        "intensity_note": "Keep heavy singles/doubles. Cut accessory volume.",
        "frequency_note": "Reduce to 3-4 sessions in final week.",
        "priority": "CNS freshness. Minimize fatigue, maximize expression.",
        "protein_minimum_per_lb": 1.0,
        "max_deficit_kcal": 0,
    },
}


def check_nutrition_constraints(profile: UserProfile) -> dict:
    """Analyze the user's nutrition context and return training constraints.

    Returns a dict with:
        - phase_adjustments: volume/intensity/frequency guidance
        - warnings: any nutritional red flags
        - recommendations: specific nutrition-related advice
    """
    phase_info = PHASE_ADJUSTMENTS[profile.phase]
    warnings: list[str] = []
    recommendations: list[str] = []

    # Protein check
    if profile.protein_g and profile.bodyweight_lbs:
        protein_per_lb = profile.protein_g / profile.bodyweight_lbs
        minimum = phase_info["protein_minimum_per_lb"]
        if protein_per_lb < minimum:
            warnings.append(
                f"Protein intake ({profile.protein_g}g) is below the recommended "
                f"{minimum}g/lb for {profile.phase.value} phase. "
                f"Target: {int(profile.bodyweight_lbs * minimum)}g+."
            )
        else:
            recommendations.append(
                f"Protein intake ({profile.protein_g}g = "
                f"{protein_per_lb:.2f}g/lb) looks adequate for {profile.phase.value} phase."
            )

    # Deficit severity check (cutting only)
    if profile.phase == TrainingPhase.CUT and profile.calories:
        if profile.bodyweight_lbs:
            # Rough TDEE estimate: bodyweight_lbs × 15
            estimated_tdee = profile.bodyweight_lbs * 15
            deficit = estimated_tdee - profile.calories
            max_deficit = phase_info["max_deficit_kcal"]
            if deficit > max_deficit:
                warnings.append(
                    f"Estimated deficit (~{int(deficit)} kcal) is aggressive. "
                    f"Recommended max: {max_deficit} kcal for muscle preservation. "
                    f"Recovery capacity will be impaired — reduce training volume."
                )
            elif deficit > 0:
                recommendations.append(
                    f"Moderate deficit (~{int(deficit)} kcal). "
                    f"Recovery should be manageable with slight volume reduction."
                )

    # Injury interaction
    if profile.injuries:
        for injury in profile.injuries:
            warnings.append(
                f"Active injury: {injury}. Avoid movements that aggravate it. "
                f"Maintain volume for unaffected muscle groups."
            )

    return {
        "phase": profile.phase.value,
        "volume_modifier": phase_info["volume_modifier"],
        "intensity_note": phase_info["intensity_note"],
        "frequency_note": phase_info["frequency_note"],
        "priority": phase_info["priority"],
        "warnings": warnings,
        "recommendations": recommendations,
    }


def adjust_volume_for_phase(
    volume_stats: list[VolumeStats],
    profile: UserProfile,
) -> list[dict]:
    """Take raw volume stats and adjust targets based on training phase.

    Returns a list of dicts with adjusted recommendations per muscle group.
    """
    modifier = PHASE_ADJUSTMENTS[profile.phase]["volume_modifier"]
    adjusted: list[dict] = []

    from src.tools.volume_calc import VOLUME_LANDMARKS

    for vs in volume_stats:
        landmarks = VOLUME_LANDMARKS.get(vs.muscle_group, {"mev": 8, "mrv": 20})
        mev = landmarks["mev"]
        mrv = landmarks["mrv"]

        # Adjust target range based on phase
        if profile.phase == TrainingPhase.CUT:
            # During a cut, target MEV to slightly above
            target_low = mev
            target_high = mev + int((mrv - mev) * 0.3)
        elif profile.phase == TrainingPhase.BULK:
            # During a bulk, target mid-range to MRV
            target_low = mev + int((mrv - mev) * 0.4)
            target_high = mrv
        elif profile.phase == TrainingPhase.PEAK:
            # Peaking: drop to MEV or below
            target_low = int(mev * 0.7)
            target_high = mev
        else:
            # Maintain: target mid-range
            target_low = mev
            target_high = mev + int((mrv - mev) * 0.6)

        status = "on_target"
        if vs.weekly_sets < target_low:
            status = "under"
        elif vs.weekly_sets > target_high:
            status = "over"

        adjusted.append({
            "muscle_group": vs.muscle_group,
            "current_sets": vs.weekly_sets,
            "target_range": f"{target_low}-{target_high}",
            "status": status,
            "trend": vs.trend,
            "action": _get_action(status, vs, profile),
        })

    return adjusted


def _get_action(
    status: str,
    vs: VolumeStats,
    profile: UserProfile,
) -> str:
    """Generate a brief action recommendation."""
    if status == "under":
        if profile.phase == TrainingPhase.CUT:
            return (
                f"Add {max(1, int(10 - vs.weekly_sets))} sets/wk if recovery allows. "
                f"Prioritize compounds over isolation."
            )
        return f"Increase by 2-3 sets/wk. Consider adding a second weekly session."

    if status == "over":
        if profile.phase == TrainingPhase.CUT:
            return "Reduce volume to conserve recovery. Drop lowest-priority isolation sets."
        return "At or above MRV. Monitor for overreaching signals."

    return "Volume is on target. Maintain current approach."