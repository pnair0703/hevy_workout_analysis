"""Pydantic models for user profile and agent recommendations."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TrainingPhase(str, Enum):
    CUT = "cut"
    BULK = "bulk"
    MAINTAIN = "maintain"
    PEAK = "peak"


class UserProfile(BaseModel):
    """User's current training context."""

    name: str = ""
    bodyweight_lbs: Optional[float] = None
    calories: Optional[int] = None
    protein_g: Optional[int] = None
    carbs_g: Optional[int] = None
    fat_g: Optional[int] = None
    phase: TrainingPhase = TrainingPhase.MAINTAIN
    training_days_per_week: int = 5
    injuries: list[str] = Field(default_factory=list)
    notes: str = ""


class ExerciseRx(BaseModel):
    """A single exercise prescription within a recommendation."""

    exercise: str
    sets: int
    rep_range: str = Field(description="e.g. '8-12' or '3-5'")
    rpe: float = Field(ge=5.0, le=10.0)
    rest_seconds: int = 120
    progression_scheme: str = Field(
        description="e.g. 'Add 5 lbs when you hit top of rep range at target RPE'"
    )
    notes: str = ""


class Recommendation(BaseModel):
    """A structured training recommendation from the agent."""

    summary: str = Field(description="1-2 sentence overview of the recommendation")
    exercises: list[ExerciseRx] = Field(default_factory=list)
    rationale: str = Field(description="Explanation of why this was recommended")
    citations: list[str] = Field(
        default_factory=list,
        description="Sources from sports science literature",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Safety flags, injury concerns, deficit considerations",
    )
    gaps_identified: list[str] = Field(
        default_factory=list,
        description="Muscle groups or movement patterns that are undertrained",
    )


class VolumeStats(BaseModel):
    """Weekly volume statistics for a muscle group."""

    muscle_group: str
    weekly_sets: float = Field(description="Average working sets per week")
    trend: str = Field(description="'increasing', 'stable', 'decreasing', or 'insufficient_data'")
    meets_minimum: bool = Field(description="Whether volume meets minimum effective volume (~10 sets/week)")
    meets_maximum: bool = Field(description="Whether volume is at or above MRV (~20+ sets/week)")
    top_exercises: list[str] = Field(
        default_factory=list,
        description="Most frequently used exercises for this muscle group",
    )


class TrainingAnalysis(BaseModel):
    """Full analysis of a user's recent training."""

    period_days: int
    total_workouts: int
    avg_duration_minutes: Optional[float] = None
    volume_by_muscle: list[VolumeStats] = Field(default_factory=list)
    gaps: list[str] = Field(
        default_factory=list,
        description="Muscle groups below minimum effective volume",
    )
    overreaching_signals: list[str] = Field(
        default_factory=list,
        description="Signs of accumulated fatigue",
    )
    pr_highlights: list[str] = Field(
        default_factory=list,
        description="Recent personal records or notable improvements",
    )