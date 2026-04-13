"""Pydantic models for Hevy workout data."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SetType(str, Enum):
    NORMAL = "normal"
    WARMUP = "warmup"
    DROPSET = "dropset"
    FAILURE = "failure"


class ExerciseSet(BaseModel):
    """A single set within an exercise."""

    index: int
    set_type: SetType
    weight_kg: Optional[float] = None
    reps: Optional[int] = None
    distance_meters: Optional[float] = None
    duration_seconds: Optional[int] = None
    rpe: Optional[float] = None

    @property
    def weight_lbs(self) -> Optional[float]:
        if self.weight_kg is None:
            return None
        return round(self.weight_kg * 2.20462, 1)

    @property
    def estimated_1rm(self) -> Optional[float]:
        """Estimate 1RM using Epley formula."""
        if self.weight_kg is None or self.reps is None or self.reps == 0:
            return None
        if self.reps == 1:
            return self.weight_kg
        return round(self.weight_kg * (1 + self.reps / 30), 1)


class Exercise(BaseModel):
    """An exercise within a workout, containing multiple sets."""

    index: int
    title: str
    notes: str = ""
    exercise_template_id: str = ""
    superset_id: Optional[int] = None
    sets: list[ExerciseSet] = Field(default_factory=list)

    @property
    def working_sets(self) -> list[ExerciseSet]:
        """Return only non-warmup sets."""
        return [s for s in self.sets if s.set_type != SetType.WARMUP]

    @property
    def top_set(self) -> Optional[ExerciseSet]:
        """Return the heaviest working set."""
        working = self.working_sets
        if not working:
            return None
        return max(working, key=lambda s: s.weight_kg or 0)

    @property
    def total_volume_kg(self) -> float:
        """Total volume (weight × reps) across working sets."""
        return sum(
            (s.weight_kg or 0) * (s.reps or 0)
            for s in self.working_sets
        )

    @property
    def num_working_sets(self) -> int:
        return len(self.working_sets)


class Workout(BaseModel):
    """A single workout session from Hevy."""

    id: str
    title: str = ""
    description: str = ""
    start_time: datetime
    end_time: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    exercises: list[Exercise] = Field(default_factory=list)

    @property
    def duration_minutes(self) -> Optional[float]:
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return round(delta.total_seconds() / 60, 1)

    @property
    def total_volume_kg(self) -> float:
        return sum(e.total_volume_kg for e in self.exercises)

    @property
    def total_working_sets(self) -> int:
        return sum(e.num_working_sets for e in self.exercises)

    @property
    def exercise_names(self) -> list[str]:
        return [e.title for e in self.exercises]


class ExerciseTemplate(BaseModel):
    """An exercise template from the Hevy library."""

    id: str
    title: str
    type: str = ""  # e.g., "barbell", "dumbbell", "machine"
    primary_muscle_group: str = ""
    secondary_muscle_groups: list[str] = Field(default_factory=list)
    is_custom: bool = False