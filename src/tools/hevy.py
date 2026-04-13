"""Hevy API client for fetching workout data."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

from src.models.workout import Exercise, ExerciseSet, ExerciseTemplate, Workout


class HevyClient:
    """Client for the Hevy REST API (v1).

    Requires a Hevy Pro subscription. Get your API key at:
    https://hevy.com/settings?developer

    Set the HEVY_API_KEY environment variable or pass it directly.
    """

    BASE_URL = "https://api.hevyapp.com/v1"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("HEVY_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Hevy API key required. Set HEVY_API_KEY env var or pass api_key."
            )
        self._client = httpx.Client(
            base_url=self.BASE_URL,
            headers={"api-key": self.api_key, "accept": "application/json"},
            timeout=30.0,
        )

    # ── Core API calls ──────────────────────────────────────────────

    def get_workout_count(self) -> int:
        """Return total number of workouts logged."""
        resp = self._client.get("/workouts/count")
        resp.raise_for_status()
        return resp.json().get("workout_count", 0)

    def get_workouts(
        self,
        page: int = 1,
        page_size: int = 10,
    ) -> list[Workout]:
        """Fetch a page of workouts (most recent first)."""
        resp = self._client.get(
            "/workouts",
            params={"page": page, "pageSize": page_size},
        )
        resp.raise_for_status()
        data = resp.json()
        return [self._parse_workout(w) for w in data.get("workouts", [])]

    def get_all_workouts(self, max_pages: int = 50) -> list[Workout]:
        """Paginate through all workouts up to max_pages."""
        all_workouts: list[Workout] = []
        page = 1
        while page <= max_pages:
            try:
                batch = self.get_workouts(page=page, page_size=10)
            except Exception:
                break
            if not batch:
                break
            all_workouts.extend(batch)
            page += 1
        return all_workouts

    def get_workout(self, workout_id: str) -> Workout:
        """Fetch a single workout by ID."""
        resp = self._client.get(f"/workouts/{workout_id}")
        resp.raise_for_status()
        return self._parse_workout(resp.json())

    def get_exercise_templates(
        self,
        page: int = 1,
        page_size: int = 100,
    ) -> list[ExerciseTemplate]:
        """Fetch exercise templates (the exercise library)."""
        resp = self._client.get(
            "/exercise_templates",
            params={"page": page, "pageSize": page_size},
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            self._parse_template(t)
            for t in data.get("exercise_templates", [])
        ]

    def get_exercise_history(
        self,
        exercise_template_id: str,
        page: int = 1,
        page_size: int = 10,
    ) -> list[dict]:
        """Fetch history for a specific exercise template."""
        resp = self._client.get(
            f"/exercise_history/{exercise_template_id}",
            params={"page": page, "pageSize": page_size},
        )
        resp.raise_for_status()
        return resp.json().get("exercise_history", [])

    def get_routines(self, page: int = 1, page_size: int = 10) -> list[dict]:
        """Fetch saved routines."""
        resp = self._client.get(
            "/routines",
            params={"page": page, "pageSize": page_size},
        )
        resp.raise_for_status()
        return resp.json().get("routines", [])

    # ── Convenience methods ─────────────────────────────────────────

    def get_recent_workouts(self, days: int = 30) -> list[Workout]:
        """Get workouts from the last N days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        workouts: list[Workout] = []
        page = 1
        while True:
            batch = self.get_workouts(page=page, page_size=10)
            if not batch:
                break
            for w in batch:
                if w.start_time >= cutoff:
                    workouts.append(w)
                else:
                    return workouts
            page += 1
        return workouts

    def build_template_lookup(self) -> dict[str, ExerciseTemplate]:
        """Build a mapping of template_id -> ExerciseTemplate.

        Useful for resolving muscle groups from workout exercises.
        """
        lookup: dict[str, ExerciseTemplate] = {}
        page = 1
        while True:
            templates = self.get_exercise_templates(page=page, page_size=100)
            if not templates:
                break
            for t in templates:
                lookup[t.id] = t
            if len(templates) < 100:
                break
            page += 1
        return lookup

    # ── Parsing helpers ─────────────────────────────────────────────

    @staticmethod
    def _parse_workout(data: dict) -> Workout:
        exercises = []
        for ex in data.get("exercises", []):
            sets = []
            for s in ex.get("sets", []):
                sets.append(
                    ExerciseSet(
                        index=s.get("index", 0),
                        set_type=s.get("set_type", "normal"),
                        weight_kg=s.get("weight_kg"),
                        reps=s.get("reps"),
                        distance_meters=s.get("distance_meters"),
                        duration_seconds=s.get("duration_seconds"),
                        rpe=s.get("rpe"),
                    )
                )
            exercises.append(
                Exercise(
                    index=ex.get("index", 0),
                    title=ex.get("title", ""),
                    notes=ex.get("notes", ""),
                    exercise_template_id=ex.get("exercise_template_id", ""),
                    superset_id=ex.get("superset_id"),
                    sets=sets,
                )
            )
        return Workout(
            id=data.get("id", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            start_time=data.get("start_time", datetime.now().isoformat()),
            end_time=data.get("end_time"),
            updated_at=data.get("updated_at"),
            exercises=exercises,
        )

    @staticmethod
    def _parse_template(data: dict) -> ExerciseTemplate:
        return ExerciseTemplate(
            id=data.get("id", ""),
            title=data.get("title", ""),
            type=data.get("type", ""),
            primary_muscle_group=data.get("primary_muscle_group", ""),
            secondary_muscle_groups=data.get("secondary_muscle_groups", []),
            is_custom=data.get("is_custom", False),
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HevyClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()