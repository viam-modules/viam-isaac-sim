"""Seeded joint-target nudges for perturbed (DART-style) demonstrations.

During a recorded episode the arm is driven toward its commanded target plus a
nudge offset, while the episode keeps recording the clean commanded target.
Each nudge follows a half-sine, so it fades in from and back out to exactly
zero and the arm returns to its commanded path.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True)
class NudgeConfig:
    # One peak for every joint, or one per joint (e.g. shaped to the policy's
    # per-joint error).
    amplitude_deg: float | Sequence[float]
    interval_s: float
    duration_s: float = 0.6

    def __post_init__(self) -> None:
        for name in ("interval_s", "duration_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"nudge {name} must be finite and positive")
            object.__setattr__(self, name, value)
        if isinstance(self.amplitude_deg, (int, float)):
            amplitude = float(self.amplitude_deg)
            if not math.isfinite(amplitude) or amplitude <= 0.0:
                raise ValueError("nudge amplitude_deg must be finite and positive")
        else:
            amplitude = tuple(float(value) for value in self.amplitude_deg)
            if (
                not amplitude
                or not all(math.isfinite(value) and value >= 0.0 for value in amplitude)
                or max(amplitude) <= 0.0
            ):
                raise ValueError(
                    "nudge amplitude_deg values must be finite, non-negative and "
                    "not all zero"
                )
        object.__setattr__(self, "amplitude_deg", amplitude)

    def amplitudes_deg(self, joint_count: int) -> list[float]:
        if isinstance(self.amplitude_deg, float):
            return [self.amplitude_deg] * joint_count
        if len(self.amplitude_deg) != joint_count:
            raise ValueError(
                f"nudge amplitude_deg has {len(self.amplitude_deg)} values for "
                f"{joint_count} joints"
            )
        return list(self.amplitude_deg)


class NudgeSchedule:
    """Per-joint nudge offsets in radians, advanced in simulator time.

    The gap before each nudge is drawn uniformly from [0.5, 1.5] * interval_s,
    and each joint's peak offset uniformly from +/- its amplitude_deg. The stream
    is seeded per episode so a seed always produces the same nudges.
    """

    def __init__(self, config: NudgeConfig, seed: int, joint_count: int) -> None:
        self._config = config
        self._joint_count = joint_count
        self._rng = random.Random(f"nudge:{int(seed)}")
        self._amplitude_rad = [
            math.radians(value) for value in config.amplitudes_deg(joint_count)
        ]
        self._wait_s = self._next_gap()
        self._elapsed_s: float | None = None
        self._peak = [0.0] * joint_count

    def _next_gap(self) -> float:
        return self._rng.uniform(0.5, 1.5) * self._config.interval_s

    def step(self, dt: float) -> list[float]:
        if self._elapsed_s is None:
            self._wait_s -= dt
            if self._wait_s > 0.0:
                return [0.0] * self._joint_count
            self._peak = [
                self._rng.uniform(-amplitude, amplitude)
                for amplitude in self._amplitude_rad
            ]
            self._elapsed_s = 0.0
        else:
            self._elapsed_s += dt
        if self._elapsed_s >= self._config.duration_s:
            self._elapsed_s = None
            self._wait_s = self._next_gap()
            return [0.0] * self._joint_count
        scale = math.sin(math.pi * self._elapsed_s / self._config.duration_s)
        return [peak * scale for peak in self._peak]
