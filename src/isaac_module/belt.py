"""A kinematic conveyor belt: props carried along a straight segment at a
fixed speed in simulator time, advanced on the sim thread every rendered step.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Vec3 = tuple[float, float, float]


class Belt:
    def __init__(self, start: Vec3, end: Vec3, speed_mps: float, wrap: bool) -> None:
        self.start = start
        self.end = end
        self.length_m = math.dist(start, end)
        if not math.isfinite(self.length_m) or self.length_m <= 0.0:
            raise ValueError("belt length must be positive: start and end differ")
        if not math.isfinite(speed_mps) or speed_mps < 0.0:
            raise ValueError("belt speed_mps must be finite and non-negative")
        self.speed_mps = speed_mps
        self.wrap = wrap
        # prop name -> distance from start along the belt, meters
        self.props: dict[str, float] = {}

    def position(self, at_m: float) -> Vec3:
        f = at_m / self.length_m
        return tuple(s + (e - s) * f for s, e in zip(self.start, self.end))  # type: ignore[return-value]

    def add(self, name: str, at_m: float) -> Vec3:
        if not math.isfinite(at_m) or not 0.0 <= at_m <= self.length_m:
            raise ValueError(f"belt at_m must be within [0, {self.length_m:g}]")
        self.props[name] = at_m
        return self.position(at_m)

    def load(self, names: Sequence[str], spacing_m: float | None) -> dict[str, Vec3]:
        """Replace the load with names placed from start, spacing_m apart
        (default: spread evenly over the belt)."""
        if spacing_m is None:
            spacing_m = self.length_m / max(1, len(names))
        if not math.isfinite(spacing_m) or spacing_m < 0.0:
            raise ValueError("belt spacing_m must be finite and non-negative")
        self.props = {}
        return {
            name: self.add(name, (i * spacing_m) % self.length_m)
            for i, name in enumerate(names)
        }

    def advance(self, dt: float) -> dict[str, Vec3]:
        """Move every prop dt seconds along; return each one's new position.
        Without wrap, a prop that reaches the end stops there and leaves."""
        moved: dict[str, Vec3] = {}
        if dt <= 0.0 or self.speed_mps == 0.0:
            return moved
        for name, at_m in list(self.props.items()):
            at_m += self.speed_mps * dt
            if at_m >= self.length_m:
                if self.wrap:
                    at_m %= self.length_m
                else:
                    del self.props[name]
                    moved[name] = self.end
                    continue
            self.props[name] = at_m
            moved[name] = self.position(at_m)
        return moved
