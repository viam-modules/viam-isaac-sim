"""A kinematic conveyor belt: props carried along a straight segment at a
fixed speed in simulator time, advanced on the sim thread every rendered step.
Each prop rides at its own offset from the belt line, so a stack keeps its
shape, and may leave the belt at its own stop.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Vec3 = tuple[float, float, float]

_NO_OFFSET: Vec3 = (0.0, 0.0, 0.0)

DEFAULT_BELT = "default"


class Belt:
    def __init__(
        self,
        start: Vec3,
        end: Vec3,
        speed_mps: float,
        wrap: bool,
        orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        self.start = start
        self.end = end
        self.length_m = math.dist(start, end)
        if not math.isfinite(self.length_m) or self.length_m <= 0.0:
            raise ValueError("belt length must be positive: start and end differ")
        if not math.isfinite(speed_mps) or speed_mps < 0.0:
            raise ValueError("belt speed_mps must be finite and non-negative")
        self.speed_mps = speed_mps
        self.wrap = wrap
        self.orientation = orientation
        # prop name -> distance from start along the belt, meters
        self.props: dict[str, float] = {}
        self.offsets: dict[str, Vec3] = {}
        self.stops: dict[str, float] = {}

    def position(self, at_m: float, offset: Vec3 = _NO_OFFSET) -> Vec3:
        f = at_m / self.length_m
        return tuple(  # type: ignore[return-value]
            s + (e - s) * f + o for s, e, o in zip(self.start, self.end, offset)
        )

    def add(
        self,
        name: str,
        at_m: float,
        offset: Vec3 = _NO_OFFSET,
        stop_m: float | None = None,
    ) -> Vec3:
        if not math.isfinite(at_m) or not 0.0 <= at_m <= self.length_m:
            raise ValueError(f"belt at_m must be within [0, {self.length_m:g}]")
        if len(offset) != 3 or not all(math.isfinite(v) for v in offset):
            raise ValueError("belt offset must be three finite numbers")
        if stop_m is not None and (
            not math.isfinite(stop_m) or not at_m <= stop_m <= self.length_m
        ):
            raise ValueError(
                f"belt stop_m must be within [at_m, {self.length_m:g}]"
            )
        self.props[name] = at_m
        self.offsets[name] = tuple(float(v) for v in offset)  # type: ignore[assignment]
        if stop_m is None:
            self.stops.pop(name, None)
        else:
            self.stops[name] = stop_m
        return self.position(at_m, self.offsets[name])

    def remove(self, name: str) -> None:
        self.props.pop(name, None)
        self.offsets.pop(name, None)
        self.stops.pop(name, None)

    def keep_load(self, old: Belt) -> None:
        """Carry old's props over, clamped to this belt's length."""
        for name, at_m in old.props.items():
            self.props[name] = min(at_m, self.length_m)
            self.offsets[name] = old.offsets.get(name, _NO_OFFSET)
            if name in old.stops:
                self.stops[name] = min(old.stops[name], self.length_m)

    def load(self, names: Sequence[str], spacing_m: float | None) -> dict[str, Vec3]:
        """Replace the load with names placed from start, spacing_m apart
        (default: spread evenly over the belt)."""
        if spacing_m is None:
            spacing_m = self.length_m / max(1, len(names))
        if not math.isfinite(spacing_m) or spacing_m < 0.0:
            raise ValueError("belt spacing_m must be finite and non-negative")
        self.props, self.offsets, self.stops = {}, {}, {}
        return {
            name: self.add(name, (i * spacing_m) % self.length_m)
            for i, name in enumerate(names)
        }

    def advance(self, dt: float) -> dict[str, Vec3]:
        """Move every prop dt seconds along; return each one's new position.
        A prop that reaches its stop leaves the belt there. One without a
        stop wraps to start at the end, or without wrap leaves at the end."""
        moved: dict[str, Vec3] = {}
        if dt <= 0.0 or self.speed_mps == 0.0:
            return moved
        for name, at_m in list(self.props.items()):
            offset = self.offsets.get(name, _NO_OFFSET)
            stop = self.stops.get(name)
            at_m += self.speed_mps * dt
            if stop is not None and at_m >= stop:
                moved[name] = self.position(stop, offset)
                self.remove(name)
                continue
            if at_m >= self.length_m:
                if self.wrap:
                    at_m %= self.length_m
                else:
                    moved[name] = self.position(self.length_m, offset)
                    self.remove(name)
                    continue
            self.props[name] = at_m
            moved[name] = self.position(at_m, offset)
        return moved
