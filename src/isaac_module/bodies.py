"""External forces Isaac Sim does not provide (PhysX has no water and no
thrusters), applied to rigid bodies. Shared by the world (floating props),
the motor (thruster) model and the movement_sensor model.

A prop with a "buoyancy" block floats on the world's water. Every physics
step the module sums one net force and torque on that body from:

* buoyancy - the body is sampled at a few body-frame points (by default the
  four corners of its footprint); each point is a vertical column of water
  that pushes up in proportion to how deep that part of the body sits.
  Tilting submerges one side more than the other, so this one rule also
  gives heave, roll and pitch restoring moments with no separate stiffness.
* hydrodynamic damping - linear plus quadratic drag per body axis (surge,
  sway, heave) and per rotation axis (roll, pitch, yaw).
* thrusters - viam-labs:isaac-sim:motor components attached to the body.
  Each pushes along its configured direction at its configured offset, so
  two thrusters either side of the centerline steer by differential thrust.

Everything here is plain python so the mock backend and the tests exercise
the same force model that drives the PhysX rigid body in Isaac.

Conventions: body frame +X forward, +Y left, +Z up, origin at the prop's
origin (the center of a cube prop). Quaternions are (w, x, y, z). The world
is Isaac's: +Z up, meters; the movement sensor maps world +X to east and +Y
to north.
"""

import math
import random
import threading
from dataclasses import dataclass
from typing import Any, Sequence

from .spatial import Quat, Vec3, quat_conj, quat_rotate, quat_to_ov

GRAVITY_MPS2 = 9.81
EARTH_RADIUS_M = 6_371_000.0
# a step bigger than this is split so the mock integrator stays stable when
# the wall clock stalls (e.g. the first step after boot)
MOCK_MAX_SUBSTEP_S = 0.02


def _vec3(value: Any, label: str, default: Vec3 | None = None) -> Vec3:
    if value is None:
        if default is None:
            raise ValueError(f"{label} is required")
        return default
    try:
        x, y, z = value
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain exactly three numbers") from exc
    out = (float(x), float(y), float(z))
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{label} must be finite")
    return out


def _positive(value: Any, label: str, default: float | None = None) -> float:
    if value is None:
        if default is None:
            raise ValueError(f"{label} is required")
        value = default
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return out


def _non_negative_vec3(value: Any, label: str, default: Vec3) -> Vec3:
    out = _vec3(value, label, default)
    if any(v < 0.0 for v in out):
        raise ValueError(f"{label} must be non-negative")
    return out


@dataclass(frozen=True)
class BuoyancySpec:
    """A prop's floating behaviour, parsed from its "buoyancy" block."""

    mass_kg: float
    size_m: Vec3  # footprint length, width and the height that can submerge
    water_level_m: float
    water_density: float
    waterplane_area_m2: float
    bottom_z_m: float  # body-frame z of the underside
    points: tuple[tuple[float, float], ...]
    linear_damping: Vec3
    quadratic_damping: Vec3
    angular_damping: Vec3
    angular_quadratic_damping: Vec3

    @staticmethod
    def from_prop(prop: dict[str, Any], water_level_m: float) -> "BuoyancySpec | None":
        """Parse a prop's buoyancy block; None when the prop does not float.
        A cube prop's dimensions come from size * scale; a usd prop has to
        say its size_m in the block."""
        raw = prop.get("buoyancy")
        if raw is None or raw is False:
            return None
        if raw is True:
            block: dict[str, Any] = {}
        elif isinstance(raw, dict):
            block = dict(raw)
        else:
            raise ValueError("buoyancy must be true or an object")
        name = prop.get("name", "?")
        if prop.get("fixed"):
            raise ValueError(f"prop {name}: a fixed prop cannot float")
        mass = _positive(prop.get("mass"), f"prop {name}: mass (kg) is required to float")

        if block.get("size_m") is not None:
            size = _vec3(block["size_m"], "buoyancy.size_m")
        elif str(prop.get("type", "cube")) == "cube":
            edge = float(prop.get("size", 0.05))
            scale = _vec3(prop.get("scale"), "scale", (1.0, 1.0, 1.0))
            size = (edge * scale[0], edge * scale[1], edge * scale[2])
        else:
            raise ValueError(f"prop {name}: buoyancy.size_m is required for a usd prop")
        if any(v <= 0.0 for v in size):
            raise ValueError(f"prop {name}: buoyancy size must be positive")
        length, width, height = size

        points: Any = block.get("points")
        if points is None:
            parsed = tuple(
                (x, y)
                for x in (-length / 4.0, length / 4.0)
                for y in (-width / 4.0, width / 4.0)
            )
        else:
            parsed_list: list[tuple[float, float]] = []
            for point in points:
                try:
                    x, y = point
                except (TypeError, ValueError) as exc:
                    raise ValueError("buoyancy.points entries are [x, y] meters") from exc
                parsed_list.append((float(x), float(y)))
            if not parsed_list:
                raise ValueError("buoyancy.points must not be empty")
            parsed = tuple(parsed_list)
        waterplane = _positive(
            block.get("waterplane_area_m2"), "buoyancy.waterplane_area_m2", length * width
        )
        density = _positive(block.get("water_density"), "buoyancy.water_density", 1000.0)
        level = float(block.get("water_level_m", water_level_m))
        if not math.isfinite(level):
            raise ValueError("buoyancy.water_level_m must be finite")
        bottom = float(block.get("bottom_z_m", -height / 2.0))
        if mass >= density * waterplane * height:
            raise ValueError(
                f"prop {name}: mass {mass:g} kg exceeds its displacement "
                f"({density * waterplane * height:g} kg); it would sink"
            )
        return BuoyancySpec(
            mass_kg=mass,
            size_m=size,
            water_level_m=level,
            water_density=density,
            waterplane_area_m2=waterplane,
            bottom_z_m=bottom,
            points=parsed,
            linear_damping=_non_negative_vec3(
                block.get("linear_damping"), "buoyancy.linear_damping", (2.0, 20.0, 150.0)
            ),
            quadratic_damping=_non_negative_vec3(
                block.get("quadratic_damping"),
                "buoyancy.quadratic_damping",
                (4.5, 60.0, 80.0),
            ),
            angular_damping=_non_negative_vec3(
                block.get("angular_damping"), "buoyancy.angular_damping", (5.0, 5.0, 3.0)
            ),
            angular_quadratic_damping=_non_negative_vec3(
                block.get("angular_quadratic_damping"),
                "buoyancy.angular_quadratic_damping",
                (2.0, 2.0, 8.0),
            ),
        )

    @property
    def draft_m(self) -> float:
        """Depth the body sits at rest: weight = displaced water."""
        return self.mass_kg / (self.water_density * self.waterplane_area_m2)

    @property
    def rest_z_m(self) -> float:
        """Body-origin height at which the body floats level."""
        return self.water_level_m - self.bottom_z_m - self.draft_m

    @property
    def yaw_inertia(self) -> float:
        length, width, _ = self.size_m
        return self.mass_kg * (length * length + width * width) / 12.0


@dataclass
class ThrusterState:
    """A thruster's command, kept on the body it pushes."""

    name: str
    position_body: Vec3
    direction_body: Vec3
    force_n: float = 0.0
    rpm: float = 0.0
    revolutions: float = 0.0

    def tick(self, dt: float) -> None:
        self.revolutions += self.rpm * dt / 60.0


@dataclass(frozen=True)
class BodyState:
    position: Vec3
    orientation_wxyz: Quat
    linear_velocity: Vec3  # world, m/s
    angular_velocity: Vec3  # world, rad/s
    linear_acceleration: Vec3  # world, m/s^2, without gravity
    sim_time_s: float


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _normalize(a: Vec3) -> Vec3:
    n = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    if n == 0.0:
        raise ValueError("direction must not be zero")
    return (a[0] / n, a[1] / n, a[2] / n)


def quat_from_yaw(yaw_rad: float) -> Quat:
    return (math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0))


def quat_to_rpy(q: Quat) -> tuple[float, float, float]:
    """(w,x,y,z) quaternion to roll, pitch, yaw radians (Z-Y-X intrinsic)."""
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def net_wrench(
    spec: BuoyancySpec | None,
    position: Vec3,
    orientation: Quat,
    linear_velocity: Vec3,
    angular_velocity: Vec3,
    thrusters: Sequence[ThrusterState],
) -> tuple[Vec3, Vec3]:
    """Net force and torque (world frame, torque about the body origin) from
    buoyancy and hydrodynamic damping (when the body floats) and the
    thrusters. Gravity is left to the physics engine."""
    q = orientation
    q_inv = quat_conj(q)
    force: Vec3 = (0.0, 0.0, 0.0)
    torque: Vec3 = (0.0, 0.0, 0.0)

    if spec is not None:
        # buoyancy: one water column per sample point
        column_area = spec.waterplane_area_m2 / len(spec.points)
        column_gain = spec.water_density * GRAVITY_MPS2 * column_area
        wet_height = spec.size_m[2]
        for bx, by in spec.points:
            bottom_world = _add(position, quat_rotate(q, (bx, by, spec.bottom_z_m)))
            depth = spec.water_level_m - bottom_world[2]
            depth = max(0.0, min(wet_height, depth))
            if depth <= 0.0:
                continue
            lift = (0.0, 0.0, column_gain * depth)
            # the column's center of buoyancy is halfway up the submerged part
            lever = _sub(_add(bottom_world, (0.0, 0.0, depth / 2.0)), position)
            force = _add(force, lift)
            torque = _add(torque, _cross(lever, lift))

        # damping in the body frame
        v_body = quat_rotate(q_inv, linear_velocity)
        w_body = quat_rotate(q_inv, angular_velocity)
        drag_body = tuple(
            -(
                spec.linear_damping[i] * v_body[i]
                + spec.quadratic_damping[i] * v_body[i] * abs(v_body[i])
            )
            for i in range(3)
        )
        spin_body = tuple(
            -(
                spec.angular_damping[i] * w_body[i]
                + spec.angular_quadratic_damping[i] * w_body[i] * abs(w_body[i])
            )
            for i in range(3)
        )
        force = _add(force, quat_rotate(q, drag_body))  # type: ignore[arg-type]
        torque = _add(torque, quat_rotate(q, spin_body))  # type: ignore[arg-type]

    # thrusters: push along their own axis, at their offset from the origin
    for thruster in thrusters:
        if thruster.force_n == 0.0:
            continue
        push = _scale(quat_rotate(q, thruster.direction_body), thruster.force_n)
        lever = quat_rotate(q, thruster.position_body)
        force = _add(force, push)
        torque = _add(torque, _cross(lever, push))

    return force, torque


class BodyHandle:
    """A rigid body the motor and movement_sensor models talk to. Thread
    safe: the physics step runs on the sim thread, everything else on the
    module's gRPC threads."""

    def __init__(
        self,
        name: str,
        spec: BuoyancySpec | None,
        spawn_position: Vec3,
        spawn_orientation: Quat,
    ) -> None:
        self.name = name
        self.spec = spec
        self.spawn_position = spawn_position
        self.spawn_orientation = spawn_orientation
        self._lock = threading.Lock()
        self._thrusters: dict[str, ThrusterState] = {}
        self._state = BodyState(
            spawn_position, spawn_orientation, (0.0,) * 3, (0.0,) * 3, (0.0,) * 3, 0.0
        )
        self._prev_velocity: Vec3 | None = None

    # --- thrusters ---------------------------------------------------------

    def register_thruster(self, name: str, position_body: Vec3, direction_body: Vec3) -> ThrusterState:
        state = ThrusterState(name, position_body, _normalize(direction_body))
        with self._lock:
            previous = self._thrusters.get(name)
            if previous is not None:
                # a reconfigured motor keeps its odometry
                state.revolutions = previous.revolutions
            self._thrusters[name] = state
        return state

    def unregister_thruster(self, name: str) -> None:
        with self._lock:
            self._thrusters.pop(name, None)

    def thrusters(self) -> list[ThrusterState]:
        with self._lock:
            return list(self._thrusters.values())

    # --- state -------------------------------------------------------------

    def get_state(self) -> BodyState:
        with self._lock:
            return self._state

    def _wrench(self, position: Vec3, orientation: Quat, linear: Vec3, angular: Vec3) -> tuple[Vec3, Vec3]:
        with self._lock:
            thrusters = list(self._thrusters.values())
        return net_wrench(self.spec, position, orientation, linear, angular, thrusters)

    def _publish(
        self,
        position: Vec3,
        orientation: Quat,
        linear: Vec3,
        angular: Vec3,
        dt: float,
        sim_time_s: float,
    ) -> None:
        with self._lock:
            if self._prev_velocity is None or dt <= 0.0:
                # first sample, or the one after a teleport: no history to
                # differentiate against
                acceleration: Vec3 = (0.0, 0.0, 0.0)
            else:
                acceleration = _scale(_sub(linear, self._prev_velocity), 1.0 / dt)
            self._prev_velocity = linear
            self._state = BodyState(position, orientation, linear, angular, acceleration, sim_time_s)
            for thruster in self._thrusters.values():
                thruster.tick(dt)

    def reset_motion(self, position: Vec3, orientation: Quat) -> None:
        """The body was teleported and its velocities zeroed: publish the new
        pose at rest and forget the velocity history, so the IMU does not
        report the jump as one huge acceleration sample."""
        with self._lock:
            self._prev_velocity = None
            self._state = BodyState(
                position,
                orientation,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                self._state.sim_time_s,
            )


class MockBodyHandle(BodyHandle):
    """Planar (surge, sway, yaw) integration of the same force model. A
    floating body sits level at its rest draft; a body without buoyancy
    slides without friction."""

    def __init__(
        self,
        sim: Any,
        name: str,
        spec: BuoyancySpec | None,
        mass_kg: float,
        spawn_position: Vec3,
        spawn_orientation: Quat,
    ) -> None:
        _roll, _pitch, yaw = quat_to_rpy(spawn_orientation)
        z = spec.rest_z_m if spec is not None else spawn_position[2]
        level_spawn = (spawn_position[0], spawn_position[1], z)
        super().__init__(name, spec, level_spawn, quat_from_yaw(yaw))
        self._sim = sim
        self._mass = spec.mass_kg if spec is not None else mass_kg
        self._yaw_inertia = spec.yaw_inertia if spec is not None else mass_kg * 0.1
        self._position = level_spawn
        self._yaw = yaw
        self._velocity = (0.0, 0.0, 0.0)
        self._yaw_rate = 0.0
        self._time_s = 0.0
        sim._step_callbacks[f"body_{name}"] = self._on_step

    def _on_step(self, dt: float) -> None:
        if dt <= 0.0:
            return
        substeps = max(1, math.ceil(dt / MOCK_MAX_SUBSTEP_S))
        h = dt / substeps
        for _ in range(substeps):
            with self._lock:
                position, yaw, velocity, yaw_rate = (
                    self._position,
                    self._yaw,
                    self._velocity,
                    self._yaw_rate,
                )
            force, torque = self._wrench(position, quat_from_yaw(yaw), velocity, (0.0, 0.0, yaw_rate))
            vx = velocity[0] + force[0] / self._mass * h
            vy = velocity[1] + force[1] / self._mass * h
            yaw_rate += torque[2] / self._yaw_inertia * h
            yaw += yaw_rate * h
            with self._lock:
                self._velocity = (vx, vy, 0.0)
                self._yaw_rate = yaw_rate
                self._yaw = math.atan2(math.sin(yaw), math.cos(yaw))
                self._position = (position[0] + vx * h, position[1] + vy * h, position[2])
            self._time_s += h
        self._publish(
            self._position,
            quat_from_yaw(self._yaw),
            self._velocity,
            (0.0, 0.0, self._yaw_rate),
            dt,
            self._time_s,
        )

    def set_pose(self, position: Vec3, orientation: Quat) -> None:
        _roll, _pitch, yaw = quat_to_rpy(orientation)
        z = self.spec.rest_z_m if self.spec is not None else position[2]
        with self._lock:
            self._position = (position[0], position[1], z)
            self._yaw = yaw
            self._velocity = (0.0, 0.0, 0.0)
            self._yaw_rate = 0.0
        self.reset_motion(self._position, quat_from_yaw(yaw))


class MotorHandle:
    """A thruster. All the physics lives on the body; this just sets the
    command and reads back the integrated odometry."""

    def __init__(
        self,
        body: BodyHandle,
        state: ThrusterState,
        max_thrust_n: float,
        reverse_thrust_n: float,
        max_rpm: float,
    ) -> None:
        self.body = body
        self._state = state
        self.max_thrust_n = max_thrust_n
        self.reverse_thrust_n = reverse_thrust_n
        self.max_rpm = max_rpm
        self._power = 0.0

    def update_limits(self, max_thrust_n: float, reverse_thrust_n: float, max_rpm: float) -> None:
        self.max_thrust_n = max_thrust_n
        self.reverse_thrust_n = reverse_thrust_n
        self.max_rpm = max_rpm
        self.set_power(self._power)

    def set_power(self, power: float) -> None:
        power = float(power)
        if not math.isfinite(power):
            raise ValueError("power must be finite")
        power = max(-1.0, min(1.0, power))
        thrust = self.max_thrust_n if power >= 0.0 else self.reverse_thrust_n
        with self.body._lock:
            self._power = power
            self._state.force_n = power * thrust
            self._state.rpm = power * self.max_rpm

    def set_rpm(self, rpm: float) -> None:
        self.set_power(rpm / self.max_rpm)

    def stop(self) -> None:
        self.set_power(0.0)

    def power(self) -> float:
        with self.body._lock:
            return self._power

    def is_powered(self) -> bool:
        return self.power() != 0.0

    def revolutions(self) -> float:
        with self.body._lock:
            return self._state.revolutions

    def reset_revolutions(self, offset: float) -> None:
        with self.body._lock:
            self._state.revolutions = -float(offset)

    def detach(self) -> None:
        self.body.unregister_thruster(self._state.name)


@dataclass(frozen=True)
class GeoOrigin:
    latitude: float
    longitude: float
    altitude_m: float = 0.0

    @staticmethod
    def from_attrs(value: Any) -> "GeoOrigin":
        if value is None:
            return GeoOrigin(0.0, 0.0, 0.0)
        if not isinstance(value, dict):
            raise ValueError("origin must be an object with latitude, longitude, altitude_m")
        lat = float(value.get("latitude", 0.0))
        lon = float(value.get("longitude", 0.0))
        alt = float(value.get("altitude_m", 0.0))
        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            raise ValueError("origin latitude/longitude out of range")
        return GeoOrigin(lat, lon, alt)

    def to_geo(self, east_m: float, north_m: float, up_m: float) -> tuple[float, float, float]:
        """Local east/north/up offset from the origin to lat, lon, altitude
        (equirectangular; accurate to well under a meter at sim scale)."""
        lat = self.latitude + math.degrees(north_m / EARTH_RADIUS_M)
        cos_lat = math.cos(math.radians(self.latitude))
        lon = self.longitude + (
            math.degrees(east_m / (EARTH_RADIUS_M * cos_lat)) if cos_lat > 1e-9 else 0.0
        )
        return lat, lon, self.altitude_m + up_m


@dataclass(frozen=True)
class MovementSensorSpec:
    origin: GeoOrigin
    position_noise_m: float = 0.0
    heading_noise_deg: float = 0.0
    velocity_noise_mps: float = 0.0
    include_gravity: bool = True
    seed: int | None = None

    @staticmethod
    def from_attrs(attrs: dict[str, Any]) -> "MovementSensorSpec":
        def noise(key: str) -> float:
            value = float(attrs.get(key, 0.0))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{key} must be finite and non-negative")
            return value

        seed = attrs.get("seed")
        return MovementSensorSpec(
            origin=GeoOrigin.from_attrs(attrs.get("origin")),
            position_noise_m=noise("position_noise_m"),
            heading_noise_deg=noise("heading_noise_deg"),
            velocity_noise_mps=noise("velocity_noise_mps"),
            include_gravity=bool(attrs.get("include_gravity", True)),
            seed=int(seed) if seed is not None else None,
        )


def compass_heading_deg(orientation: Quat) -> float:
    """Bearing of the body's +X axis, degrees clockwise from north (world +Y)."""
    forward = quat_rotate(orientation, (1.0, 0.0, 0.0))
    heading = math.degrees(math.atan2(forward[0], forward[1]))
    return heading % 360.0


class MovementSensorHandle:
    """GPS + IMU + compass readings derived from a body's rigid-body state."""

    def __init__(self, body: BodyHandle, spec: MovementSensorSpec) -> None:
        self.body = body
        self.spec = spec
        self._rng = random.Random(spec.seed)

    def update_spec(self, spec: MovementSensorSpec) -> None:
        self.spec = spec

    def _noise(self, sigma: float) -> float:
        return self._rng.gauss(0.0, sigma) if sigma > 0.0 else 0.0

    def readings(self) -> dict[str, Any]:
        state = self.body.get_state()
        spec = self.spec
        x, y, z = state.position
        lat, lon, alt = spec.origin.to_geo(
            x + self._noise(spec.position_noise_m),
            y + self._noise(spec.position_noise_m),
            z,
        )
        q = state.orientation_wxyz
        q_inv = quat_conj(q)
        w_body = quat_rotate(q_inv, state.angular_velocity)
        accel_world = state.linear_acceleration
        if spec.include_gravity:
            # an IMU measures specific force: at rest it reads +g upward
            accel_world = _add(accel_world, (0.0, 0.0, GRAVITY_MPS2))
        a_body = quat_rotate(q_inv, accel_world)
        heading = (compass_heading_deg(q) + self._noise(spec.heading_noise_deg)) % 360.0
        ox, oy, oz, theta = quat_to_ov(q)
        velocity = tuple(v + self._noise(spec.velocity_noise_mps) for v in state.linear_velocity)
        roll, pitch, yaw = quat_to_rpy(q)
        return {
            "position": {"latitude": lat, "longitude": lon, "altitude_m": alt},
            "linear_velocity": {"x": velocity[0], "y": velocity[1], "z": velocity[2]},
            "angular_velocity": {
                "x": math.degrees(w_body[0]),
                "y": math.degrees(w_body[1]),
                "z": math.degrees(w_body[2]),
            },
            "linear_acceleration": {"x": a_body[0], "y": a_body[1], "z": a_body[2]},
            "compass_heading": heading,
            "orientation": {"o_x": ox, "o_y": oy, "o_z": oz, "theta": math.degrees(theta)},
            "sim_position_m": {"x": x, "y": y, "z": z},
            "roll_deg": math.degrees(roll),
            "pitch_deg": math.degrees(pitch),
            "yaw_deg": math.degrees(yaw),
            "speed_mps": math.hypot(velocity[0], velocity[1]),
            "sim_time_s": state.sim_time_s,
        }
