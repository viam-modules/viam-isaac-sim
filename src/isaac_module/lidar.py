"""RTX lidar presets and handles. Isaac imports stay inside IsaacLidarHandle,
which only runs on the sim thread in Isaac mode."""

from dataclasses import dataclass, replace
from typing import Any, Optional

import numpy as np
from viam.logging import getLogger

LOGGER = getLogger(__name__)

Scan = tuple[float, np.ndarray]  # (sim time s, (N, 3) float32 meters, lidar frame)


@dataclass(frozen=True)
class LidarPreset:
    configs: tuple[str, ...]  # Isaac RTX config names, tried in order (5.0 names first)
    native_hz: float
    planar: bool
    stand_in_for: Optional[str] = None


_OUSTER_OS1_32 = LidarPreset(
    configs=("OS1_REV6_32ch10hz1024res", "OS1_32ch10hz1024res"),
    native_hz=10.0,
    planar=False,
)

LIDAR_PRESETS: dict[str, LidarPreset] = {
    "rplidar_s2e": LidarPreset(configs=("RPLIDAR_S2E",), native_hz=10.0, planar=True),
    "ouster_os1_32": _OUSTER_OS1_32,
    # Dexmate Vega's RoboSense: exact model unconfirmed, OS1-32 stands in
    "robosense_vega": replace(_OUSTER_OS1_32, stand_in_for="ouster_os1_32"),
}


def effective_rate(attrs: dict[str, Any]) -> float:
    preset = LIDAR_PRESETS[attrs["sensor"]]
    rate = float(attrs.get("rate_hz", preset.native_hz))
    if rate > preset.native_hz:
        raise ValueError(
            f"rate_hz {rate:g} exceeds {attrs['sensor']} native rate "
            f"{preset.native_hz:g} Hz"
        )
    return rate


def _due(scan: Optional[Scan], now: float, rate_hz: float) -> bool:
    # now < stamp: sim time went backwards (world reset), don't serve stale
    return scan is None or now < scan[0] or now - scan[0] >= 1.0 / rate_hz - 1e-9


class LidarHandle:
    def latest_scan(self) -> Optional[Scan]:
        raise NotImplementedError

    def reconfigure(self, attrs: dict[str, Any]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


def _mock_points(planar: bool) -> np.ndarray:
    az = np.radians(np.arange(360, dtype=np.float32))
    r = np.where(np.arange(360) % 2 == 0, 1.0, 8.0).astype(np.float32)
    elevations = [0.0] if planar else np.radians(np.linspace(-15.0, 15.0, 16))
    rings = [
        np.stack(
            [
                r * np.cos(e) * np.cos(az),
                r * np.cos(e) * np.sin(az),
                r * np.sin(e) * np.ones_like(az),
            ],
            axis=1,
        )
        for e in elevations
    ]
    return np.concatenate(rings).astype(np.float32)


class MockLidarHandle(LidarHandle):
    """Alternating 1 m / 8 m returns. Decimation is applied lazily on read
    because the mock sim clock only advances in the run loop."""

    def __init__(self, sim: Any, attrs: dict[str, Any]) -> None:
        self._sim = sim
        self._preset = LIDAR_PRESETS[attrs["sensor"]]
        self._rate = effective_rate(attrs)
        self._points = _mock_points(self._preset.planar)
        self._scan: Optional[Scan] = None

    def reconfigure(self, attrs: dict[str, Any]) -> None:
        self._rate = effective_rate(attrs)

    def latest_scan(self) -> Optional[Scan]:
        now = self._sim.sim_time_s()
        if now >= 1.0 / self._preset.native_hz - 1e-9 and _due(self._scan, now, self._rate):
            self._scan = (now, self._points)
        return self._scan


# Scan-buffer annotator was renamed in Isaac 5.0; try newest first.
_SCAN_ANNOTATORS = (
    "IsaacCreateRTXLidarScanBuffer",
    "RtxSensorCpuIsaacCreateRTXLidarScanBuffer",
)


def _read_lidar_scan(annotator: Any) -> Optional[np.ndarray]:
    """The only Isaac-version shim: latest full scan as (N, 3) float32 meters
    in the lidar frame (empty if every beam missed), or None if the annotator
    has nothing yet."""
    data = annotator.get_data()
    points = data.get("data") if isinstance(data, dict) else data
    if points is None:
        return None
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    points = points[np.any(points != 0, axis=1)]  # RTX reports no-return beams as (0,0,0)
    return points


class IsaacLidarHandle(LidarHandle):
    def __init__(self, sim: Any, name: str, prim_path: str, attrs: dict[str, Any]) -> None:
        # runs on the sim thread
        import omni.kit.commands
        import omni.replicator.core as rep

        sensor_name = attrs["sensor"]
        preset = LIDAR_PRESETS[sensor_name]
        if preset.stand_in_for:
            LOGGER.warning(
                "%s: %s is simulated with the %s preset as a stand-in",
                name, sensor_name, preset.stand_in_for,
            )
        sensor = None
        for config in preset.configs:
            try:
                ok, prim = omni.kit.commands.execute(
                    "IsaacSensorCreateRtxLidar", path=prim_path, parent=None, config=config
                )
            except Exception:
                LOGGER.debug("RTX lidar config %s unavailable", config, exc_info=True)
                continue
            if ok and prim is not None and prim.IsValid():
                sensor = prim
                break
        if sensor is None:
            raise ValueError(
                f"{name}: none of the Isaac RTX lidar configs {list(preset.configs)} "
                f"for {sensor_name!r} exist in this Isaac Sim version"
            )
        self._render_product = rep.create.render_product(
            str(sensor.GetPath()), [1, 1], name=f"{name}_lidar"
        )
        self._annotator = None
        for annotator_name in _SCAN_ANNOTATORS:
            try:
                annotator = rep.AnnotatorRegistry.get_annotator(annotator_name)
            except Exception:
                LOGGER.debug("annotator %s unavailable", annotator_name, exc_info=True)
                continue
            annotator.attach([self._render_product.path])
            self._annotator = annotator
            break
        if self._annotator is None:
            raise RuntimeError(f"{name}: no RTX lidar scan-buffer annotator in this Isaac Sim")
        self._prim_path = str(sensor.GetPath())
        self._xform = sim._isaac.SingleXFormPrim(self._prim_path)
        self._rate = effective_rate(attrs)
        self._native_hz = preset.native_hz
        self._scan: Optional[Scan] = None
        self._apply_pose(attrs)

    def _apply_pose(self, attrs: dict[str, Any]) -> None:
        # lidar frames are Isaac-native (X forward, Z up): identity default,
        # and a parent_prim pose is always local to that prim
        if "local_position" in attrs or "local_orientation_wxyz" in attrs or attrs.get("parent_prim"):
            self._xform.set_local_pose(
                translation=list(attrs.get("local_position", (0.0, 0.0, 0.0))),
                orientation=list(attrs.get("local_orientation_wxyz", (1.0, 0.0, 0.0, 0.0))),
            )
        else:
            self._xform.set_world_pose(
                position=list(attrs.get("position", (0.0, 0.0, 0.0))),
                orientation=list(attrs.get("orientation_wxyz", (1.0, 0.0, 0.0, 0.0))),
            )

    def reconfigure(self, attrs: dict[str, Any]) -> None:
        self._rate = effective_rate(attrs)
        self._apply_pose(attrs)

    def poll(self, now: float) -> None:
        """Called after each rendered frame on the sim thread."""
        if self._scan is not None and now < self._scan[0]:
            self._scan = None  # world reset: drop the pre-reset scan
        # the buffer holds the previous full rotation until one completes
        # after start/reset, so don't stamp it as new before then
        if now < 1.0 / self._native_hz - 1e-9:
            return
        if not _due(self._scan, now, self._rate):
            return
        points = _read_lidar_scan(self._annotator)
        if points is not None:
            self._scan = (now, points.copy())

    def latest_scan(self) -> Optional[Scan]:
        return self._scan  # tuple swap is atomic; no sim-thread hop

    def close(self) -> None:
        """Stop rendering this lidar and remove its prim (sim thread)."""
        import omni.usd

        self._annotator.detach()
        self._render_product.destroy()
        omni.usd.get_context().get_stage().RemovePrim(self._prim_path)
