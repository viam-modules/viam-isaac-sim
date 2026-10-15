"""viam-labs:isaac-sim:gripper - a simulated parallel gripper."""

from __future__ import annotations

import asyncio
import json
import math
import time
from typing import Any, ClassVar, Mapping, Optional, Sequence, Tuple

from grpclib import Status
from typing_extensions import Self
from viam.components.gripper import Gripper
from viam.errors import ViamGRPCError
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import (
    Geometry,
    KinematicsFileFormat,
    Pose,
    RectangularPrism,
    ResourceName,
    Vector3,
)
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes

from .. import FAMILY, NAMESPACE
from ..sim_manager import (
    KNOWN_ASSETS,
    LITE6_JAW_LAYOUTS,
    GripperHandle,
    SimManager,
    _prim_name,
)
from .utils import get_attrs

DEFAULT_ASSET = "robotiq_2f_85"
DEFAULT_TCP_OFFSET_M = 0.134
DEFAULT_TIMEOUT_S = 5.0
POLL_INTERVAL_S = 1.0 / 120.0
JAW_BOX_MM = (36.0, 146.0, 153.0)
JAW_BOX_CENTER_Z_MM = JAW_BOX_MM[2] / 2.0 - DEFAULT_TCP_OFFSET_M * 1000.0


def _default_parent_prim(arm_name: str) -> str:
    return f"/World/{_prim_name(arm_name)}/wrist_3_link"


def _kinematics(
    link_id: str,
    jaw_box_mm: tuple[float, float, float],
    jaw_box_center_z_mm: float,
) -> bytes:
    return json.dumps(
        {
            "name": link_id,
            "kinematic_param_type": "SVA",
            "links": [
                {
                    "id": link_id,
                    "parent": "world",
                    "translation": {"x": 0, "y": 0, "z": 0},
                    "orientation": {
                        "type": "ov_degrees",
                        "value": {"x": 0, "y": 0, "z": 1, "th": 0},
                    },
                    "geometry": {
                        "x": jaw_box_mm[0],
                        "y": jaw_box_mm[1],
                        "z": jaw_box_mm[2],
                        "translation": {"x": 0, "y": 0, "z": jaw_box_center_z_mm},
                    },
                }
            ],
            "joints": [],
        }
    ).encode()


class IsaacGripper(Gripper, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "gripper")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: Optional[GripperHandle] = None
        self._timeout = DEFAULT_TIMEOUT_S
        self._kinematics: Optional[Tuple[KinematicsFileFormat.ValueType, bytes]] = None
        self._jaw_box_mm = JAW_BOX_MM
        self._jaw_box_center_z_mm = JAW_BOX_CENTER_Z_MM
        self._asset = ""
        self._sensorless = False
        self._active_grip = False

    @classmethod
    def new(
        cls,
        config: ComponentConfig,
        dependencies: Mapping[ResourceName, ResourceBase],
    ) -> Self:
        gripper = cls(config.name)
        gripper.reconfigure(config, dependencies)
        return gripper

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = get_attrs(config)
        world = attrs.get("world")
        arm = attrs.get("arm")
        if not isinstance(world, str) or not world:
            raise ValueError(
                f'{config.name}: set "world" to the viam-labs:isaac-sim:world component name'
            )
        if not isinstance(arm, str) or not arm:
            raise ValueError(
                f'{config.name}: set "arm" to the viam-labs:isaac-sim:arm component name'
            )
        if config.HasField("frame") and config.frame.parent.split(":")[0] != arm:
            raise ValueError(
                f"{config.name}: frame.parent must be the attached arm {arm!r}"
            )
        asset = str(attrs.get("asset", DEFAULT_ASSET))
        meta = KNOWN_ASSETS.get(asset)
        if meta is None or meta.get("kind") != "gripper":
            supported = sorted(
                name
                for name, value in KNOWN_ASSETS.items()
                if value.get("kind") == "gripper"
            )
            raise ValueError(
                f"{config.name}: unsupported gripper asset {asset!r}; use one of {supported}"
            )
        jaw_layout = attrs.get("jaw_layout")
        if jaw_layout is not None:
            if asset != "lite6_gripper":
                raise ValueError(
                    f"{config.name}: jaw_layout is only supported by "
                    "asset 'lite6_gripper'"
                )
            if not isinstance(jaw_layout, str) or jaw_layout not in LITE6_JAW_LAYOUTS:
                raise ValueError(
                    f"{config.name}: unknown Lite6 jaw_layout {jaw_layout!r}; "
                    f"use one of {sorted(LITE6_JAW_LAYOUTS)}"
                )
        sensorless = attrs.get("sensorless", False)
        if not isinstance(sensorless, bool):
            raise ValueError(f"{config.name}: sensorless must be a boolean")
        timeout = float(attrs.get("grab_timeout_sec", DEFAULT_TIMEOUT_S))
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError(f"{config.name}: grab_timeout_sec must be positive")
        return [world, arm], []

    def reconfigure(
        self,
        config: ComponentConfig,
        dependencies: Mapping[ResourceName, ResourceBase],
    ) -> None:
        attrs = get_attrs(config)
        attrs.setdefault("asset", DEFAULT_ASSET)
        meta = KNOWN_ASSETS[str(attrs["asset"])]
        sensorless = attrs.get("sensorless", False)
        if not isinstance(sensorless, bool):
            raise ValueError(f"{config.name}: sensorless must be a boolean")
        if attrs["asset"] == "lite6_gripper":
            if not sensorless:
                self.logger.warning(
                    "%s: lite6_gripper is normally sensorless; set sensorless=true "
                    "to match production IsHolding semantics",
                    config.name,
                )
            attrs.setdefault("jaw_layout", "wide")
        if not meta.get("integrated_arm_asset"):
            attrs.setdefault("parent_prim", _default_parent_prim(str(attrs["arm"])))
        self._timeout = float(attrs.get("grab_timeout_sec", DEFAULT_TIMEOUT_S))
        self._sensorless = sensorless
        self._active_grip = False
        self._jaw_box_mm = tuple(meta.get("jaw_box_mm", JAW_BOX_MM))
        tcp_offset_m = float(meta.get("tcp_offset_m", DEFAULT_TCP_OFFSET_M))
        self._jaw_box_center_z_mm = self._jaw_box_mm[2] / 2.0 - tcp_offset_m * 1000.0
        self._kinematics = None
        self._handle = SimManager.get().create_gripper(self.name, attrs)
        self._asset = str(attrs["asset"])

    def policy_metadata(self) -> Mapping[str, Any]:
        """Return the simulator embodiment identity used by policy validation."""
        return {
            "simulation": True,
            "gripper_model": self._asset,
            "gripper_mode": "binary" if self._asset == "lite6_gripper" else "position",
        }

    def _h(self) -> GripperHandle:
        if self._handle is None:
            raise RuntimeError(f"gripper {self.name} is not attached to the sim")
        return self._handle

    async def _wait_stopped(
        self, *, timeout_is_error: bool = True, timeout_s: Optional[float] = None
    ) -> None:
        deadline = time.monotonic() + (
            self._timeout if timeout_s is None else timeout_s
        )
        while await asyncio.to_thread(self._h().is_moving):
            if time.monotonic() >= deadline:
                if timeout_is_error:
                    details = ""
                    contact_summary = getattr(self._h(), "contact_force_summary", None)
                    if contact_summary is not None:
                        summary = await asyncio.to_thread(contact_summary)
                        if summary is not None:
                            details = f"; contact_forces={json.dumps(summary, sort_keys=True)}"
                    raise TimeoutError(
                        f"gripper {self.name} did not settle within {self._timeout}s"
                        f"{details}"
                    )
                return

            await asyncio.sleep(POLL_INTERVAL_S)

    async def open(self, **kwargs) -> None:
        await asyncio.to_thread(self._h().open)
        self._active_grip = False
        if self._asset != "lite6_gripper":
            await self._wait_stopped(timeout_is_error=False, timeout_s=1.0)

    async def grab(self, **kwargs) -> bool:
        await asyncio.to_thread(self._h().close)
        self._active_grip = True
        if self._asset == "lite6_gripper":
            return True
        await self._wait_stopped()
        return await asyncio.to_thread(self._h().is_holding)

    async def is_holding_something(self, **kwargs) -> Gripper.HoldingStatus:
        handle = self._h()
        open_position, closed_position = await asyncio.to_thread(handle.jaw_limits)
        jaw_position = await asyncio.to_thread(handle.get_jaw)
        span = closed_position - open_position
        normalized = (jaw_position - open_position) / span if span else 0.0
        meta = {"input": min(max(normalized, 0.0), 1.0)}
        if await asyncio.to_thread(handle.jaw_unit) == "rad":
            meta.update(
                jaw_deg=math.degrees(jaw_position),
                open_deg=math.degrees(open_position),
                closed_deg=math.degrees(closed_position),
            )
        else:
            meta.update(
                jaw_m=jaw_position,
                open_m=open_position,
                closed_m=closed_position,
            )
        contact_summary = getattr(handle, "contact_force_summary", None)
        if contact_summary is not None:
            force_meta = await asyncio.to_thread(contact_summary)
            if force_meta is not None:
                meta.update(force_meta)
        firm_grasp = await asyncio.to_thread(handle.is_holding)
        meta["firm_grasp"] = firm_grasp
        is_holding = self._active_grip if self._sensorless else firm_grasp
        return Gripper.HoldingStatus(
            is_holding_something=is_holding,
            meta=meta,
        )

    async def stop(self, **kwargs) -> None:
        await asyncio.to_thread(self._h().stop)
        self._active_grip = False

    async def is_moving(self) -> bool:
        return await asyncio.to_thread(self._h().is_moving)

    async def get_kinematics(
        self, **kwargs
    ) -> Tuple[KinematicsFileFormat.ValueType, bytes]:
        if self._kinematics is None:
            self._kinematics = (
                KinematicsFileFormat.KINEMATICS_FILE_FORMAT_SVA,
                _kinematics(
                    self.name,
                    self._jaw_box_mm,
                    self._jaw_box_center_z_mm,
                ),
            )
        return self._kinematics

    async def get_current_inputs(self, **kwargs) -> list[float]:
        handle = self._h()
        if self._asset == "lite6_gripper":
            _, state = await asyncio.to_thread(handle.capture_state)
            return [state]
        open_position, closed_position = await asyncio.to_thread(handle.jaw_limits)
        jaw_position = await asyncio.to_thread(handle.get_jaw)
        span = closed_position - open_position
        normalized = (jaw_position - open_position) / span if span else 0.0
        return [min(max(normalized, 0.0), 1.0)]

    async def go_to_inputs(self, values: list[float], **kwargs) -> None:
        if len(values) != 1:
            raise ViamGRPCError(
                f"gripper {self.name}: expected one input, got {len(values)}",
                Status.INVALID_ARGUMENT,
            )
        value = float(values[0])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ViamGRPCError(
                f"gripper {self.name}: input must be in [0, 1], got {value}",
                Status.INVALID_ARGUMENT,
            )
        handle = self._h()
        if self._asset == "lite6_gripper":
            if value not in (0.0, 1.0):
                raise ViamGRPCError(
                    f"gripper {self.name}: Lite6 input must be binary, got {value}",
                    Status.INVALID_ARGUMENT,
                )
            await asyncio.to_thread(handle.open if value == 0.0 else handle.close)
            self._active_grip = value == 1.0
            return
        open_position, closed_position = await asyncio.to_thread(handle.jaw_limits)
        await asyncio.to_thread(
            handle.set_jaw, open_position + value * (closed_position - open_position)
        )
        self._active_grip = value == 1.0
        await self._wait_stopped()

    async def get_geometries(self, **kwargs) -> list[Geometry]:
        return [
            Geometry(
                center=Pose(
                    x=0,
                    y=0,
                    z=self._jaw_box_center_z_mm,
                    o_x=0,
                    o_y=0,
                    o_z=1,
                    theta=0,
                ),
                box=RectangularPrism(
                    dims_mm=Vector3(
                        x=self._jaw_box_mm[0],
                        y=self._jaw_box_mm[1],
                        z=self._jaw_box_mm[2],
                    )
                ),
                label=self.name,
            )
        ]

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        if command.get("command") == "policy_metadata":
            return self.policy_metadata()
        if command.get("command") == "set_input":
            raw_value = command.get("value")
            if not isinstance(raw_value, (int, float)):
                raise ValueError("set_input value must be numeric")
            value = float(raw_value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("set_input value must be finite and in [0, 1]")
            handle = self._h()
            if self._asset == "lite6_gripper":
                if value not in (0.0, 1.0):
                    raise ValueError("Lite6 set_input value must be binary")
                await asyncio.to_thread(handle.open if value == 0.0 else handle.close)
                self._active_grip = value == 1.0
                return {"accepted": True}
            open_position, closed_position = await asyncio.to_thread(handle.jaw_limits)
            await asyncio.to_thread(
                handle.set_jaw,
                open_position + value * (closed_position - open_position),
            )
            self._active_grip = value == 1.0
            return {"accepted": True}
        raise ValueError(f"unknown command: {command}")
