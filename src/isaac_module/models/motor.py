"""viam-labs:isaac-sim:motor - a thruster that pushes a rigid body.

Attributes:
  world (string, required)   - name of the viam-labs:isaac-sim:world component
  prop (string)              - the world prop this pushes (a dynamic prop;
                               give it "mass" and, for a boat, "buoyancy")
  prim_path (string)         - or any rigid-body prim already in the stage
  position ([x,y,z] meters)  - where on the body it pushes, in the body frame,
                               default the origin
  direction ([x,y,z])        - the direction it pushes at full forward power,
                               in the body frame, default [1, 0, 0]
  max_thrust_n (float)       - thrust at full forward power, default 20
  reverse_thrust_n (float)   - thrust at full reverse power, default max_thrust_n
  max_rpm (float)            - propeller speed at full power, default 3000;
                               only used to integrate GetPosition and to
                               honour GoFor / SetRPM

Two thrusters either side of a body's centerline steer by differential
thrust; viam's rdk:builtin:wheeled base can drive them as left/right motors.

The thruster has no encoder: GetPosition reports revolutions integrated from
the commanded rpm in simulator time, which makes GoFor and GoTo behave like
the fake motor's. SetPower is the real control input.
"""

import asyncio
import time
from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.components.motor import Motor
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import struct_to_dict

from .. import FAMILY, NAMESPACE
from ..bodies import MotorHandle
from ..sim_manager import SimManager
from .utils import get_attrs


def validate_body_attached(config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
    """Shared by motor and movement_sensor: they name their world and the
    body (prop or prim) they attach to."""
    attrs = struct_to_dict(config.attributes)
    world = attrs.get("world")
    if not world or not isinstance(world, str):
        raise ValueError(
            f'{config.name}: set the "world" attribute to the name of your '
            "viam-labs:isaac-sim:world component"
        )
    prop = attrs.get("prop")
    prim_path = attrs.get("prim_path")
    if not (prop and isinstance(prop, str)) and not (prim_path and isinstance(prim_path, str)):
        raise ValueError(
            f'{config.name}: set "prop" (the name of a dynamic world prop) or '
            '"prim_path" (a rigid-body prim in the stage)'
        )
    return [world], []


GO_FOR_POLL_S = 0.02
GO_FOR_STALL_TIMEOUT_S = 30.0


class IsaacMotor(Motor, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "motor")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: Optional[MotorHandle] = None
        self._attrs: Dict[str, Any] = {}
        self._go_task: Optional[asyncio.Task] = None

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        motor = cls(config.name)
        motor.reconfigure(config, dependencies)
        return motor

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        return validate_body_attached(config)

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = get_attrs(config)
        self._attrs = attrs
        self._handle = SimManager.get().create_motor(self.name, attrs)

    def _h(self) -> MotorHandle:
        if self._handle is None:
            raise RuntimeError(f"motor {self.name} is not attached to the sim")
        return self._handle

    def _cancel_go(self) -> None:
        """A new command supersedes a timed GoFor: its sleep is cancelled so
        it does not stop the motor out from under the new command."""
        task = self._go_task
        if task is not None and not task.done():
            task.cancel()
        self._go_task = None

    async def set_power(self, power: float, **kwargs) -> None:
        self._cancel_go()
        self._h().set_power(power)

    async def set_rpm(self, rpm: float, **kwargs) -> None:
        self._cancel_go()
        self._h().set_rpm(rpm)

    async def go_for(self, rpm: float, revolutions: float, **kwargs) -> None:
        if rpm == 0:
            raise ValueError("rpm must be nonzero")
        handle = self._h()
        self._cancel_go()
        if revolutions == 0:
            handle.set_rpm(rpm)
            return
        direction = 1.0 if (rpm > 0) == (revolutions > 0) else -1.0
        speed = min(abs(rpm), handle.max_rpm)
        target = handle.revolutions() + direction * abs(revolutions)
        handle.set_rpm(direction * speed)
        # Revolutions are integrated in simulator time, which runs slower
        # than the wall clock when Isaac is rendering (and stops when it is
        # paused), so wait on the odometry rather than on a wall-clock sleep.
        waiter = asyncio.ensure_future(self._wait_for_revolutions(target, direction))
        self._go_task = waiter
        try:
            await waiter
        except asyncio.CancelledError:
            return  # superseded; the newer command owns the motor now
        finally:
            if self._go_task is waiter:
                self._go_task = None
        handle.stop()

    async def _wait_for_revolutions(self, target: float, direction: float) -> None:
        handle = self._h()
        sim = SimManager.get()
        started_sim_s = sim.sim_time_s()
        stalled_since = time.monotonic()
        while True:
            revolutions = handle.revolutions()
            if (revolutions - target) * direction >= 0.0:
                return
            await asyncio.sleep(GO_FOR_POLL_S)
            # a paused simulator never advances; give up after a while
            # instead of holding the RPC open forever
            if sim.sim_time_s() > started_sim_s:
                started_sim_s = sim.sim_time_s()
                stalled_since = time.monotonic()
            elif time.monotonic() - stalled_since > GO_FOR_STALL_TIMEOUT_S:
                handle.stop()
                raise TimeoutError(
                    f"motor {self.name}: simulator time stopped advancing during GoFor"
                )

    async def go_to(self, rpm: float, position_revolutions: float, **kwargs) -> None:
        handle = self._h()
        delta = position_revolutions - handle.revolutions()
        if abs(delta) < 1e-6:
            return
        await self.go_for(abs(rpm) if rpm != 0 else handle.max_rpm, delta)

    async def reset_zero_position(self, offset: float, **kwargs) -> None:
        self._h().reset_revolutions(offset)

    async def get_position(self, **kwargs) -> float:
        return self._h().revolutions()

    async def get_properties(self, **kwargs) -> Motor.Properties:
        return Motor.Properties(position_reporting=True)

    async def stop(self, **kwargs) -> None:
        self._cancel_go()
        self._h().stop()

    async def close(self) -> None:
        """viam-server removed or renamed this motor: take the thruster off
        the body (stop() has already zeroed it) and drop the cached handle."""
        self._cancel_go()
        handle = self._handle
        if handle is not None:
            handle.stop()
            handle.detach()
            self._handle = None
        SimManager.get().remove_handle(self.name)

    async def is_powered(self, **kwargs) -> Tuple[bool, float]:
        power = self._h().power()
        return power != 0.0, power

    async def is_moving(self) -> bool:
        return self._h().is_powered()

    async def get_geometries(self, **kwargs):
        # a thruster is small; reporting nothing keeps the frame system happy
        return []

    async def do_command(self, command: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        handle = self._h()
        cmd = str(command.get("command", ""))
        if cmd == "status":
            power = handle.power()
            return {
                "power": power,
                "thrust_n": power
                * (handle.max_thrust_n if power >= 0 else handle.reverse_thrust_n),
                "max_thrust_n": handle.max_thrust_n,
                "reverse_thrust_n": handle.reverse_thrust_n,
                "max_rpm": handle.max_rpm,
                "revolutions": handle.revolutions(),
                "body": handle.body.name,
            }
        raise ValueError(f"unknown command {cmd!r}; supported: status")
