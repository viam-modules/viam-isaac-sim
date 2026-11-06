"""viam-labs:isaac-sim:ufactory-xarm6 - a simulated UFactory xArm 6.

Takes the same attributes as viam:ufactory:xArm6, so a machine's real arm
config runs unchanged in sim; the app swaps the model and adds `world` when it
simulates a machine. Connection settings (host, port, the UFactory Studio
proxy) and hardware-only settings (collision_sensitivity, bad-joints) are
accepted and ignored.

Attributes:
  world (string, required)                     - name of the viam-labs:isaac-sim:world component
  speed_degs_per_sec (float)                   - max joint speed (default 60, range [0.25, 180])
  acceleration_degs_per_sec_per_sec (float)    - max joint acceleration (default 381.67, max 1145)
  move_hz (float)                              - rate trajectories are streamed at (default 100,
                                                 range [20, 1000])
  motion (string)                              - motion service used by MoveToPosition
                                                 (default builtin)
"""

from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence, Tuple

from viam.components.arm import Arm, Pose
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PoseInFrame, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily
from viam.services.motion import Motion
from viam.utils import dict_to_struct

from .. import FAMILY, NAMESPACE
from .arm import IsaacArm
from .utils import get_attrs, validate_sim_component

# the viam:ufactory:xArm6 driver's limits and defaults
MIN_SPEED_DEG_S = 0.25
MAX_SPEED_DEG_S = 180.0
MAX_ACCEL_DEG_S2 = 1145.0
DEFAULT_SPEED_DEG_S = MAX_SPEED_DEG_S / 3
DEFAULT_ACCEL_DEG_S2 = MAX_ACCEL_DEG_S2 / 3
DEFAULT_MOVE_HZ = 100.0
MIN_MOVE_HZ = 20.0
MAX_MOVE_HZ = 1000.0

ASSET = "xarm6"
DEFAULT_MOTION = "builtin"


def _number(attrs: Mapping[str, Any], key: str, name: str) -> float:
    raw = attrs.get(key, 0)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{name}: {key} must be numeric")
    return float(raw)


def _limits(attrs: Mapping[str, Any], name: str) -> Tuple[float, float, float]:
    """(speed deg/s, acceleration deg/s^2, move hz), validated and defaulted
    the way the real driver does it."""
    speed = _number(attrs, "speed_degs_per_sec", name)
    accel = _number(attrs, "acceleration_degs_per_sec_per_sec", name)
    move_hz = _number(attrs, "move_hz", name)
    if accel < 0:
        raise ValueError(f"{name}: given acceleration {accel} cannot be negative")
    if accel > MAX_ACCEL_DEG_S2:
        raise ValueError(
            f"{name}: given acceleration {accel} cannot be more than {MAX_ACCEL_DEG_S2}"
        )
    if speed != 0 and not MIN_SPEED_DEG_S <= speed <= MAX_SPEED_DEG_S:
        raise ValueError(
            f"{name}: given speed {speed} must be between "
            f"{MIN_SPEED_DEG_S} and {MAX_SPEED_DEG_S}"
        )
    if move_hz > 0 and not MIN_MOVE_HZ <= move_hz <= MAX_MOVE_HZ:
        raise ValueError(
            f"{name}: move_hz has to be between {MIN_MOVE_HZ:g} and {MAX_MOVE_HZ:g}"
        )
    return (
        speed or DEFAULT_SPEED_DEG_S,
        accel or DEFAULT_ACCEL_DEG_S2,
        move_hz or DEFAULT_MOVE_HZ,
    )


def isaac_arm_attrs(attrs: Mapping[str, Any], name: str) -> Dict[str, Any]:
    """Translate xArm attributes into viam-labs:isaac-sim:arm attributes."""
    speed, accel, move_hz = _limits(attrs, name)
    translated: Dict[str, Any] = {
        "world": attrs.get("world"),
        "asset": ASSET,
        "max_joint_deg_s": speed,
        "max_joint_deg_s2": accel,
        "robot_control_freq_hz": move_hz,
        # each streamed target can move a joint at most what the speed limit
        # allows in one control period, so streamed moves keep the arm's pace
        "path_max_step_deg": speed / move_hz,
    }
    if "mock_dof" in attrs:
        translated["mock_dof"] = attrs["mock_dof"]
    return translated


def _motion_name(attrs: Mapping[str, Any]) -> str:
    return str(attrs.get("motion") or DEFAULT_MOTION)


class UfactoryXArm6(IsaacArm):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "ufactory-xarm6")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._motion: Optional[Motion] = None
        self._motion_name = DEFAULT_MOTION

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        world, _ = validate_sim_component(config, needs_source=False)
        attrs = get_attrs(config)
        _limits(attrs, config.name)
        motion = str(Motion.get_resource_name(_motion_name(attrs)))
        if attrs.get("motion"):
            return [*world, motion], []
        return world, [motion]

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = get_attrs(config)
        arm_config = ComponentConfig()
        arm_config.CopyFrom(config)
        arm_config.attributes.CopyFrom(
            dict_to_struct(isaac_arm_attrs(attrs, config.name))
        )
        super().reconfigure(arm_config, dependencies)
        self._motion_name = _motion_name(attrs)
        motion = dependencies.get(Motion.get_resource_name(self._motion_name))
        self._motion = motion if isinstance(motion, Motion) else None

    async def move_to_position(self, pose: Pose, **kwargs) -> None:
        if self._motion is None:
            raise ValueError(
                f"{self.name} cannot do MoveToPosition without the "
                f"{self._motion_name!r} motion service"
            )
        await self._motion.move(
            component_name=self.name,
            destination=PoseInFrame(reference_frame=f"{self.name}_origin", pose=pose),
        )

    async def get_properties(self, **kwargs) -> Arm.Properties:
        return Arm.Properties(
            support_manual_mode=False,
            support_cartesian_commands=True,
        )
