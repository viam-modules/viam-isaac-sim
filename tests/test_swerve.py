"""Wheel-level swerve bases: per-module steer/speed math, the closed-loop
controller, and module geometry read from a USD stage."""

import math
from types import SimpleNamespace

import pytest

from isaac_module.swerve import (
    SwerveDrive,
    SwerveModule,
    parse_swerve_attrs,
    swerve_targets,
    to_viam_xy,
)

# a square of modules, each rolling forward (viam +y) at steer 0
FORWARD = math.pi / 2
SQUARE = [
    SwerveModule("fl_steer", "fl_drive", (-0.2, 0.2), FORWARD, 1.0),
    SwerveModule("fr_steer", "fr_drive", (0.2, 0.2), FORWARD, 1.0),
    SwerveModule("bl_steer", "bl_drive", (-0.2, -0.2), FORWARD, 1.0),
    SwerveModule("br_steer", "br_drive", (0.2, -0.2), FORWARD, 1.0),
]


def test_forward_keeps_the_wheels_straight():
    angles, speeds = swerve_targets(SQUARE, 0.0, 0.5, 0.0, [0.0] * 4)
    assert list(angles) == pytest.approx([0.0] * 4)
    assert list(speeds) == pytest.approx([0.5] * 4)


def test_strafing_right_turns_each_module_a_quarter_turn_the_short_way():
    # right is clockwise from forward: -90 deg, rolling forward
    angles, speeds = swerve_targets(SQUARE, 0.3, 0.0, 0.0, [0.0] * 4)
    assert list(angles) == pytest.approx([-math.pi / 2] * 4)
    assert list(speeds) == pytest.approx([0.3] * 4)
    # from -80 deg, left is nearer as -90 rolled backward than +90
    angles, speeds = swerve_targets(SQUARE, -0.3, 0.0, 0.0, [math.radians(-80)] * 4)
    assert list(angles) == pytest.approx([-math.pi / 2] * 4)
    assert list(speeds) == pytest.approx([-0.3] * 4)


def test_reversing_rolls_backward_instead_of_turning_around():
    angles, speeds = swerve_targets(SQUARE, 0.0, -0.5, 0.0, [0.0] * 4)
    assert list(angles) == pytest.approx([0.0] * 4)
    assert list(speeds) == pytest.approx([-0.5] * 4)


def test_spinning_left_points_each_module_along_its_circle():
    w = 1.0
    angles, speeds = swerve_targets(SQUARE, 0.0, 0.0, w, [0.0] * 4)
    r = math.hypot(0.2, 0.2)
    for m, angle, speed in zip(SQUARE, angles, speeds):
        x, y = m.position
        # v = w x r: (-w y, w x)
        direction = m.roll0 + m.steer_sign * angle
        assert speed * math.cos(direction) == pytest.approx(-w * y)
        assert speed * math.sin(direction) == pytest.approx(w * x)
        assert abs(speed) == pytest.approx(w * r)
        assert abs(angle) <= math.pi / 2 + 1e-9


def test_a_still_module_keeps_its_angle():
    angles, speeds = swerve_targets(SQUARE, 0.0, 0.0, 0.0, [0.3, -0.2, 0.1, 0.0])
    assert list(angles) == pytest.approx([0.3, -0.2, 0.1, 0.0])
    assert list(speeds) == pytest.approx([0.0] * 4)


def test_a_reversed_steer_axis_mirrors_the_angle():
    flipped = [SwerveModule("s", "d", (0.0, 0.2), FORWARD, -1.0)]
    angles, _ = swerve_targets(flipped, 0.3, 0.0, 0.0, [0.0])
    assert angles[0] == pytest.approx(math.pi / 2)


@pytest.mark.parametrize(
    "axis, file_xy, viam_xy",
    [
        ("+x", (1.0, 0.0), (0.0, 1.0)),  # forward
        ("+x", (0.0, 1.0), (-1.0, 0.0)),  # file +y is left
        ("-x", (-1.0, 0.0), (0.0, 1.0)),  # Mabel: forward is -x
        ("-x", (0.0, 1.0), (1.0, 0.0)),  # ...so file +y is right
        ("+y", (0.0, 1.0), (0.0, 1.0)),
        ("-y", (1.0, 0.0), (-1.0, 0.0)),
    ],
)
def test_file_vectors_turn_into_the_viam_frame(axis, file_xy, viam_xy):
    assert to_viam_xy(file_xy, axis) == pytest.approx(viam_xy)


def test_swerve_attrs_are_checked():
    good = {
        "swerve_modules": [{"steer_joint": "a_steer", "drive_joint": "a_drive"}] * 3,
        "wheel_radius_m": 0.038,
        "forward_axis": "-x",
    }
    parsed = parse_swerve_attrs(good)
    assert parsed["pairs"] == [("a_steer", "a_drive")] * 3
    assert parsed["wheel_radius_m"] == 0.038
    assert parsed["forward_axis"] == "-x"
    assert parse_swerve_attrs({**good, "forward_axis": None})["forward_axis"] == "+x"
    for bad, match in [
        ({"wheel_radius_m": None}, "wheel_radius_m"),
        ({"wheel_radius_m": -1}, "wheel_radius_m"),
        ({"forward_axis": "z"}, "forward_axis"),
        ({"swerve_modules": [{"steer_joint": "a"}]}, "drive_joint"),
        ({"swerve_modules": []}, "swerve_modules"),
        ({"chassis_link": ""}, "chassis_link"),
    ]:
        with pytest.raises(ValueError, match=match):
            parse_swerve_attrs({**good, **bad})


# --- the controller, against an ideal body that does what the wheels say ---


class IdealBody:
    """Integrates the body velocity a set of module speeds implies (least
    squares over the modules), so the test exercises the whole loop."""

    def __init__(self, modules, yaw=0.3):
        self.modules, self.x, self.y, self.yaw, self.t = modules, 1.0, -2.0, yaw, 0.0

    def run(self, drive, dt=1 / 60, seconds=30.0, until=None):
        import numpy as np

        for _ in range(int(seconds / dt)):
            angles, speeds = drive.step(self.x, self.y, self.yaw, dt, self.t)
            rows, rhs = [], []
            for m, angle, speed in zip(self.modules, angles, speeds):
                d = m.roll0 + m.steer_sign * angle
                c, s = math.cos(d), math.sin(d)
                x, y = m.position
                # wheel speed along d of the body velocity (u, v, w)
                rows.append([c, s, -y * c + x * s])
                rhs.append(speed)
            u, v, w = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
            # viam body frame (u right, v forward) into the world, forward = yaw
            fx, fy = math.cos(self.yaw), math.sin(self.yaw)
            rx, ry = fy, -fx
            self.x += (u * rx + v * fx) * dt
            self.y += (u * ry + v * fy) * dt
            self.yaw += w * dt
            self.t += dt
            if until is not None and until():
                return


def _drive(**kw):
    return SwerveDrive(SQUARE, max_linear_mps=0.5, max_angular_rps=1.0, **kw)


def test_move_straight_lands_on_the_distance_along_its_heading():
    drive, body = _drive(), IdealBody(SQUARE)
    x0, y0, yaw0 = body.x, body.y, body.yaw
    goal = drive.start_goal("straight", 1.0, 0.3)
    body.run(drive, until=goal.done.is_set)
    assert goal.done.is_set() and goal.error is None
    along = (body.x - x0) * math.cos(yaw0) + (body.y - y0) * math.sin(yaw0)
    assert along == pytest.approx(1.0, abs=0.005)
    assert body.yaw == pytest.approx(yaw0, abs=math.radians(0.5))
    assert not drive.moving


def test_move_straight_backward():
    drive, body = _drive(), IdealBody(SQUARE)
    x0, yaw0 = body.x, body.yaw
    goal = drive.start_goal("straight", -0.5, 0.3)
    body.run(drive, until=goal.done.is_set)
    assert (body.x - x0) / math.cos(yaw0) == pytest.approx(-0.5, abs=0.005)


def test_spin_lands_on_the_angle():
    drive, body = _drive(), IdealBody(SQUARE)
    yaw0 = body.yaw
    goal = drive.start_goal("spin", math.radians(90), math.radians(45))
    body.run(drive, until=goal.done.is_set)
    assert goal.done.is_set() and goal.error is None
    assert body.yaw - yaw0 == pytest.approx(math.radians(90), abs=math.radians(0.5))


def test_strafing_holds_the_heading_it_set_off_on():
    drive, body = _drive(), IdealBody(SQUARE)
    yaw0 = body.yaw
    drive.set_velocity(0.3, 0.0, 0.0)
    body.run(drive, seconds=0.5)
    body.yaw += 0.1  # a bump twists the base
    body.run(drive, seconds=3.0)
    assert body.yaw == pytest.approx(yaw0, abs=math.radians(1))


def test_diagonal_moves_on_the_diagonal():
    drive, body = _drive(), IdealBody(SQUARE, yaw=math.pi / 2)  # forward = world +y
    drive.set_velocity(0.2, 0.2, 0.0)
    body.run(drive, seconds=2.0)
    dx, dy = body.x - 1.0, body.y + 2.0
    assert dx == pytest.approx(dy, rel=0.02)  # right = world +x
    assert dx > 0.3


def test_set_velocity_clamps_to_the_maxima():
    drive = _drive()
    drive.set_velocity(3.0, 4.0, 9.0)
    assert drive.command == pytest.approx((0.3, 0.4, 1.0))


def test_a_new_command_fails_the_running_goal():
    drive = _drive()
    goal = drive.start_goal("straight", 1.0, 0.3)
    drive.set_velocity(0.0, 0.1, 0.0)
    assert goal.done.is_set() and "superseded" in str(goal.error)
    goal = drive.start_goal("spin", 1.0, 0.3)
    drive.stop()
    assert goal.done.is_set() and "stopped" in str(goal.error)
    assert not drive.moving


def test_a_blocked_goal_fails_as_stalled():
    drive = _drive(stall_s=2.0)
    goal = drive.start_goal("straight", 1.0, 0.3)
    t = 0.0
    while not goal.done.is_set() and t < 10.0:
        drive.step(0.0, 0.0, 0.0, 0.1, t)  # never moves
        t += 0.1
    assert "stalled" in str(goal.error)
    assert t == pytest.approx(2.0, abs=0.2)


def test_misaligned_modules_only_drive_their_aligned_share():
    drive = _drive()
    drive.set_velocity(0.3, 0.0, 0.0)  # steer to -90
    _, speeds = drive.step(0.0, 0.0, 0.0, 0.01, 0.0, measured=[0.0] * 4)
    assert list(speeds) == pytest.approx([0.0] * 4, abs=1e-9)
    _, speeds = drive.step(0.0, 0.0, 0.0, 0.01, 0.01, measured=[-math.pi / 2] * 4)
    assert list(speeds) == pytest.approx([0.3] * 4, abs=0.01)


def test_stopped_wheels_hold_the_last_steer_angles():
    drive = _drive()
    drive.set_velocity(0.3, 0.0, 0.0)
    drive.step(0.0, 0.0, 0.0, 0.01, 0.0)
    drive.stop()
    angles, speeds = drive.step(0.0, 0.0, 0.0, 0.01, 0.01)
    assert list(angles) == pytest.approx([-math.pi / 2] * 4)
    assert list(speeds) == [0.0] * 4


# --- module geometry from a USD stage (Mabel's front-left module) ---


def _mabel_fl_stage():
    """A stage with Mabel's front-left module under /r."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/r")
    UsdGeom.Xform.Define(stage, "/r/base_link")
    # link xforms are whatever physics last wrote (stale, mid-fall): only the
    # joints' authored frames may count
    housing = UsdGeom.Xform.Define(stage, "/r/fl_housing")
    housing.AddTranslateOp().Set(Gf.Vec3d(0.3, 0.5, 0.2))
    housing.AddRotateZOp().Set(-20.0)
    steer_body = UsdGeom.Xform.Define(stage, "/r/fl_steer_body")
    steer_body.AddTranslateOp().Set(Gf.Vec3d(-0.17, -0.17, 0.0))
    steer_body.AddRotateZOp().Set(135.0)  # mid-turn: must not matter
    UsdGeom.Xform.Define(stage, "/r/fl_wheel")

    fixed = UsdPhysics.FixedJoint.Define(stage, "/r/joints/fl_housing_fixed")
    fixed.CreateBody0Rel().SetTargets(["/r/base_link"])
    fixed.CreateBody1Rel().SetTargets(["/r/fl_housing"])
    half = math.sqrt(0.5)
    fixed.CreateLocalPos0Attr(Gf.Vec3f(-0.17, -0.17, 0.0))
    fixed.CreateLocalRot0Attr(Gf.Quatf(half, 0, 0, half))  # yaw 90
    fixed.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    fixed.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    steer = UsdPhysics.RevoluteJoint.Define(stage, "/r/joints/fl_steer")
    steer.CreateBody0Rel().SetTargets(["/r/fl_housing"])
    steer.CreateBody1Rel().SetTargets(["/r/fl_steer_body"])
    steer.CreateAxisAttr("Z")
    steer.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    steer.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    steer.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    steer.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    drive = UsdPhysics.RevoluteJoint.Define(stage, "/r/joints/fl_drive")
    drive.CreateBody0Rel().SetTargets(["/r/fl_steer_body"])
    drive.CreateBody1Rel().SetTargets(["/r/fl_wheel"])
    drive.CreateAxisAttr("Z")
    drive.CreateLocalPos0Attr(Gf.Vec3f(0, 0, -0.04))
    # URDF rpy (0, -pi/2, 0): the axle is the steer body's -x
    drive.CreateLocalRot0Attr(Gf.Quatf(half, 0, -half, 0))
    return stage


def test_modules_are_read_from_the_stage():
    pytest.importorskip("pxr")
    from isaac_module.swerve import stage_modules

    stage = _mabel_fl_stage()
    modules = stage_modules(
        stage.GetPrimAtPath("/r"), "/r/base_link", [("fl_steer", "fl_drive")], "-x"
    )
    (m,) = modules
    assert (m.steer, m.drive) == ("fl_steer", "fl_drive")
    assert m.position == pytest.approx((-0.17, 0.17))  # front left
    assert m.roll0 == pytest.approx(math.pi / 2)  # rolls forward at steer 0
    assert m.steer_sign == 1.0

    with pytest.raises(ValueError, match="nope"):
        stage_modules(stage.GetPrimAtPath("/r"), "/r/base_link", [("nope", "fl_drive")], "-x")


# --- the sim-side handle: steer positions and wheel velocities ---


def test_handle_drives_steer_positions_and_wheel_speeds():
    import numpy as np

    from isaac_module.sim_manager import SwerveBaseHandle

    actions = []
    art = SimpleNamespace(
        get_joint_positions=lambda joint_indices: np.zeros(len(joint_indices)),
        apply_action=lambda action: actions.append(action),
    )
    chassis = SimpleNamespace(get_world_pose=lambda: (np.zeros(3), np.array([1.0, 0, 0, 0])))
    sim = SimpleNamespace(
        _isaac=SimpleNamespace(ArticulationAction=lambda **kw: kw),
        world=SimpleNamespace(current_time=0.0),
    )
    drive = SwerveDrive(SQUARE, max_linear_mps=0.5, max_angular_rps=1.0)
    handle = SwerveBaseHandle(sim, art, chassis, drive, [0, 2, 4, 6], [1, 3, 5, 7], 0.05, "+x")
    handle.set_velocity(0.25, 0.0)  # forward m/s
    handle._on_physics_step(1 / 60)
    steer, wheels = actions[-2:]
    assert list(steer["joint_indices"]) == [0, 2, 4, 6]
    assert list(steer["joint_positions"]) == pytest.approx([0.0] * 4)
    assert list(wheels["joint_indices"]) == [1, 3, 5, 7]
    assert list(wheels["joint_velocities"]) == pytest.approx([5.0] * 4)  # 0.25 / 0.05
    assert handle.is_moving()
    handle.stop()
    assert not handle.is_moving()


def test_swerve_base_needs_the_arm_that_loaded_its_articulation():
    from isaac_module.sim_manager import SimManager

    manager = object.__new__(SimManager)
    manager._arm_articulations = {}
    attrs = {
        "prim_path": "/World/mabel",
        "swerve_modules": [{"steer_joint": "a", "drive_joint": "b"}] * 2,
        "wheel_radius_m": 0.04,
    }
    with pytest.raises(ValueError, match="depends_on"):
        manager._create_base_isaac("base", attrs)
    with pytest.raises(ValueError, match="wheel_radius_m"):
        manager._create_base_isaac("base", {**attrs, "wheel_radius_m": None})


def test_move_straight_and_spin_run_closed_loop_goals_on_a_swerve_handle():
    import asyncio

    from isaac_module.models.base import IsaacBase

    calls = []
    handle = SimpleNamespace(
        run_goal=lambda kind, target, rate: calls.append((kind, target, rate)),
        stop=lambda: calls.append("stop"),
    )
    base = IsaacBase("b")
    base._handle = handle
    asyncio.run(base.move_straight(1000, -300))  # back 1 m
    asyncio.run(base.spin(-90, 45))  # right a quarter turn
    assert calls[0] == ("straight", pytest.approx(-1.0), pytest.approx(0.3))
    assert calls[2] == ("spin", pytest.approx(-math.pi / 2), pytest.approx(math.pi / 4))
    assert calls[1] == calls[3] == "stop"


def test_a_stopped_base_brakes_its_wheels_where_they_stopped():
    import numpy as np

    from isaac_module.sim_manager import SWERVE_BRAKE_GAIN, SwerveBaseHandle

    joints = np.zeros(8)
    actions = []
    art = SimpleNamespace(
        get_joint_positions=lambda joint_indices: joints[list(joint_indices)].copy(),
        apply_action=lambda action: actions.append(action),
    )
    chassis = SimpleNamespace(get_world_pose=lambda: (np.zeros(3), np.array([1.0, 0, 0, 0])))
    sim = SimpleNamespace(
        _isaac=SimpleNamespace(ArticulationAction=lambda **kw: kw),
        world=SimpleNamespace(current_time=0.0),
    )
    drive = SwerveDrive(SQUARE, max_linear_mps=0.5, max_angular_rps=1.0)
    handle = SwerveBaseHandle(sim, art, chassis, drive, [0, 2, 4, 6], [1, 3, 5, 7], 0.05, "+x")
    handle._on_physics_step(1 / 60)  # idle: brakes where the wheels are
    assert list(actions[-1]["joint_velocities"]) == pytest.approx([0.0] * 4)
    joints[[1, 3]] = 0.1  # something shoves the base
    handle._on_physics_step(1 / 60)
    assert list(actions[-1]["joint_velocities"]) == pytest.approx(
        [-SWERVE_BRAKE_GAIN * 0.1, -SWERVE_BRAKE_GAIN * 0.1, 0.0, 0.0]
    )
    handle.set_velocity(0.25, 0.0)  # driving releases the brake
    handle._on_physics_step(1 / 60)
    assert list(actions[-1]["joint_velocities"]) == pytest.approx([5.0] * 4)
    handle.stop()  # and stopping sets it again where the wheels are now
    joints[[1, 3, 5, 7]] = 2.0
    handle._on_physics_step(1 / 60)
    assert list(actions[-1]["joint_velocities"]) == pytest.approx([0.0] * 4)


def test_status_reports_each_module():
    import numpy as np

    from isaac_module.sim_manager import SwerveBaseHandle

    art = SimpleNamespace(
        get_joint_positions=lambda joint_indices: np.full(len(joint_indices), 0.5),
        get_joint_velocities=lambda joint_indices: np.full(len(joint_indices), 2.0),
        apply_action=lambda action: None,
        get_articulation_controller=lambda: SimpleNamespace(
            get_gains=lambda: (np.arange(8.0), np.arange(8.0) * 10)
        ),
    )
    on_sim_thread = []
    sim = SimpleNamespace(
        _isaac=SimpleNamespace(ArticulationAction=lambda **kw: kw),
        run=lambda fn: on_sim_thread.append(fn) or fn(),
    )
    drive = SwerveDrive(SQUARE, max_linear_mps=0.5, max_angular_rps=1.0)
    handle = SwerveBaseHandle(sim, art, None, drive, [0, 2, 4, 6], [1, 3, 5, 7], 0.05, "+x")
    status = handle.status()
    assert on_sim_thread  # the articulation is read on the sim thread
    fl = status["modules"][0]
    assert fl["steer"] == "fl_steer" and fl["drive"] == "fl_drive"
    assert fl["steer_deg"] == pytest.approx(math.degrees(0.5))
    assert fl["steer_target_deg"] == 0.0
    assert fl["wheel_rad_s"] == 2.0 and fl["wheel_rad"] == 0.5
    assert fl["steer_gains"] == [0.0, 0.0] and fl["wheel_gains"] == [1.0, 10.0]
    assert fl["position_m"] == [-0.2, 0.2]
    assert status["moving"] is False


def test_wheels_become_force_velocity_drives():
    # URDF imports default to acceleration drives, where the damping is per unit
    # of the wheel's own (tiny) inertia: no torque to move the robot
    pytest.importorskip("pxr")
    import numpy as np
    from pxr import Usd, UsdPhysics

    from isaac_module.sim_manager import SWERVE_WHEEL_ARMATURE, _velocity_drives

    stage = Usd.Stage.CreateInMemory()
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/r/joints/fl_drive")
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("acceleration")
    drive.CreateStiffnessAttr(1000.0)
    gains = {}
    controller = SimpleNamespace(
        get_gains=lambda: (np.array([[5.0, 7.0]]), np.array([[1.0, 2.0]])),
        set_gains=lambda kps, kds: gains.update(kps=kps, kds=kds),
    )
    art = SimpleNamespace(dof_names=["fl_steer", "fl_drive"], get_articulation_controller=lambda: controller)
    _velocity_drives(stage.GetPrimAtPath("/r"), art, ["fl_drive"], 10.0, lambda: None)
    assert drive.GetTypeAttr().Get() == "force"
    # a ~1e-4 kg m^2 wheel under a 40 kg robot: armature evens out the contact's mass ratio
    assert joint.GetPrim().GetAttribute("physxJoint:armature").Get() == pytest.approx(SWERVE_WHEEL_ARMATURE)
    assert drive.GetStiffnessAttr().Get() == 0.0
    assert drive.GetDampingAttr().Get() == pytest.approx(10.0 * math.pi / 180)
    assert gains["kps"].tolist() == [[5.0, 0.0]] and gains["kds"].tolist() == [[1.0, 10.0]]


def test_the_base_rebuilds_physics_after_making_the_wheels_force_drives():
    # PhysX reads a drive's type when it builds the articulation on play; set
    # while playing, the import's acceleration type can stick
    pytest.importorskip("pxr")
    import numpy as np
    from pxr import UsdPhysics

    from isaac_module.sim_manager import SimManager

    stage = _mabel_fl_stage()
    drive_type = UsdPhysics.DriveAPI(stage.GetPrimAtPath("/r/joints/fl_drive"), "angular")
    seen = []
    controller = SimpleNamespace(
        get_gains=lambda: (np.zeros(2), np.zeros(2)), set_gains=lambda kps, kds: None
    )
    manager = object.__new__(SimManager)
    manager._arm_articulations = {
        "/r": SimpleNamespace(dof_names=["fl_steer", "fl_drive"], get_articulation_controller=lambda: controller)
    }
    manager._isaac = SimpleNamespace(
        get_prim_at_path=stage.GetPrimAtPath, SingleXFormPrim=lambda path: path
    )
    manager._require_prim = lambda path: None
    manager._reset_world = lambda: seen.append(drive_type.GetTypeAttr().Get())
    manager.world = SimpleNamespace(add_physics_callback=lambda name, fn: None)
    attrs = {
        "prim_path": "/r",
        "swerve_modules": [{"steer_joint": "fl_steer", "drive_joint": "fl_drive"}] * 2,
        "wheel_radius_m": 0.038,
        "forward_axis": "-x",
    }
    manager._create_swerve_base_isaac("base", attrs)
    assert seen == ["force"]


def test_the_brake_sees_through_wheel_angles_wrapping_at_pi():
    # PhysX reports a continuous joint's position in (-pi, pi]
    import numpy as np

    from isaac_module.sim_manager import SWERVE_BRAKE_GAIN, SwerveBaseHandle

    joints = np.zeros(8)
    joints[[1, 3, 5, 7]] = math.pi - 0.01
    actions = []
    art = SimpleNamespace(
        get_joint_positions=lambda joint_indices: joints[list(joint_indices)].copy(),
        apply_action=lambda action: actions.append(action),
    )
    chassis = SimpleNamespace(get_world_pose=lambda: (np.zeros(3), np.array([1.0, 0, 0, 0])))
    sim = SimpleNamespace(
        _isaac=SimpleNamespace(ArticulationAction=lambda **kw: kw),
        world=SimpleNamespace(current_time=0.0),
    )
    drive = SwerveDrive(SQUARE, max_linear_mps=0.5, max_angular_rps=1.0)
    handle = SwerveBaseHandle(sim, art, chassis, drive, [0, 2, 4, 6], [1, 3, 5, 7], 0.05, "+x")
    handle._on_physics_step(1 / 60)
    joints[[1, 3, 5, 7]] = -math.pi + 0.01  # rolled 0.02 rad forward, across the wrap
    handle._on_physics_step(1 / 60)
    assert list(actions[-1]["joint_velocities"]) == pytest.approx([-SWERVE_BRAKE_GAIN * 0.02] * 4)


def test_a_timed_set_velocity_runs_for_its_sim_seconds_then_stops(monkeypatch):
    # agents whose calls carry no session (each SetVelocity is stopped when the
    # call returns) can still hold a velocity for a while
    import asyncio
    import itertools

    from isaac_module.models import base as base_model

    clock = itertools.count(0.0, 0.5)
    monkeypatch.setattr(base_model.SimManager, "get", lambda: SimpleNamespace(sim_time_s=lambda: next(clock)))
    calls = []
    handle = SimpleNamespace(
        set_velocity=lambda *args: calls.append(("v", *args)),
        stop=lambda: calls.append("stop"),
        is_moving=lambda: True,
    )
    base = base_model.IsaacBase("b")
    base._handle = handle
    asyncio.run(base.do_command({
        "command": "set_velocity", "linear": {"x": 300, "y": 100}, "angular": {"z": 0}, "duration_s": 2,
    }))
    assert calls[0] == ("v", pytest.approx(0.1), 0.0, pytest.approx(-0.3))
    assert calls[-1] == "stop" and len(calls) == 2


def test_translating_while_spinning_adds_each_modules_spin_share():
    u, v, w = 0.1, 0.3, 0.8
    angles, speeds = swerve_targets(SQUARE, u, v, w, [0.0] * 4)
    for m, angle, speed in zip(SQUARE, angles, speeds):
        x, y = m.position
        direction = m.roll0 + m.steer_sign * angle
        assert speed * math.cos(direction) == pytest.approx(u - w * y)
        assert speed * math.sin(direction) == pytest.approx(v + w * x)


def _swerve_handle(joints):
    import numpy as np

    from isaac_module.sim_manager import SwerveBaseHandle

    actions = []
    art = SimpleNamespace(
        get_joint_positions=lambda joint_indices: joints[list(joint_indices)].copy(),
        apply_action=lambda action: actions.append(action),
    )
    chassis = SimpleNamespace(get_world_pose=lambda: (np.zeros(3), np.array([1.0, 0, 0, 0])))
    sim = SimpleNamespace(
        _isaac=SimpleNamespace(ArticulationAction=lambda **kw: kw),
        world=SimpleNamespace(current_time=0.0),
    )
    drive = SwerveDrive(SQUARE, max_linear_mps=0.5, max_angular_rps=1.0)
    handle = SwerveBaseHandle(sim, art, chassis, drive, [0, 2, 4, 6], [1, 3, 5, 7], 0.05, "+x")
    return handle, drive, actions


def test_steer_targets_go_to_the_nearest_turn_of_the_joint():
    # the controller's angles grow without bound as the travel direction keeps
    # turning; the joint gets the equivalent angle nearest where it is (a world
    # reset puts the joints back at 0)
    import numpy as np

    joints = np.zeros(8)
    handle, drive, actions = _swerve_handle(joints)
    drive.angles = [4 * math.pi + 0.2] * 4
    handle._on_physics_step(1 / 60)
    assert list(actions[-2]["joint_positions"]) == pytest.approx([0.2] * 4)


def test_a_world_reset_clears_a_swerve_bases_motion_state():
    import numpy as np

    from isaac_module.sim_manager import SimManager

    joints = np.zeros(8)
    handle, drive, actions = _swerve_handle(joints)
    handle._on_physics_step(1 / 60)  # stopped: brake latched at 0
    goal = drive.start_goal("straight", 1.0, 0.3)
    handle._on_physics_step(1 / 60)  # the goal took its start pose
    manager = object.__new__(SimManager)
    manager._handles = {"base": ({}, handle)}
    manager._shared_bases = []

    def world_reset():
        joints[[1, 3, 5, 7]] = 5.0  # wheels come back somewhere else

    manager.world = SimpleNamespace(reset=world_reset)
    manager._reset_world()
    assert goal.done.is_set() and "reset" in str(goal.error)  # a running goal doesn't jump
    assert not drive.moving and drive._yaw is None and drive._hold is None
    handle._on_physics_step(1 / 60)
    assert list(actions[-1]["joint_velocities"]) == pytest.approx([0.0] * 4)  # no roll back


def test_a_timed_set_velocity_yields_to_a_newer_command(monkeypatch):
    import asyncio
    import itertools

    from isaac_module.models import base as base_model
    from viam.proto.common import Vector3

    clock = itertools.count(0.0, 0.5)
    monkeypatch.setattr(base_model.SimManager, "get", lambda: SimpleNamespace(sim_time_s=lambda: next(clock)))
    calls = []
    handle = SimpleNamespace(
        set_velocity=lambda *args: calls.append(("v", *args)),
        stop=lambda: calls.append("stop"),
        is_moving=lambda: True,
    )
    base = base_model.IsaacBase("b")
    base._handle = handle

    async def both():
        hold = asyncio.create_task(base.do_command(
            {"command": "set_velocity", "linear": {"y": 100}, "duration_s": 60}
        ))
        await asyncio.sleep(0.05)
        await base.set_velocity(Vector3(y=200), Vector3())
        await hold

    asyncio.run(both())
    assert "stop" not in calls  # the newer command keeps running
    with pytest.raises(ValueError, match="duration_s"):
        asyncio.run(base.do_command({"command": "set_velocity", "linear": {"y": 1}}))
    for bad in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="duration_s"):
            asyncio.run(base.do_command({"command": "set_velocity", "linear": {"y": 1}, "duration_s": bad}))


def test_a_goal_the_sim_never_finishes_times_out(monkeypatch):
    import numpy as np

    from isaac_module import sim_manager

    monkeypatch.setattr(sim_manager, "SWERVE_GOAL_MIN_WAIT_S", 0.05)
    monkeypatch.setattr(sim_manager, "SWERVE_GOAL_WAIT_FACTOR", 0.0)
    handle, drive, _ = _swerve_handle(np.zeros(8))
    with pytest.raises(TimeoutError):
        handle.run_goal("straight", 1.0, 0.3)  # no physics steps run
    assert not drive.moving
