# ruff: noqa: RUF012
import asyncio
import json
import sys
from types import SimpleNamespace
import numpy as np

import pytest
from grpclib import Status
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module import sim_manager
from isaac_module.models.arm import IsaacArm
from isaac_module.models.gripper import IsaacGripper
from isaac_module.sim_manager import (
    IsaacArmHandle,
    IsaacGripperHandle,
    MockGripperHandle,
    SimManager,
)


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


@pytest.fixture
def manager(monkeypatch):
    value = SimManager()
    value.mock = True
    value._booted.set()
    monkeypatch.setattr(SimManager, "_instance", value)
    return value


def _arm(asset: str = "ur20") -> IsaacArm:
    return IsaacArm.new(
        _config("pick-arm", {"world": "sim-world", "asset": asset, "mock_dof": 6}),
        {},
    )


def _gripper(extra: dict | None = None) -> IsaacGripper:
    attrs = {
        "world": "sim-world",
        "arm": "pick-arm",
        "closed_deg": 10,
        "grab_timeout_sec": 1,
    }
    attrs.update(extra or {})
    return IsaacGripper.new(_config("pick-gripper", attrs), {})


def test_validate_requires_world_and_arm():
    with pytest.raises(ValueError, match="world"):
        IsaacGripper.validate_config(_config("g", {"arm": "a"}))
    with pytest.raises(ValueError, match="arm"):
        IsaacGripper.validate_config(_config("g", {"world": "w"}))


def test_validate_returns_both_dependencies_and_checks_frame_parent():
    config = _config("g", {"world": "sim-world", "arm": "pick-arm"})
    config.frame.parent = "pick-arm"
    dependencies, implicit = IsaacGripper.validate_config(config)
    assert list(dependencies) == ["sim-world", "pick-arm"]
    assert list(implicit) == []

    config.frame.parent = "other-arm"
    with pytest.raises(ValueError, match="frame.parent"):
        IsaacGripper.validate_config(config)


def test_validate_rejects_unknown_or_unsupported_jaw_layout():
    base = {"world": "sim-world", "arm": "pick-arm"}
    with pytest.raises(ValueError, match="unknown Lite6 jaw_layout"):
        IsaacGripper.validate_config(
            _config("g", {**base, "asset": "lite6_gripper", "jaw_layout": "left"})
        )
    with pytest.raises(ValueError, match="only supported"):
        IsaacGripper.validate_config(
            _config("g", {**base, "asset": "robotiq_2f_85", "jaw_layout": "wide"})
        )


def test_validate_rejects_non_boolean_sensorless():
    with pytest.raises(ValueError, match="sensorless must be a boolean"):
        IsaacGripper.validate_config(
            _config(
                "g",
                {"world": "sim-world", "arm": "pick-arm", "sensorless": "true"},
            )
        )


def test_manager_requires_an_attached_arm(manager):
    with pytest.raises(ValueError, match="not attached"):
        manager.create_gripper("g", {"arm": "missing", "asset": "robotiq_2f_85"})


def test_manager_rejects_non_gripper_asset(manager):
    _arm()
    with pytest.raises(ValueError, match="unknown gripper asset"):
        manager.create_gripper("g", {"arm": "pick-arm", "asset": "ur20"})


def test_lite6_gripper_requires_lite6_arm(manager):
    _arm()
    with pytest.raises(ValueError, match="requires arm asset 'lite6'"):
        manager.create_gripper(
            "pick-gripper", {"arm": "pick-arm", "asset": "lite6_gripper"}
        )


@pytest.mark.parametrize(
    ("jaw_layout", "tip_signs"),
    (
        ("narrow", (-1.0, 1.0)),
        ("middle_positive_y", (1.0, 1.0)),
        ("middle_negative_y", (-1.0, -1.0)),
        ("wide", (1.0, -1.0)),
    ),
)
def test_integrated_lite6_gripper_authors_compound_collisions(
    monkeypatch, jaw_layout, tip_signs
):
    initialization_events = []

    class FakeArticulation:
        dof_names = ["joint1", "finger_joint1", "finger_joint2"]

        @property
        def dof_properties(self):
            assert initialization_events[-2:] == [("reset",), ("refresh",)]
            return {
                "lower": [-1.0, 0.0, -0.00178],
                "upper": [1.0, 0.0089, 0.01068],
            }

    arm = object.__new__(IsaacArmHandle)
    arm._prim_path = "/World/pick_arm"
    arm._art = FakeArticulation()
    arm.refresh_dofs = lambda: initialization_events.append(("refresh",))

    manager = object.__new__(SimManager)
    created = {}
    cubes = {}
    collision_applications = []
    collision_enabled = {}
    material_bindings = {}
    contact_offsets = {}
    visual_rotations = {}
    visual_translations = {}
    mimic_attributes = {}
    contact_views = []

    class FakeOp:
        def __init__(self, op_type, setter):
            self.op_type = op_type
            self.setter = setter

        def GetOpType(self):
            return self.op_type

        def Set(self, value):
            self.setter(value)

    class FakePrim:
        def __init__(self, path, stage=None):
            self.path = path
            self._stage = stage
            self.ops = {}

        def IsValid(self):
            return True

        def GetStage(self):
            return self._stage

        def GetAttribute(self, name):
            return SimpleNamespace(
                Set=lambda value: mimic_attributes.update({(self.path, name): value})
            )

    class FakeXformOps:
        def __init__(self, prim):
            self.prim = prim

        def GetOrderedXformOps(self):
            return list(self.prim.ops.values())

        def _add_op(self, op_type, setter):
            if op_type in self.prim.ops:
                raise RuntimeError(f"duplicate xform op {op_type}")
            op = FakeOp(op_type, setter)
            self.prim.ops[op_type] = op
            return op

    class Proxy(FakeXformOps):
        def __init__(self, path):
            state = cubes.setdefault(path, {"prim": FakePrim(path)})
            self.path = path
            super().__init__(state["prim"])

        def CreateSizeAttr(self, value):
            cubes[self.path]["size"] = value

        def AddTranslateOp(self):
            return self._add_op(
                "translate", lambda value: cubes[self.path].update(translate=value)
            )

        def AddScaleOp(self):
            return self._add_op(
                "scale", lambda value: cubes[self.path].update(scale=value)
            )

        def CreateVisibilityAttr(self, value):
            cubes[self.path]["visibility"] = value

        def GetPrim(self):
            return self.prim

    class FakeXformable(FakeXformOps):
        def AddTranslateOp(self):
            return self._add_op(
                "translate",
                lambda value: visual_translations.update({self.prim.path: value}),
            )

        def AddRotateZOp(self):
            return self._add_op(
                "rotate_z",
                lambda value: visual_rotations.update({self.prim.path: value}),
            )

        def SetXformOpOrder(self, ops):
            self.prim.ops = {op.GetOpType(): op for op in ops}

    class FakeCollisionAPI:
        def __init__(self, prim):
            self.prim = prim

        def CreateCollisionEnabledAttr(self, value):
            collision_enabled[self.prim.path] = value

        @staticmethod
        def Apply(prim):
            collision_applications.append(prim.path)

    class FakePhysxCollisionAPI:
        def __init__(self, prim):
            self.prim = prim

        @classmethod
        def Apply(cls, prim):
            return cls(prim)

        def CreateRestOffsetAttr(self, value):
            contact_offsets[(self.prim.path, "rest")] = value

        def CreateContactOffsetAttr(self, value):
            contact_offsets[(self.prim.path, "contact")] = value

    class FakeContactView:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.initialized = False
            contact_views.append(self)

        def initialize(self):
            self.initialized = True

    class Binding:
        def __init__(self, prim):
            self.prim = prim

        def Bind(self, material, **kwargs):
            material_bindings[self.prim.path] = (material, kwargs)

    material = object()
    stage = SimpleNamespace(GetPrimAtPath=lambda path: prims.get(path))
    prims = {
        "/World/pick_arm/gripper": FakePrim("/World/pick_arm/gripper", stage),
        "/World/pick_arm/gripper/uflite_gripper_link": FakePrim(
            "/World/pick_arm/gripper/uflite_gripper_link"
        ),
        "/World/pick_arm/gripper/uflite_finger1/visuals": FakePrim(
            "/World/pick_arm/gripper/uflite_finger1/visuals"
        ),
        "/World/pick_arm/gripper/uflite_finger1/collisions": FakePrim(
            "/World/pick_arm/gripper/uflite_finger1/collisions"
        ),
        "/World/pick_arm/gripper/uflite_finger2/collisions": FakePrim(
            "/World/pick_arm/gripper/uflite_finger2/collisions"
        ),
        "/World/pick_arm/gripper/uflite_finger2/visuals": FakePrim(
            "/World/pick_arm/gripper/uflite_finger2/visuals"
        ),
        "/World/pick_arm/gripper/joints/finger_joint2": FakePrim(
            "/World/pick_arm/gripper/joints/finger_joint2"
        ),
    }
    manager._prop_specs = {"pick_block": {}}
    manager.cfg = SimpleNamespace(physics_dt=0.02)
    manager._isaac = SimpleNamespace(
        get_prim_at_path=lambda path: prims.get(path),
        PhysicsMaterial=lambda **kwargs: SimpleNamespace(material=material),
        SingleXFormPrim=lambda path: SimpleNamespace(path=path),
        RigidContactView=FakeContactView,
    )
    manager.world = SimpleNamespace(
        reset=lambda: initialization_events.append(("reset",))
    )
    monkeypatch.setitem(
        sys.modules,
        "pxr",
        SimpleNamespace(
            PhysxSchema=SimpleNamespace(PhysxCollisionAPI=FakePhysxCollisionAPI),
            Gf=SimpleNamespace(Vec3d=lambda *values: values),
            UsdGeom=SimpleNamespace(
                Cube=SimpleNamespace(Define=lambda stage, path: Proxy(path)),
                Tokens=SimpleNamespace(invisible="invisible"),
                XformOp=SimpleNamespace(
                    TypeRotateZ="rotate_z", TypeTranslate="translate", TypeScale="scale"
                ),
                Xformable=FakeXformable,
            ),
            UsdPhysics=SimpleNamespace(CollisionAPI=FakeCollisionAPI),
            UsdShade=SimpleNamespace(
                MaterialBindingAPI=SimpleNamespace(Apply=lambda prim: Binding(prim)),
                Tokens=SimpleNamespace(strongerThanDescendants="stronger"),
            ),
        ),
    )

    def fake_handle(*args, **kwargs):
        created["args"] = args
        created["kwargs"] = kwargs
        return created

    monkeypatch.setattr(sim_manager, "IsaacGripperHandle", fake_handle)
    result = manager._create_integrated_gripper_isaac(
        "pick-gripper",
        {"jaw_layout": jaw_layout},
        arm,
        sim_manager.KNOWN_ASSETS["lite6_gripper"],
    )

    assert result is created
    assert created["args"][3:5] == pytest.approx((0.0089, 0.0))
    assert initialization_events == [("reset",), ("refresh",)]
    assert created["kwargs"] == {
        "position_unit": "m",
        "minimum_grasp_closure": 0.05,
        "binary": True,
        "coupled_drive_joints": ("finger_joint2",),
        "drive_stiffness": 300.0,
        "drive_damping": 5.0,
        "contact_view": contact_views[0],
        "contact_dt": 0.02,
    }
    assert contact_views[0].initialized is True
    assert contact_views[0].kwargs["prim_paths_expr"] == [
        "/World/pick_arm/gripper/uflite_finger1/viam_collision/outer_tip",
        "/World/pick_arm/gripper/uflite_finger2/viam_collision/outer_tip",
    ]
    assert contact_views[0].kwargs["filter_paths_expr"] == [
        ["/World/pick_block"],
        ["/World/pick_block"],
    ]
    assert mimic_attributes == {
        (
            "/World/pick_arm/gripper/joints/finger_joint2",
            "physxMimicJoint:rotY:naturalFrequency",
        ): 1000.0,
        (
            "/World/pick_arm/gripper/joints/finger_joint2",
            "physxMimicJoint:rotY:dampingRatio",
        ): 1.0,
    }
    expected_geometry = {}
    expected_visual_translations = {}
    for finger_index, (natural_y_sign, y_sign) in enumerate(
        zip((1.0, -1.0), tip_signs), start=1
    ):
        prefix = f"/World/pick_arm/gripper/uflite_finger{finger_index}"
        mount_offset_y = -y_sign * 0.015600001 if y_sign != natural_y_sign else 0.0
        expected_visual_translations[f"{prefix}/visuals"] = (
            0.0,
            mount_offset_y,
            0.0,
        )
        expected_geometry[f"{prefix}/viam_collision/base"] = {
            "translate": (
                0.0,
                mount_offset_y + y_sign * 0.0078000005,
                0.0034,
            ),
            "scale": (0.010000002, 0.0078000005, 0.0034),
        }
        expected_geometry[f"{prefix}/viam_collision/outer_tip"] = {
            "translate": (
                0.0,
                mount_offset_y + y_sign * 0.0134000005,
                0.012050001,
            ),
            "scale": (0.010000002, 0.0022000005, 0.014750002),
        }
    assert visual_rotations == {
        "/World/pick_arm/gripper/uflite_finger1/visuals": (
            0.0 if tip_signs[0] == 1.0 else 180.0
        ),
        "/World/pick_arm/gripper/uflite_finger2/visuals": (
            0.0 if tip_signs[1] == -1.0 else 180.0
        ),
    }
    assert visual_translations == expected_visual_translations
    assert set(cubes) == set(expected_geometry)
    for path, expected in expected_geometry.items():
        assert cubes[path]["translate"] == expected["translate"]
        assert cubes[path]["scale"] == expected["scale"]
        assert cubes[path]["size"] == 1.0
        assert cubes[path]["visibility"] == "invisible"
    assert collision_applications == list(expected_geometry)
    assert set(material_bindings) == set(expected_geometry)
    for bound_material, binding_options in material_bindings.values():
        assert bound_material is material
        assert binding_options == {
            "bindingStrength": "stronger",
            "materialPurpose": "physics",
        }
    assert collision_enabled == {
        "/World/pick_arm/gripper/uflite_finger1/collisions": False,
        "/World/pick_arm/gripper/uflite_finger2/collisions": False,
    }
    for path in expected_geometry:
        assert contact_offsets[(path, "rest")] == 0.0
        assert contact_offsets[(path, "contact")] == 0.001
    # Reconfigure can rebuild the resource against the same stage. Existing
    # xform ops must be reused rather than authored a second time.
    manager._create_integrated_gripper_isaac(
        "pick-gripper",
        {"jaw_layout": jaw_layout},
        arm,
        sim_manager.KNOWN_ASSETS["lite6_gripper"],
    )


def test_handle_is_cached_and_arm_keeps_six_joint_contract(manager):
    arm = _arm()
    first = manager.create_gripper(
        "pick-gripper", {"arm": "pick-arm", "asset": "robotiq_2f_85"}
    )
    second = manager.create_gripper(
        "pick-gripper", {"arm": "pick-arm", "asset": "robotiq_2f_85"}
    )

    assert isinstance(first, MockGripperHandle)
    assert second is first
    assert len(asyncio.run(arm.get_joint_positions()).values) == 6


def test_open_grab_release_and_normalized_input(manager):
    _arm()
    gripper = _gripper({"mock_object_width_m": 0.04})

    async def scenario():
        await gripper.open()
        assert await gripper.get_current_inputs() == pytest.approx([0.0])
        assert await gripper.grab() is True
        holding = await gripper.is_holding_something()
        assert holding.is_holding_something is True
        assert 0.0 < holding.meta["input"] < 1.0

        await gripper.open()
        assert (await gripper.is_holding_something()).is_holding_something is False
        await gripper.go_to_inputs([0.5])
        assert await gripper.get_current_inputs() == pytest.approx([0.5], abs=1e-3)

    asyncio.run(scenario())


def test_lite6_gripper_uses_nonblocking_binary_commands(manager):
    _arm("lite6")
    gripper = _gripper({"asset": "lite6_gripper", "mock_object_width_m": 0.01})

    async def scenario():
        await gripper.open()
        assert await gripper.get_current_inputs() == pytest.approx([0.0])
        assert await gripper.grab() is True
        assert await gripper.get_current_inputs() == pytest.approx([1.0])
        assert await gripper.is_moving()
        await gripper._wait_stopped()
        status = await gripper.is_holding_something()
        assert status.is_holding_something is True
        assert 0.0 < status.meta["input"] < 1.0
        assert status.meta["open_m"] > status.meta["closed_m"]
        with pytest.raises(Exception) as error:
            await gripper.go_to_inputs([0.5])
        assert error.value.grpc_code == Status.INVALID_ARGUMENT

    asyncio.run(scenario())


def test_lite6_gripper_reports_command_success_before_failed_grasp(manager):
    _arm("lite6")
    gripper = _gripper({"asset": "lite6_gripper", "mock_object_width_m": 0.02})

    async def scenario():
        assert await gripper.grab() is True
        await gripper._wait_stopped()
        status = await gripper.is_holding_something()
        assert status.is_holding_something is False
        assert status.meta["firm_grasp"] is False

    asyncio.run(scenario())


def test_sensorless_holding_tracks_active_grip_commands(manager):
    _arm("lite6")
    gripper = _gripper({"asset": "lite6_gripper", "sensorless": True})

    async def scenario():
        assert (await gripper.is_holding_something()).is_holding_something is False

        assert await gripper.grab() is True
        status = await gripper.is_holding_something()
        assert status.is_holding_something is True
        assert status.meta["firm_grasp"] is False

        await gripper.stop()
        assert (await gripper.is_holding_something()).is_holding_something is False

        await gripper.go_to_inputs([1.0])
        assert (await gripper.is_holding_something()).is_holding_something is True

        await gripper.open()
        assert (await gripper.is_holding_something()).is_holding_something is False

        await gripper.do_command({"command": "set_input", "value": 1.0})
        assert (await gripper.is_holding_something()).is_holding_something is True

        await gripper.do_command({"command": "set_input", "value": 0.0})
        assert (await gripper.is_holding_something()).is_holding_something is False

    asyncio.run(scenario())


def test_sensorless_status_exposes_raw_firm_grasp_independently():
    class Handle:
        @staticmethod
        def jaw_limits():
            return 0.0, 1.0

        @staticmethod
        def get_jaw():
            return 0.5

        @staticmethod
        def jaw_unit():
            return "m"

        @staticmethod
        def is_holding():
            return True

    gripper = IsaacGripper("pick-gripper")
    gripper._handle = Handle()
    gripper._sensorless = True
    gripper._active_grip = False

    status = asyncio.run(gripper.is_holding_something())

    assert status.is_holding_something is False
    assert status.meta["firm_grasp"] is True


def test_lite6_without_sensorless_logs_warning(manager, caplog):
    _arm("lite6")
    _gripper({"asset": "lite6_gripper"})
    assert "lite6_gripper is normally sensorless" in caplog.text


def test_empty_closure_does_not_report_holding(manager):
    _arm()
    gripper = _gripper()
    assert asyncio.run(gripper.grab()) is False


def test_grab_timeout_reports_contact_force_stats():
    class Handle:
        @staticmethod
        def close():
            pass

        @staticmethod
        def is_moving():
            return True

        @staticmethod
        def contact_force_summary():
            return {
                "contact_force_n": [0.31, 0.04],
                "max_contact_force_n": [0.72, 0.09],
            }

    gripper = IsaacGripper("pick-gripper")
    gripper._handle = Handle()
    gripper._timeout = 0.0

    with pytest.raises(TimeoutError) as exc:
        asyncio.run(gripper.grab())

    message = str(exc.value)
    assert "contact_force_n" in message
    assert "[0.31, 0.04]" in message
    assert "max_contact_force_n" in message
    assert "[0.72, 0.09]" in message


def test_invalid_normalized_inputs_are_rpc_errors(manager):
    _arm()
    gripper = _gripper()

    async def scenario():
        with pytest.raises(Exception) as length_error:
            await gripper.go_to_inputs([])
        assert length_error.value.grpc_code == Status.INVALID_ARGUMENT
        with pytest.raises(Exception) as range_error:
            await gripper.go_to_inputs([1.1])
        assert range_error.value.grpc_code == Status.INVALID_ARGUMENT

    asyncio.run(scenario())


def test_set_input_command_latches_without_waiting(manager, monkeypatch):
    _arm()
    gripper = _gripper()

    async def must_not_wait(*args, **kwargs):
        raise AssertionError("immediate command waited for settlement")

    monkeypatch.setattr(gripper, "_wait_stopped", must_not_wait)

    async def scenario():
        result = await gripper.do_command({"command": "set_input", "value": 0.4})
        assert result == {"accepted": True}
        assert await gripper.is_moving()
        await gripper.stop()
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            await gripper.do_command({"command": "set_input", "value": float("nan")})

    asyncio.run(scenario())


def test_isaac_gripper_requires_no_progress_before_stall():
    position = [0.0]

    class FakeArticulation:
        @staticmethod
        def get_joint_positions(*, joint_indices):
            assert joint_indices == [0]
            return position

    class FakeSim:
        mock = True
        _mock_sim_time_s = 0.0

    handle = object.__new__(IsaacGripperHandle)
    handle._art = FakeArticulation()
    handle._sim = FakeSim()
    handle._index = 0
    handle._drive_indices = [0]
    handle._measure_indices = [0]
    handle._target = 1.0
    handle._open = 0.0
    handle._closed = 1.0
    handle._holding_tolerance = 0.01
    handle._progress_epsilon = 0.001
    handle._best_gap = 1.0
    handle._minimum_grasp_closure = 0.0
    handle._has_progressed = False
    handle._last_progress_sim_time = 0.0
    handle._held_latch = False

    handle._sim._mock_sim_time_s = 0.30
    _gap, stalled = handle._gap_and_stall()
    assert stalled is False

    position[0] = 0.2
    _gap, stalled = handle._gap_and_stall()
    assert stalled is False
    assert handle._has_progressed is True

    handle._sim._mock_sim_time_s = 0.54
    _gap, stalled = handle._gap_and_stall()
    assert stalled is False
    handle._sim._mock_sim_time_s = 0.56
    _gap, stalled = handle._gap_and_stall()
    assert stalled is True


def test_isaac_gripper_closes_with_continuous_effort():
    position = [0.6]
    force_matrix = np.array([[[0.3, 0.0, 0.0]], [[0.0, 0.0, 0.0]]])
    actions = []
    gains = [np.array([600.0]), np.array([7.0])]

    class FakeController:
        @staticmethod
        def get_gains():
            return tuple(value.copy() for value in gains)

        @staticmethod
        def set_gains(*, kps, kds):
            gains[:] = [kps, kds]

    class FakeArticulation:
        @staticmethod
        def get_joint_positions(*, joint_indices):
            return position

        @staticmethod
        def get_articulation_controller():
            return FakeController()

        @staticmethod
        def apply_action(action):
            actions.append(action)

    class FakeContactView:
        @staticmethod
        def get_contact_force_matrix(*, dt):
            assert dt == pytest.approx(0.02)
            return force_matrix

    class FakeSim:
        mock = True
        _mock_sim_time_s = 0.0
        _isaac = SimpleNamespace(
            ArticulationAction=lambda **kwargs: SimpleNamespace(**kwargs)
        )

        @staticmethod
        def run(callback):
            return callback()

    handle = object.__new__(IsaacGripperHandle)
    handle._art = FakeArticulation()
    handle._sim = FakeSim()
    handle._drive_indices = [0]
    handle._last_contact_forces = (0.0, 0.0)
    handle._max_contact_forces = (0.0, 0.0)
    handle._measure_indices = [0]
    handle._target = 1.0
    handle._open = 0.0
    handle._closed = 1.0
    handle._holding_tolerance = 0.01
    handle._minimum_grasp_closure = 0.05
    handle._drive_stiffness = 600.0
    handle._drive_damping = 7.0
    handle._contact_view = FakeContactView()
    handle._contact_dt = 0.02
    handle._best_gap = 1.0
    handle._progress_epsilon = 0.001
    handle._has_progressed = True
    handle._last_progress_sim_time = 0.0
    handle._held_latch = False
    handle._effort_closing = False

    handle.close()
    assert handle._effort_closing is True
    assert handle._held_latch is False
    assert gains[0] == pytest.approx([0.0])
    assert gains[1] == pytest.approx([0.0])
    assert not hasattr(actions[-1], "joint_positions")
    assert actions[-1].joint_efforts == pytest.approx([2.0])

    summary = handle.contact_force_summary()
    assert summary == {
        "contact_force_n": pytest.approx([0.3, 0.0]),
        "max_contact_force_n": pytest.approx([0.3, 0.0]),
    }
    handle._on_physics_step(0.02)
    assert handle._held_latch is False

    force_matrix[1, 0, 0] = 0.3
    handle._sim._mock_sim_time_s = 0.01
    handle._on_physics_step(0.02)
    assert handle._held_latch is True
    assert handle.is_holding() is True
    assert actions[-1].joint_efforts == pytest.approx([2.0])
    assert handle.contact_force_summary() == {
        "contact_force_n": pytest.approx([0.3, 0.3]),
        "max_contact_force_n": pytest.approx([0.3, 0.3]),
    }

    handle.set_jaw(0.0)
    assert handle._effort_closing is False
    assert gains[0] == pytest.approx([600.0])
    assert gains[1] == pytest.approx([7.0])
    assert actions[-1].joint_positions == pytest.approx([0.0])
    assert actions[-1].joint_efforts == pytest.approx([0.0])


@pytest.mark.parametrize("jaw_layout", tuple(sim_manager.LITE6_JAW_LAYOUTS))
def test_lite6_jaw_layout_preserves_tcp_and_planning_envelope(manager, jaw_layout):
    _arm("lite6")
    gripper = _gripper({"asset": "lite6_gripper", "jaw_layout": jaw_layout})

    async def scenario():
        _format, data = await gripper.get_kinematics()
        geometry = json.loads(data)["links"][0]["geometry"]
        assert (geometry["x"], geometry["y"], geometry["z"]) == (110.0, 80.0, 84.0)
        assert geometry["translation"]["z"] == pytest.approx(-41.6)
        reported = (await gripper.get_geometries())[0]
        assert reported.center.z == pytest.approx(-41.6)
        assert (
            reported.box.dims_mm.x,
            reported.box.dims_mm.y,
            reported.box.dims_mm.z,
        ) == (110.0, 80.0, 84.0)

    asyncio.run(scenario())


def test_kinematics_and_geometry_describe_tcp_box(manager):
    _arm()
    gripper = _gripper()

    async def scenario():
        _format, data = await gripper.get_kinematics()
        sva = json.loads(data)
        assert sva["joints"] == []
        geometry = sva["links"][0]["geometry"]
        assert (geometry["x"], geometry["y"], geometry["z"]) == (36.0, 146.0, 153.0)
        geometries = await gripper.get_geometries()
        assert len(geometries) == 1
        assert geometries[0].center.z == pytest.approx(-57.5)

    asyncio.run(scenario())
