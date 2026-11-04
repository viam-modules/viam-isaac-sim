"""Drive the Isaac-only code paths of the floating-body support (RigidPrim
views on props, the water surface) against stand-ins for pxr and the isaac
namespace, so a typo there fails here instead of on a GPU machine."""

import sys
import types
from typing import ClassVar
from types import SimpleNamespace

import numpy as np
import pytest

from isaac_module.sim_manager import IsaacBodyHandle, SimManager


class _Attr:
    def __init__(self):
        self.value = None

    def Set(self, value):
        self.value = value


class _Prim:
    def __init__(self, path):
        self.path = path
        self.apis = []
        self.stage = SimpleNamespace()

    def IsValid(self):
        return True

    def GetStage(self):
        return self.stage


class _Shape:
    def __init__(self, path):
        self.prim = _Prim(path)
        self.size = _Attr()
        self.color = _Attr()

    def GetPrim(self):
        return self.prim

    def GetSizeAttr(self):
        return self.size

    def GetDisplayColorAttr(self):
        return self.color


class _Xformable:
    def __init__(self, prim):
        self.prim = prim
        prim.ops = getattr(prim, "ops", [])

    def AddTranslateOp(self):
        attr = _Attr()
        self.prim.ops.append(("translate", attr))
        return attr

    def AddScaleOp(self):
        attr = _Attr()
        self.prim.ops.append(("scale", attr))
        return attr


class _Api:
    name = "api"

    @classmethod
    def Apply(cls, prim):
        prim.apis.append(cls.__name__)
        return cls()

    def CreateMassAttr(self, value):
        self.mass = value
        return _Attr()


class _Shader:
    def __init__(self):
        self.inputs = {}

    def CreateIdAttr(self, value):
        return _Attr()

    def CreateInput(self, name, kind):
        attr = _Attr()
        self.inputs[name] = attr
        return attr

    def ConnectableAPI(self):
        return self


class _Material:
    def CreateSurfaceOutput(self):
        return SimpleNamespace(ConnectToSource=lambda *args: None)


@pytest.fixture
def fake_pxr(monkeypatch):
    defined = {}

    def definer(factory):
        def define(stage, path):
            obj = factory(str(path))
            defined[str(path)] = obj
            return obj

        return SimpleNamespace(Define=define)

    shader = _Shader()
    pxr = types.ModuleType("pxr")
    pxr.Gf = SimpleNamespace(
        Vec3d=lambda *v: ("Vec3d",) + v, Vec3f=lambda *v: ("Vec3f",) + v
    )
    pxr.Sdf = SimpleNamespace(
        Path=lambda p: p,
        ValueTypeNames=SimpleNamespace(Color3f="Color3f", Float="Float"),
    )
    pxr.UsdGeom = SimpleNamespace(
        Xform=definer(_Shape),
        Cube=definer(_Shape),
        Xformable=_Xformable,
    )
    pxr.UsdPhysics = SimpleNamespace(
        CollisionAPI=type("CollisionAPI", (_Api,), {}),
        RigidBodyAPI=type("RigidBodyAPI", (_Api,), {}),
        MassAPI=type("MassAPI", (_Api,), {}),
    )
    pxr.UsdShade = SimpleNamespace(
        Material=SimpleNamespace(Define=lambda stage, path: _Material()),
        Shader=SimpleNamespace(Define=lambda stage, path: shader),
        MaterialBindingAPI=SimpleNamespace(
            Apply=lambda prim: SimpleNamespace(Bind=lambda *a, **k: None)
        ),
        Tokens=SimpleNamespace(strongerThanDescendants="strong"),
    )
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    return SimpleNamespace(defined=defined, shader=shader)


class FakeRigidPrim:
    instances: ClassVar[list] = []

    def __init__(self, prim_paths_expr, name):
        self.prim_paths_expr = prim_paths_expr
        self.name = name
        self.initialized = False
        FakeRigidPrim.instances.append(self)

    def initialize(self):
        self.initialized = True

    def get_world_poses(self):
        return np.array([[1.0, 2.0, 0.1]]), np.array([[1.0, 0.0, 0.0, 0.0]])


class FakeCuboid:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def set_rest_offset(self, value):
        pass

    def set_contact_offset(self, value):
        pass


def _manager(fake_pxr):
    from isaac_module.sim_manager import SimConfig

    manager = object.__new__(SimManager)
    added = []
    callbacks = {}
    manager.cfg = SimConfig(water={"level_m": 0.0})
    manager.mock = False
    manager._bodies = {}
    manager._prop_specs = {}
    manager.world = SimpleNamespace(
        stage=SimpleNamespace(),
        scene=SimpleNamespace(add=added.append),
        reset=lambda: added.append("reset"),
        add_physics_callback=lambda name, fn: callbacks.__setitem__(name, fn),
        current_time=0.0,
    )
    manager._isaac = SimpleNamespace(
        RigidPrim=FakeRigidPrim,
        DynamicCuboid=FakeCuboid,
        FixedCuboid=lambda **kwargs: ("FixedCuboid", kwargs),
        PhysicsMaterial=lambda **kwargs: object(),
        get_prim_at_path=lambda path: _Prim(path),
    )
    return manager, added, callbacks


def test_floating_cube_prop_gets_a_rigid_body_and_a_physics_callback(fake_pxr):
    manager, added, callbacks = _manager(fake_pxr)
    FakeRigidPrim.instances.clear()

    manager._spawn_prop(
        {
            "name": "boat",
            "type": "cube",
            "position": [1.0, 2.0, 0.1],
            "size": 1.0,
            "scale": [1.0, 0.6, 0.3],
            "mass": 20,
            "buoyancy": True,
        }
    )

    cuboid = added[0]
    assert isinstance(cuboid, FakeCuboid)
    assert cuboid.kwargs["mass"] == 20.0
    view = FakeRigidPrim.instances[-1]
    assert view.prim_paths_expr == "/World/boat"
    assert added[1] is view
    # spawned before the initial world.reset, which initializes the view
    assert view.initialized is False
    assert "World_boat_body" in callbacks
    body = manager._bodies["boat"]
    assert manager._bodies["/World/boat"] is body
    assert isinstance(body, IsaacBodyHandle)
    assert body.spec is not None and body.spec.mass_kg == 20.0
    assert body.spawn_position == pytest.approx((1.0, 2.0, 0.1))


def test_body_for_prim_path_initializes_a_late_view(fake_pxr):
    manager, added, callbacks = _manager(fake_pxr)
    manager._booted = SimpleNamespace(is_set=lambda: True)
    manager._boot_error = None
    manager.run = lambda fn, timeout=30.0: fn()
    FakeRigidPrim.instances.clear()

    body = manager.body({"prim_path": "/World/Robot/base_link"}, "motor m")

    view = FakeRigidPrim.instances[-1]
    assert view.initialized is True
    assert body.spec is None
    assert manager.body({"prim_path": "/World/Robot/base_link"}, "motor m") is body
    assert "World_Robot_base_link_body" in callbacks


def test_spawn_water_adds_a_translucent_surface_and_a_seabed(fake_pxr):
    manager, added, _ = _manager(fake_pxr)

    manager._spawn_water({"level_m": 0.5, "size_m": 100, "depth_m": 4})

    surface = fake_pxr.defined["/World/Water/Surface"]
    ops = dict(surface.GetPrim().ops)
    assert ops["translate"].value == ("Vec3d", 0.0, 0.0, 0.49)
    assert ops["scale"].value == ("Vec3f", 100.0, 100.0, 0.02)
    assert fake_pxr.shader.inputs["opacity"].value == 0.8
    kind, kwargs = added[0]
    assert kind == "FixedCuboid"
    assert kwargs["position"][2] == pytest.approx(0.5 - 4 - 0.5)
    assert np.array_equal(kwargs["scale"], np.array([100.0, 100.0, 1.0]))
    with pytest.raises(ValueError, match="positive"):
        manager._spawn_water({"size_m": -1})


def test_floating_usd_prop_gets_its_mass_and_needs_a_rigid_body(fake_pxr, monkeypatch):
    import sys

    manager, added, callbacks = _manager(fake_pxr)
    FakeRigidPrim.instances.clear()
    applied = {}

    class RigidBodyAPI:
        pass

    class MassAPI:
        @staticmethod
        def Apply(prim):
            class _Api:
                def CreateMassAttr(self, value):
                    applied["mass"] = value

            return _Api()

    sys.modules["pxr"].UsdPhysics = SimpleNamespace(RigidBodyAPI=RigidBodyAPI, MassAPI=MassAPI)

    class RigidPrimPrim(_Prim):
        def HasAPI(self, api):
            return api is RigidBodyAPI

    class PlainPrim(_Prim):
        def HasAPI(self, api):
            return False

    poses = []
    manager._isaac.add_reference_to_stage = lambda usd_path, prim_path: None
    manager._isaac.SingleXFormPrim = lambda path: SimpleNamespace(
        set_world_pose=lambda **kw: poses.append(kw)
    )
    manager._usd_exists = lambda path: True
    manager._isaac.get_prim_at_path = lambda path: RigidPrimPrim(path)

    manager._spawn_prop(
        {
            "name": "skiff",
            "type": "usd",
            "usd_path": "/Isaac/skiff.usd",
            "position": [0, 0, 0.1],
            "mass": 12,
            "buoyancy": {"size_m": [2.0, 0.8, 0.4]},
        }
    )
    assert applied["mass"] == 12.0
    assert FakeRigidPrim.instances[-1].prim_paths_expr == "/World/skiff"
    assert manager._bodies["skiff"].spec.mass_kg == 12.0

    manager._isaac.get_prim_at_path = lambda path: PlainPrim(path)
    with pytest.raises(ValueError, match="RigidBodyAPI"):
        manager._spawn_prop(
            {
                "name": "rock",
                "type": "usd",
                "usd_path": "/Isaac/rock.usd",
                "mass": 3,
                "buoyancy": {"size_m": [1, 1, 1]},
            }
        )


def test_two_prim_path_bodies_with_the_same_leaf_name_do_not_collide(fake_pxr):
    manager, added, callbacks = _manager(fake_pxr)
    manager._booted = SimpleNamespace(is_set=lambda: True)
    manager._boot_error = None
    manager.run = lambda fn, timeout=30.0: fn()
    FakeRigidPrim.instances.clear()

    a = manager.body({"prim_path": "/World/A/hull"}, "motor a")
    b = manager.body({"prim_path": "/World/B/hull"}, "motor b")

    assert a is not b
    assert {v.name for v in FakeRigidPrim.instances[-2:]} == {"World_A_hull_body", "World_B_hull_body"}
    assert {"World_A_hull_body", "World_B_hull_body"} <= set(callbacks)
