import math
from xml.etree import ElementTree

import numpy as np
import pytest

from isaac_module import stage_geometry


def _move(x=0.0, y=0.0, z=0.0, rot_x_deg=0.0):
    m = np.eye(4)
    c, s = math.cos(math.radians(rot_x_deg)), math.sin(math.radians(rot_x_deg))
    m[1:3, 1:3] = [[c, -s], [s, c]]
    m[:3, 3] = (x, y, z)
    return m


def _box(lo, hi):
    corners = np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
        float,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]] * 6)  # topology doesn't matter for these tests
    return corners, faces


def _chain():
    return {
        "name": "left-arm",
        "base": "torso",
        "steps": [
            {
                "joint": "j1",
                "type": "revolute",
                "axis": (0.0, 0.0, 1.0),
                "lower": -math.pi / 2,
                "upper": math.pi / 2,
                # joint frame 0.1 above the torso, rotated 90 about x
                "parent_frame": _move(z=0.1, rot_x_deg=90),
                # and 0.05 above the child body's origin
                "child_frame": _move(z=0.05),
                "link": "l1",
                "collision": _box((-0.01, -0.02, 0.0), (0.01, 0.02, 0.2)),
                "visual": _box((-0.01, -0.02, 0.0), (0.01, 0.02, 0.2)),
            },
            {
                "joint": "mount",
                "type": "fixed",
                "axis": (1.0, 0.0, 0.0),
                "lower": 0.0,
                "upper": 0.0,
                "parent_frame": _move(z=0.2),
                "child_frame": np.eye(4),
                "link": "hand",
                "collision": None,
                "visual": None,
            },
        ],
        "end": {"name": "hand_mount", "frame": _move(x=0.03)},
    }


def _floats(text):
    return [float(v) for v in text.split()]


def test_urdf_joint_origins_are_the_usd_joint_frames_in_the_parent_link():
    urdf, meshes = stage_geometry.build_urdf(_chain(), meshes=False)
    root = ElementTree.fromstring(urdf)
    assert [link.get("name") for link in root.iterfind("link")] == [
        "torso", "l1", "hand", "hand_mount_end",
    ]
    joints = {j.get("name"): j for j in root.iterfind("joint")}
    j1 = joints["j1"]
    assert j1.get("type") == "revolute"
    assert (j1.find("parent").get("link"), j1.find("child").get("link")) == ("torso", "l1")
    assert _floats(j1.find("origin").get("xyz")) == pytest.approx([0, 0, 0.1])
    assert _floats(j1.find("origin").get("rpy")) == pytest.approx([math.pi / 2, 0, 0])
    assert _floats(j1.find("axis").get("xyz")) == pytest.approx([0, 0, 1])
    limit = j1.find("limit")
    assert float(limit.get("lower")) == pytest.approx(-math.pi / 2)
    assert float(limit.get("upper")) == pytest.approx(math.pi / 2)
    # l1's URDF frame is its joint frame, 0.05 above its body: mount sits 0.15 above it
    assert _floats(joints["mount"].find("origin").get("xyz")) == pytest.approx([0, 0, 0.15])
    assert joints["mount"].get("type") == "fixed"
    end = joints["hand_mount_end_joint"]
    assert end.get("type") == "fixed"
    assert _floats(end.find("origin").get("xyz")) == pytest.approx([0.03, 0, 0])
    assert meshes == {}


def test_urdf_link_collision_is_a_box_fitted_in_the_link_frame():
    urdf, _ = stage_geometry.build_urdf(_chain(), meshes=False)
    root = ElementTree.fromstring(urdf)
    links = {link.get("name"): link for link in root.iterfind("link")}
    assert links["torso"].find("collision") is None  # the base belongs to the parent limb
    assert links["hand"].find("collision") is None
    collision = links["l1"].find("collision")
    assert _floats(collision.find("origin").get("xyz")) == pytest.approx([0, 0, 0.05])
    assert _floats(collision.find("geometry/box").get("size")) == pytest.approx(
        [0.02, 0.04, 0.2]
    )


def test_urdf_mesh_collisions_are_served_as_ply_in_the_link_frame():
    urdf, meshes = stage_geometry.build_urdf(_chain(), meshes=True)
    mesh = ElementTree.fromstring(urdf).find("link[@name='l1']/collision/geometry/mesh")
    assert mesh.get("filename") == "l1.ply"
    kind, data = meshes["l1.ply"]
    assert kind == "ply"
    text = data.decode()
    assert "element vertex 8" in text and "element face 12" in text
    zs = [float(line.split()[2]) for line in text.splitlines()[9:17]]
    assert min(zs) == pytest.approx(-0.05) and max(zs) == pytest.approx(0.15)


def test_visual_models_are_glbs_in_each_collision_frame():
    trimesh = pytest.importorskip("trimesh")
    import io

    for meshes, z in ((True, (-0.05, 0.15)), (False, (-0.1, 0.1))):
        models = stage_geometry.visual_glbs(_chain(), meshes=meshes)
        assert list(models) == ["l1"]
        scene = trimesh.load(io.BytesIO(models["l1"]), file_type="glb")
        lo, hi = scene.bounds
        assert (lo[2], hi[2]) == pytest.approx(z)


def test_hull_keeps_only_the_outside_of_a_piece():
    corners, _ = _box((0, 0, 0), (1, 1, 1))
    inside = np.array([[0.5, 0.5, 0.5], [0.2, 0.3, 0.4]])
    vertices, faces = stage_geometry.hull(np.vstack([inside, corners]))
    assert len(vertices) == 8 and len(faces) == 12
    assert faces.max() == 7


def test_hull_of_a_flat_piece_is_the_piece():
    flat = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    vertices, faces = stage_geometry.hull(flat)
    assert len(vertices) == 3 and faces.tolist() == [[0, 1, 2]]


def test_urdf_names_are_unique_across_links_and_joints():
    # RDK keeps links and joints in one namespace (Mabel's torso joint moves its torso link)
    chain = _chain()
    chain["steps"][0]["link"] = "j1"
    urdf, meshes = stage_geometry.build_urdf(chain, meshes=True)
    root = ElementTree.fromstring(urdf)
    names = [e.get("name") for e in (*root.iterfind("link"), *root.iterfind("joint"))]
    assert len(names) == len(set(names))
    assert root.find("joint[@name='j1']/child").get("link") == "j1_link"
    assert root.find("joint[@name='mount']/parent").get("link") == "j1_link"
    assert "j1_link.ply" in meshes


def test_hull_of_coplanar_points_joggles_into_a_hull():
    square = np.array([[0.5, 0.5, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    vertices, faces = stage_geometry.hull(square)  # the centre point is not on its outside
    assert len(vertices) == 4 and len(faces) >= 2


# -- a real (in-memory) USD stage -----------------------------------------------


def _stage(meters_per_unit=1.0):
    """/World/r: base -j1 (revolute z, -90..45 deg)-> l1 -mount (fixed)-> l2,
    ee an Xform under l2, and a finger link hanging off l2 by f1."""
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    UsdGeom.Xform.Define(stage, "/World/r")
    for name in ("base", "l1", "l2", "finger"):
        UsdGeom.Xform.Define(stage, f"/World/r/{name}")
    for name, size in (("l1", 0.1), ("finger", 0.02)):
        cube = UsdGeom.Cube.Define(stage, f"/World/r/{name}/collision")
        cube.CreateSizeAttr(size)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdGeom.Xform(stage.GetPrimAtPath("/World/r/finger")).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.3))
    UsdGeom.Xform.Define(stage, "/World/r/l2/ee").AddTranslateOp().Set(Gf.Vec3d(0.03, 0, 0))

    def joint(kind, name, body0, body1, pos0):
        j = kind.Define(stage, f"/World/r/joints/{name}")
        j.CreateBody0Rel().SetTargets([Sdf.Path(f"/World/r/{body0}")])
        j.CreateBody1Rel().SetTargets([Sdf.Path(f"/World/r/{body1}")])
        j.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
        j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        return j

    j1 = joint(UsdPhysics.RevoluteJoint, "j1", "base", "l1", (0, 0, 0.1))
    j1.CreateAxisAttr("Z")
    j1.CreateLowerLimitAttr(-90.0)
    j1.CreateUpperLimitAttr(45.0)
    joint(UsdPhysics.FixedJoint, "mount", "l1", "l2", (0, 0, 0.2))
    f1 = joint(UsdPhysics.RevoluteJoint, "f1", "l2", "finger", (0, 0, 0.3))
    f1.CreateAxisAttr("X")
    return stage


def test_stage_chain_walks_fixed_joints_to_the_end_effector():
    stage = _stage()
    chain = stage_geometry.stage_chain(
        stage.GetPrimAtPath("/World/r"), "arm", ["j1"], "/World/r/base", "/World/r/l2/ee", visual=False
    )
    root = ElementTree.fromstring(stage_geometry.build_urdf(chain, meshes=False)[0])
    assert [link.get("name") for link in root.iterfind("link")] == ["base", "l1", "l2", "ee_end"]
    joints = {j.get("name"): j for j in root.iterfind("joint")}
    assert joints["mount"].get("type") == "fixed"
    assert _floats(joints["mount"].find("origin").get("xyz")) == pytest.approx([0, 0, 0.2])
    assert _floats(joints["ee_end_joint"].find("origin").get("xyz")) == pytest.approx([0.03, 0, 0])
    limit = joints["j1"].find("limit")  # authored in degrees
    assert float(limit.get("lower")) == pytest.approx(-math.pi / 2)
    assert float(limit.get("upper")) == pytest.approx(math.pi / 4)
    box = root.find("link[@name='l1']/collision/geometry/box")
    assert _floats(box.get("size")) == pytest.approx([0.1, 0.1, 0.1])


def test_stage_chain_reads_stage_units_as_meters():
    stage = _stage(meters_per_unit=0.01)  # the same numbers, now centimeters
    chain = stage_geometry.stage_chain(
        stage.GetPrimAtPath("/World/r"), "arm", ["j1"], "/World/r/base", "/World/r/l2/ee", visual=False
    )
    root = ElementTree.fromstring(stage_geometry.build_urdf(chain, meshes=False)[0])
    assert _floats(root.find("joint[@name='j1']/origin").get("xyz")) == pytest.approx([0, 0, 0.001])
    assert _floats(root.find("link[@name='l1']/collision/geometry/box").get("size")) == pytest.approx(
        [0.001] * 3
    )


def test_stage_chain_one_sided_limit_keeps_its_finite_side():
    stage = _stage()
    from pxr import UsdPhysics

    UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/World/r/joints/j1")).GetLowerLimitAttr().Set(float("-inf"))
    chain = stage_geometry.stage_chain(
        stage.GetPrimAtPath("/World/r"), "arm", ["j1"], "/World/r/base", "/World/r/l2/ee", visual=False
    )
    assert (chain["steps"][0]["lower"], chain["steps"][0]["upper"]) == pytest.approx(
        (-2 * math.pi, math.pi / 4)
    )


def test_stage_chain_rejects_joint_names_out_of_chain_order():
    stage = _stage()
    with pytest.raises(ValueError, match="joint_names"):
        stage_geometry.stage_chain(
            stage.GetPrimAtPath("/World/r"), "arm", ["j1", "f1"], "/World/r/base",
            "/World/r/l2/ee", visual=False,
        )


def test_hand_links_are_the_links_below_the_end_effector_link():
    stage = _stage()
    assert stage_geometry.hand_links(stage.GetPrimAtPath("/World/r"), "/World/r/l2/ee") == [
        "/World/r/finger"
    ]


@pytest.mark.parametrize("meters", [1.0, 0.01])
def test_joint_name_gripper_pieces_follow_the_finger_link(meters):
    stage = _stage(meters)
    from types import SimpleNamespace

    from pxr import Gf, UsdGeom

    from isaac_module.sim_manager import IsaacGripperHandle

    class Xform:  # Isaac's SingleXFormPrim, read from the USD stage
        def __init__(self, path):
            self.prim_path = path

        def get_world_pose(self):
            m = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(self.prim_path))
            q = m.ExtractRotationQuat()
            return list(m.ExtractTranslation()), [q.GetReal(), *q.GetImaginary()]

    handle = object.__new__(IsaacGripperHandle)
    handle._sim = SimpleNamespace(
        run=lambda fn: fn(),
        _isaac=SimpleNamespace(get_prim_at_path=stage.GetPrimAtPath, SingleXFormPrim=Xform),
    )
    handle._stage_arm = SimpleNamespace(_prim_path="/World/r", _ee=Xform("/World/r/l2/ee"))
    (name, before, _), = handle.collision_pieces((0, 0, 0), (1, 0, 0, 0))
    assert name == "finger"
    # in the gripper frame (the ee, 0.03 along x): the finger's 2 cm cube at z 0.3
    assert before.mean(axis=0) == pytest.approx(np.array([-0.03, 0, 0.3]) * meters)
    finger = UsdGeom.Xformable(stage.GetPrimAtPath("/World/r/finger"))
    finger.GetOrderedXformOps()[0].Set(Gf.Vec3d(0, 0.05, 0.3))
    (_, after, _), = handle.collision_pieces((0, 0, 0), (1, 0, 0, 0))
    assert after.mean(axis=0) == pytest.approx(np.array([-0.03, 0.05, 0.3]) * meters)
    # the gripper's Viam frame offset is meters, whatever the stage's unit
    (_, offset, _), = handle.collision_pieces((0, 0.01, 0), (1, 0, 0, 0))
    assert offset.mean(axis=0) == pytest.approx(np.array([-0.03, 0.05, 0.3]) * meters - [0, 0.01, 0])


def test_visual_models_are_capped_per_link(monkeypatch):
    trimesh = pytest.importorskip("trimesh")
    import io

    sphere = trimesh.creation.icosphere(subdivisions=4)  # 5120 faces
    chain = _chain()
    chain["steps"][0]["visual"] = (np.asarray(sphere.vertices), np.asarray(sphere.faces))
    monkeypatch.setattr(stage_geometry, "VISUAL_FACE_CAP", 500)
    glb = stage_geometry.visual_glbs(chain, meshes=True)["l1"]
    mesh = trimesh.load(io.BytesIO(glb), file_type="glb", force="mesh")
    assert 0 < len(mesh.faces) <= 500


def _torus(major=1.0, minor=0.3, n=200, m=100):
    """A torus as a triangle soup: every triangle has its own 3 vertices, as an STL does."""
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, n, endpoint=False), np.linspace(0, 2 * np.pi, m, endpoint=False))
    points = np.stack(
        [(major + minor * np.cos(v)) * np.cos(u), (major + minor * np.cos(v)) * np.sin(u), minor * np.sin(v)], -1
    ).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(n), np.arange(m))
    a, b, c, d = j * n + i, j * n + (i + 1) % n, ((j + 1) % m) * n + i, ((j + 1) % m) * n + (i + 1) % n
    faces = np.concatenate([np.stack([a, b, d], -1).reshape(-1, 3), np.stack([a, d, c], -1).reshape(-1, 3)])
    return points[faces].reshape(-1, 3), np.arange(len(faces) * 3).reshape(-1, 3)


def _sphere(radius=1.0, n=200, m=100):
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, n, endpoint=False), np.linspace(0, np.pi, m + 1))
    points = radius * np.stack([np.sin(v) * np.cos(u), np.sin(v) * np.sin(u), np.cos(v)], -1).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(n), np.arange(m))
    a, b, c, d = j * n + i, j * n + (i + 1) % n, (j + 1) * n + i, (j + 1) * n + (i + 1) % n
    faces = np.concatenate([np.stack([a, b, d], -1).reshape(-1, 3), np.stack([a, d, c], -1).reshape(-1, 3)])
    return points, faces


def _clean(faces):
    assert (faces[:, 0] != faces[:, 1]).all() and (faces[:, 1] != faces[:, 2]).all()
    assert (faces[:, 0] != faces[:, 2]).all()
    assert len(np.unique(np.sort(faces, axis=1), axis=0)) == len(faces)


def test_simplified_clusters_a_dense_sphere_to_just_under_the_cap():
    vertices, faces = stage_geometry.simplified(*_sphere(), cap=5000)
    assert 3500 <= len(faces) <= 5000
    _clean(faces)
    assert np.abs(np.linalg.norm(vertices, axis=1) - 1).max() < 0.02


def test_simplified_keeps_a_torus_hole_where_a_hull_would_fill_it():
    vertices, faces = stage_geometry.simplified(*_torus(), cap=4000)
    assert 2800 <= len(faces) <= 4000
    _clean(faces)
    ring = np.hypot(vertices[:, 0], vertices[:, 1])
    tube = np.hypot(ring - 1.0, vertices[:, 2])
    assert np.abs(tube - 0.3).max() < 0.02
    assert ring.min() < 0.75  # the inner wall survives: a convex hull has none inside the ring


def test_simplified_welds_a_soup_and_drops_degenerate_and_duplicate_faces():
    vertices, faces = _torus(n=20, m=10)
    faces = np.vstack([faces, [[0, 0, 1]], faces[:1, ::-1]])
    welded, kept = stage_geometry.simplified(vertices, faces, cap=10_000)
    assert len(welded) == 200  # each grid point once: shared normals shade smooth
    assert len(kept) == 400
    _clean(kept)


def test_simplified_needs_no_trimesh(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "trimesh", None)
    _, faces = stage_geometry.simplified(*_sphere(), cap=2000)
    assert 0 < len(faces) <= 2000


def test_visual_models_without_quadric_decimation_are_clustered_not_hulled(monkeypatch):
    trimesh = pytest.importorskip("trimesh")
    import io

    def unavailable(*_, **__):
        raise ImportError("fast_simplification")

    monkeypatch.setattr(trimesh.Trimesh, "simplify_quadric_decimation", unavailable)
    monkeypatch.setattr(stage_geometry, "VISUAL_FACE_CAP", 4000)
    chain = _chain()
    vertices, faces = _torus(major=0.1, minor=0.03)
    chain["steps"][0]["visual"] = (vertices, faces)
    glb = stage_geometry.visual_glbs(chain, meshes=True)["l1"]
    assert b'"NORMAL"' in glb  # smooth shaded
    mesh = trimesh.load(io.BytesIO(glb), file_type="glb", force="mesh")
    assert 2800 <= len(mesh.faces) <= 4000
    centre = mesh.vertices.mean(axis=0)
    ring = np.hypot(*(mesh.vertices - centre)[:, :2].T)
    assert ring.min() < 0.075
