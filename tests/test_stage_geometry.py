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
