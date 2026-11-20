"""Collision geometry and kinematics read from the live Isaac stage.

An arm's chain (its joints from its base link to its end effector) becomes a
URDF for Viam's motion planner: each URDF joint's origin is the USD joint's
frame on its parent link, each URDF link's frame is the joint's frame on that
link, and its <collision> is the link's collision prims (a fitted box, or
their mesh). Matrices are 4x4 numpy, column vectors, meters.

The reading functions (stage_chain, hand_links, the *_pieces) run on the sim thread and
need pxr; the rest is plain numpy.
"""

from __future__ import annotations

import math
from typing import Any, Sequence
from xml.etree import ElementTree

import numpy as np

UNLIMITED_RAD = 2 * math.pi  # a revolute joint with no USD limits plans over -360..360 degrees


def ply(pieces: Sequence[tuple[np.ndarray, np.ndarray]]) -> bytes:
    """Triangle pieces (vertices m, faces) merged into one ASCII PLY, in meters
    (RDK reads PLY as meters)."""
    vertices, faces, offset = [], [], 0
    for piece_vertices, piece_faces in pieces:
        vertices.extend(np.asarray(piece_vertices).tolist())
        faces.extend((np.asarray(piece_faces) + offset).tolist())
        offset += len(piece_vertices)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
        *(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices),
        *(f"3 {a} {b} {c}" for a, b, c in faces),
    ]
    return ("\n".join(lines) + "\n").encode()


def moved(matrix: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    return np.asarray(vertices, float) @ matrix[:3, :3].T + matrix[:3, 3]


def _rpy(matrix: np.ndarray) -> tuple[float, float, float]:
    """URDF roll, pitch, yaw (fixed-axis xyz) of a rotation."""
    r = matrix[:3, :3]
    pitch = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    if abs(r[2, 0]) < 1.0 - 1e-9:
        return math.atan2(r[2, 1], r[2, 2]), pitch, math.atan2(r[1, 0], r[0, 0])
    return 0.0, pitch, math.atan2(-r[0, 1], r[1, 1])  # gimbal lock: fold roll into yaw


def _origin(parent: ElementTree.Element, matrix: np.ndarray) -> None:
    ElementTree.SubElement(
        parent,
        "origin",
        xyz=" ".join(f"{v:.9g}" for v in matrix[:3, 3]),
        rpy=" ".join(f"{v:.9g}" for v in _rpy(matrix)),
    )


def _renamed(chain: dict) -> dict:
    """The chain with its links renamed apart from its joints (RDK keeps both
    in one namespace) and its end as a link of its own."""
    end = f"{chain['end']['name']}_end"
    taken = {step["joint"] for step in chain["steps"]} | {f"{end}_joint"}

    def unique(name: str) -> str:
        result = name if name not in taken else f"{name}_link"
        while result in taken:
            result += "_"
        taken.add(result)
        return result

    return {
        **chain,
        "base": unique(chain["base"]),
        "steps": [{**step, "link": unique(step["link"])} for step in chain["steps"]],
        "end": {**chain["end"], "link": unique(end), "joint": f"{end}_joint"},
    }


def _links(chain: dict, meshes: bool):
    """(link name, collision frame in the URDF link frame, collision vertices
    and faces in that frame, visual piece in that frame) per chain link with
    collision prims: a fitted box's centre, or with meshes the link frame."""
    for step in chain["steps"]:
        if step["collision"] is None:
            continue
        to_link = np.linalg.inv(step["child_frame"])
        vertices, faces = step["collision"]
        vertices = moved(to_link, vertices)
        frame = np.eye(4)
        if not meshes:
            frame[:3, 3] = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        to_frame = np.linalg.inv(frame)
        visual = step["visual"]
        if visual is not None:
            visual = (moved(to_frame @ to_link, visual[0]), visual[1])
        yield step["link"], frame, (moved(to_frame, vertices), faces), visual


def build_urdf(chain: dict, meshes: bool) -> tuple[bytes, dict[str, tuple[str, bytes]]]:
    """The chain as URDF kinematics, and its collision meshes keyed by the
    filename the URDF gives them, as GetKinematics serves them."""
    chain = _renamed(chain)
    robot = ElementTree.Element("robot", name=chain["name"])
    collisions = {name: (frame, piece) for name, frame, piece, _ in _links(chain, meshes)}
    files: dict[str, tuple[str, bytes]] = {}

    def link(name: str) -> None:
        element = ElementTree.SubElement(robot, "link", name=name)
        if name not in collisions:
            return
        frame, (vertices, faces) = collisions[name]
        collision = ElementTree.SubElement(element, "collision")
        geometry = ElementTree.SubElement(collision, "geometry")
        if meshes:
            files[f"{name}.ply"] = ("ply", ply([(vertices, faces)]))
            ElementTree.SubElement(geometry, "mesh", filename=f"{name}.ply")
        else:
            _origin(collision, frame)
            size = vertices.max(axis=0) - vertices.min(axis=0)
            ElementTree.SubElement(geometry, "box", size=" ".join(f"{v:.9g}" for v in size))

    def joint(name: str, kind: str, parent: str, child: str, origin: np.ndarray) -> ElementTree.Element:
        element = ElementTree.SubElement(robot, "joint", name=name, type=kind)
        _origin(element, origin)
        ElementTree.SubElement(element, "parent", link=parent)
        ElementTree.SubElement(element, "child", link=child)
        return element

    link(chain["base"])
    parent, parent_frame = chain["base"], np.eye(4)
    for step in chain["steps"]:
        link(step["link"])
        element = joint(
            step["joint"], step["type"], parent, step["link"],
            np.linalg.inv(parent_frame) @ step["parent_frame"],
        )
        if step["type"] != "fixed":
            ElementTree.SubElement(element, "axis", xyz=" ".join(f"{v:g}" for v in step["axis"]))
            ElementTree.SubElement(
                element, "limit", lower=f"{step['lower']:.9g}", upper=f"{step['upper']:.9g}",
                effort="0", velocity="0",
            )
        parent, parent_frame = step["link"], step["child_frame"]
    link(chain["end"]["link"])
    joint(
        chain["end"]["joint"], "fixed", parent, chain["end"]["link"],
        np.linalg.inv(parent_frame) @ chain["end"]["frame"],
    )
    return ElementTree.tostring(robot), files


def visual_glbs(chain: dict, meshes: bool) -> dict[str, bytes]:
    """A GLB of each link's visual prims, keyed by link name, in the frame of
    the link's collision (where a viewer draws a link's 3D model); a link with
    no collision gets none. Untextured, grey."""
    import trimesh

    return {
        name: trimesh.Trimesh(*visual, process=False).export(file_type="glb")
        for name, _, _, visual in _links(_renamed(chain), meshes)
        if visual is not None
    }


# ----------------------------------------------------------------------------
# Reading the stage (sim thread, pxr)
# ----------------------------------------------------------------------------


def _matrix(gf_matrix: Any) -> np.ndarray:
    return np.array(gf_matrix, float).T  # Gf matrices are row-vector


def _pose(position: Any, rotation: Any) -> np.ndarray:
    """A 4x4 from a USD joint's localPos and localRot (Gf.Quatf)."""
    return matrix_from_pose(position, (rotation.GetReal(), *rotation.GetImaginary()))


def matrix_from_pose(position: Sequence[float], quat_wxyz: Sequence[float]) -> np.ndarray:
    from .spatial import quat_rotate

    q = tuple(float(v) for v in quat_wxyz)
    m = np.eye(4)
    m[:3, :3] = np.column_stack(
        [quat_rotate(q, axis) for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]  # type: ignore[arg-type]
    )
    m[:3, 3] = [float(v) for v in position]
    return m


def _triangles(prim: Any) -> tuple[np.ndarray, np.ndarray]:
    """A gprim's triangles in its own frame: a mesh's points fanned into
    triangles, anything else as its extent box."""
    from pxr import Usd, UsdGeom

    if prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
        points = np.array(mesh.GetPointsAttr().Get() or [], float).reshape(-1, 3)
        counts = mesh.GetFaceVertexCountsAttr().Get() or []
        indices = mesh.GetFaceVertexIndicesAttr().Get() or []
        faces, start = [], 0
        for count in counts:
            faces.extend(
                (indices[start], indices[start + k], indices[start + k + 1])
                for k in range(1, count - 1)
            )
            start += count
        return points, np.array(faces, int).reshape(-1, 3)
    # ponytail: a cylinder, sphere or capsule collider is served as its extent box
    extent = UsdGeom.Boundable.ComputeExtentFromPlugins(UsdGeom.Boundable(prim), Usd.TimeCode.Default())
    if extent is None:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    lo, hi = np.array(extent[0], float), np.array(extent[1], float)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    faces = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
             [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    return corners, np.array(faces, int)


def hull(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A point cloud's convex hull (vertices, faces); a flat or tiny one as
    the triangles of its points in order (it has no volume to hull)."""
    from scipy.spatial import ConvexHull, QhullError

    vertices = np.asarray(vertices, float)
    try:
        simplices = ConvexHull(vertices).simplices
    except (QhullError, ValueError):
        return vertices, np.array([[0, k, k + 1] for k in range(1, len(vertices) - 1)], int)
    used = np.unique(simplices)
    remap = np.full(len(vertices), -1)
    remap[used] = np.arange(len(used))
    return vertices[used], remap[simplices]


def _merged(pieces: list) -> tuple[np.ndarray, np.ndarray] | None:
    if not pieces:
        return None
    offset, faces = 0, []
    for vertices, piece_faces in pieces:
        faces.append(np.asarray(piece_faces) + offset)
        offset += len(vertices)
    return np.vstack([v for v, _ in pieces]), np.vstack(faces)


def _gprims(link: Any, collision: bool) -> list:
    """A link prim's collision (or visible, non-colliding) gprims' triangles,
    in the link's frame. Nested rigid bodies are left to their own link."""
    from pxr import Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.XformCache()
    pieces = []
    it = iter(Usd.PrimRange(link, Usd.TraverseInstanceProxies()))
    for prim in it:
        if prim != link and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            it.PruneChildren()
            continue
        if not prim.IsA(UsdGeom.Gprim) or prim.HasAPI(UsdPhysics.CollisionAPI) != collision:
            continue
        if not collision and UsdGeom.Imageable(prim).ComputePurpose() not in (
            UsdGeom.Tokens.default_, UsdGeom.Tokens.render
        ):
            continue
        vertices, faces = _triangles(prim)
        if len(faces):
            relative, _ = cache.ComputeRelativeTransform(prim, link)
            pieces.append((moved(_matrix(relative), vertices), faces))
    return pieces


def planning_pieces(link: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """A link's shape for the motion planner, in its frame: the convex hull of
    each collision prim (what PhysX collides with), or of each visible prim
    when it has none (a URDF link with only <visual>s); None for neither."""
    pieces = _gprims(link, collision=True) or _gprims(link, collision=False)
    return _merged([hull(vertices) for vertices, _ in pieces])


def visual_pieces(link: Any) -> tuple[np.ndarray, np.ndarray] | None:
    return _merged(_gprims(link, collision=False))


def _joints_by_child(root: Any) -> dict[str, Any]:
    from pxr import Usd, UsdPhysics

    joints = {}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        body0, body1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if body0 and body1:
            joints[str(body1[0])] = prim
    return joints


def stage_chain(
    root: Any, name: str, joint_names: Sequence[str], base_path: str, ee_path: str,
    visual: bool,
) -> dict:
    """The chain from base_path's link to the end effector prim, as
    build_urdf takes it: every joint (fixed ones too) on the way, in order.
    The chain's movable joints must be joint_names, in that order."""
    from pxr import UsdGeom, UsdPhysics

    stage = root.GetStage()
    by_child = _joints_by_child(root)
    ee = stage.GetPrimAtPath(ee_path)
    if not ee.IsValid():
        raise ValueError(f"{name}: no end_effector_prim at {ee_path}")
    owner = _owner_link(ee, set(by_child) | {base_path})
    if owner is None:
        raise ValueError(f"{name}: end_effector_prim {ee_path} is not on a link of {root.GetPath()}")
    end, _ = UsdGeom.XformCache().ComputeRelativeTransform(ee, owner)

    steps, path = [], str(owner.GetPath())
    while path != base_path:
        prim = by_child.get(path)
        if prim is None:
            raise ValueError(f"{name}: {ee_path} doesn't hang off the base link {base_path}")
        joint = UsdPhysics.Joint(prim)
        revolute, prismatic = prim.IsA(UsdPhysics.RevoluteJoint), prim.IsA(UsdPhysics.PrismaticJoint)
        step = {
            "joint": prim.GetName(),
            "type": "revolute" if revolute else "prismatic" if prismatic else "fixed",
            "axis": (1.0, 0.0, 0.0), "lower": 0.0, "upper": 0.0,
            "parent_frame": _pose(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()),
            "child_frame": _pose(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()),
            "link": stage.GetPrimAtPath(path).GetName(),
            "collision": planning_pieces(stage.GetPrimAtPath(path)),
            "visual": visual_pieces(stage.GetPrimAtPath(path)) if visual else None,
        }
        if revolute or prismatic:
            typed = UsdPhysics.RevoluteJoint(prim) if revolute else UsdPhysics.PrismaticJoint(prim)
            step["axis"] = tuple(1.0 if a == typed.GetAxisAttr().Get() else 0.0 for a in "XYZ")
            lower, upper = typed.GetLowerLimitAttr().Get(), typed.GetUpperLimitAttr().Get()
            if lower is None or upper is None or not (math.isfinite(lower) and math.isfinite(upper)):
                if prismatic:
                    raise ValueError(f"{name}: prismatic joint {prim.GetName()} has no limits")
                step["lower"], step["upper"] = -UNLIMITED_RAD, UNLIMITED_RAD
            else:  # USD revolute limits are degrees
                scale = math.pi / 180.0 if revolute else 1.0
                step["lower"], step["upper"] = lower * scale, upper * scale
        steps.append(step)
        path = str(joint.GetBody0Rel().GetTargets()[0])
    steps.reverse()
    movable = [s["joint"] for s in steps if s["type"] != "fixed"]
    if movable != list(joint_names):
        raise ValueError(
            f"{name}: the joints from {base_path} to {ee_path} are {movable}, not "
            f"joint_names {list(joint_names)}; derived kinematics need the chain's joints in order"
        )
    return {
        "name": name,
        "base": stage.GetPrimAtPath(base_path).GetName(),
        "steps": steps,
        "end": {"name": ee.GetName(), "frame": _matrix(end)},
    }


def _owner_link(prim: Any, links: set[str]) -> Any:
    """The prim, or its nearest ancestor, that is one of `links`; None if none is."""
    while prim.IsValid() and not prim.IsPseudoRoot():
        if str(prim.GetPath()) in links:
            return prim
        prim = prim.GetParent()
    return None


def hand_links(root: Any, ee_path: str) -> list[str]:
    """Paths of the links hanging below the end effector's link (a hand:
    palm, fingers), through any joints; empty when ee_path isn't on a link."""
    from pxr import UsdPhysics

    joints = _joints_by_child(root)
    below: dict[str, list[str]] = {}
    for child, prim in joints.items():
        parent = str(UsdPhysics.Joint(prim).GetBody0Rel().GetTargets()[0])
        below.setdefault(parent, []).append(child)
    owner = _owner_link(root.GetStage().GetPrimAtPath(ee_path), set(joints) | set(below))
    if owner is None:
        return []
    paths, queue = [], list(below.get(str(owner.GetPath()), []))
    while queue:
        path = queue.pop(0)
        paths.append(path)
        queue.extend(below.get(path, []))
    return paths
