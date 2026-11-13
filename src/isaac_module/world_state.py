"""Publish the sim's props to a world state store service.

Props exist only in Isaac Sim: they are not in the machine's frame system, so
the visualizer can't show them and nothing can plan around them. When the
machine has a world state store (rdk's `rdk:builtin:builtin` model), the world
component keeps one transform per prop in it, with live poses. The sync is
one-way, sim to store.

Each prop becomes a "collider" transform: a box, sphere or capsule in the
world frame. A prop with a mesh can also get a "visual" transform, a child of
the collider's frame that carries the mesh, so only the collider's pose ever
needs updating.

Nothing here imports Isaac; SimManager.world_state_obstacles supplies the
snapshot.
"""

import asyncio
import base64
import hashlib
import math
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from google.protobuf.json_format import MessageToDict
from viam.logging import getLogger
from viam.proto.common import (
    Capsule,
    Geometry,
    Mesh,
    Pose,
    PoseInFrame,
    RectangularPrism,
    Sphere,
    Transform,
    Vector3,
)
from viam.utils import dict_to_struct

from .spatial import Quat, Vec3, quat_to_ov

LOGGER = getLogger("viam-isaac-sim.world_state")

# the name rdk gives a world state store that is configured without one
DEFAULT_STORE_NAME = "world_state_store"
WORLD_FRAME = "world"
RENDER_MODES = ("colliders", "models", "both")
DEFAULT_SYNC_HZ = 5.0
# The store takes transforms as proto JSON inside a DoCommand, so a mesh's
# bytes travel base64 encoded, a third larger than the PLY. viam-server accepts
# messages up to 32 MiB (goutils rpc.MaxMessageSize); one command stays well
# under that, and a mesh has to fit in a command with room to spare.
MAX_COMMAND_BYTES = 16 * 1024 * 1024
MAX_MESH_BYTES = 4 * 1024 * 1024
# how often to check that the store still holds what was published: a prop at
# rest sends nothing, so nothing else would notice the store being emptied
DEFAULT_VERIFY_S = 2.0

# a pose has to move this far before it is sent again, so physics jitter on a
# resting prop is not a stream of updates
POSITION_EPSILON_M = 0.0005
ANGLE_EPSILON_RAD = math.radians(0.1)

# the visualizer's draw.v1.ColorFormat.COLOR_FORMAT_RGB
_COLOR_FORMAT_RGB = 1
_IDENTITY_QUAT: Quat = (1.0, 0.0, 0.0, 0.0)
_POSE_FIELD = "pose_in_observer_frame.pose"


@dataclass(frozen=True)
class Collider:
    """A primitive shape, in meters, in the frame of the prop it belongs to.

    dims_m is (x, y, z) for a box, (radius,) for a sphere and (radius, length)
    for a capsule, whose length runs along z and includes both caps.
    """

    kind: str
    dims_m: tuple[float, ...]
    center_m: Vec3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Obstacle:
    """One prop as the sim sees it right now."""

    name: str
    position: Vec3
    orientation_wxyz: Quat
    collider: Collider
    color: tuple[float, float, float] | None = None
    # binary PLY in the prop's frame, in meters
    mesh_ply: bytes | None = None


def parse_collider(raw: Any, label: str) -> Collider:
    """Validate a prop's "collider" block:
    {"type": "box", "size": [x, y, z]} | {"type": "sphere", "radius": r} |
    {"type": "capsule", "radius": r, "length": l}, all in meters, each with an
    optional "center": [x, y, z] offset from the prop's origin."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label}: collider must be an object with a type")
    kind = str(raw.get("type", ""))
    center = _floats(raw.get("center", (0.0, 0.0, 0.0)), 3, f"{label}: collider.center")
    if kind == "box":
        dims = _floats(raw.get("size"), 3, f"{label}: collider.size")
    elif kind == "sphere":
        dims = (_float(raw.get("radius"), f"{label}: collider.radius"),)
    elif kind == "capsule":
        radius = _float(raw.get("radius"), f"{label}: collider.radius")
        length = _float(raw.get("length"), f"{label}: collider.length")
        if length < 2.0 * radius:
            raise ValueError(
                f"{label}: collider.length includes both caps, so it must be at "
                "least twice the radius"
            )
        dims = (radius, length)
    else:
        raise ValueError(
            f"{label}: collider.type must be box, sphere or capsule, not {kind!r}"
        )
    if any(v <= 0.0 for v in dims):
        raise ValueError(f"{label}: collider dimensions must be positive")
    return Collider(kind, dims, (center[0], center[1], center[2]))


def _float(value: Any, label: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number") from None
    if not math.isfinite(out):
        raise ValueError(f"{label} must be finite")
    return out


def _floats(value: Any, count: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != count:
        raise ValueError(f"{label} must be a list of {count} numbers")
    return tuple(_float(v, label) for v in value)


def world_prefix(world_name: str) -> bytes:
    """The first half of every uuid a world publishes. A store outlives the
    module, so this is how a restarted world recognizes what it left there."""
    return hashlib.sha256(f"viam-isaac-sim:{world_name}".encode()).digest()[:8]


def obstacle_uuid(world_name: str, name: str, role: str) -> bytes:
    """A stable 16-byte id for one of a prop's transforms ("collider" or
    "visual")."""
    return world_prefix(world_name) + hashlib.sha256(f"{role}:{name}".encode()).digest()[:8]


def pose_proto(position_m: Sequence[float], quat_wxyz: Quat = _IDENTITY_QUAT) -> Pose:
    """A sim pose (meters, w-x-y-z quaternion) as a Viam pose (millimeters,
    orientation vector in degrees)."""
    ox, oy, oz, theta = quat_to_ov(quat_wxyz)
    return Pose(
        x=position_m[0] * 1000.0,
        y=position_m[1] * 1000.0,
        z=position_m[2] * 1000.0,
        o_x=ox,
        o_y=oy,
        o_z=oz,
        theta=math.degrees(theta),
    )


def collider_geometry(collider: Collider, label: str) -> Geometry:
    center = pose_proto(collider.center_m)
    dims = [v * 1000.0 for v in collider.dims_m]
    if collider.kind == "box":
        return Geometry(
            center=center,
            label=label,
            box=RectangularPrism(dims_mm=Vector3(x=dims[0], y=dims[1], z=dims[2])),
        )
    if collider.kind == "sphere":
        return Geometry(center=center, label=label, sphere=Sphere(radius_mm=dims[0]))
    if collider.kind == "capsule":
        return Geometry(
            center=center,
            label=label,
            capsule=Capsule(radius_mm=dims[0], length_mm=dims[1]),
        )
    raise ValueError(f"unknown collider kind {collider.kind!r}")


def _metadata(color: Sequence[float] | None, invisible: bool) -> dict[str, Any]:
    """Only keys the visualizer reads; it ignores the rest."""
    out: dict[str, Any] = {}
    if color is not None:
        rgb = bytes(max(0, min(255, round(float(c) * 255.0))) for c in color[:3])
        out["colors"] = base64.b64encode(rgb).decode()
        out["color_format"] = _COLOR_FORMAT_RGB
    if invisible:
        out["invisible"] = True
    return out


def collider_transform(
    world_name: str, obstacle: Obstacle, invisible: bool = False
) -> Transform:
    transform = Transform(
        uuid=obstacle_uuid(world_name, obstacle.name, "collider"),
        reference_frame=obstacle.name,
        pose_in_observer_frame=PoseInFrame(
            reference_frame=WORLD_FRAME,
            pose=pose_proto(obstacle.position, obstacle.orientation_wxyz),
        ),
        physical_object=collider_geometry(obstacle.collider, obstacle.name),
    )
    metadata = _metadata(obstacle.color, invisible)
    if metadata:
        transform.metadata.CopyFrom(dict_to_struct(metadata))
    return transform


def visual_transform(world_name: str, obstacle: Obstacle) -> Transform:
    """The prop's mesh, as a child of its collider's frame so it follows the
    collider without updates of its own."""
    if obstacle.mesh_ply is None:
        raise ValueError(f"{obstacle.name} has no mesh")
    label = f"{obstacle.name}_visual"
    transform = Transform(
        uuid=obstacle_uuid(world_name, obstacle.name, "visual"),
        reference_frame=label,
        pose_in_observer_frame=PoseInFrame(
            reference_frame=obstacle.name, pose=pose_proto((0.0, 0.0, 0.0))
        ),
        physical_object=Geometry(
            center=pose_proto((0.0, 0.0, 0.0)),
            label=label,
            mesh=Mesh(content_type="ply", mesh=obstacle.mesh_ply),
        ),
    )
    metadata = _metadata(obstacle.color, False)
    if metadata:
        transform.metadata.CopyFrom(dict_to_struct(metadata))
    return transform


def obstacle_transforms(
    world_name: str, obstacle: Obstacle, render: str
) -> list[Transform]:
    """What a prop publishes under a render mode: "colliders" is the primitive
    alone, "both" adds the mesh, and "models" adds the mesh and hides the
    primitive. A prop without a mesh is always just its visible primitive."""
    has_mesh = render != "colliders" and obstacle.mesh_ply is not None
    transforms = [
        collider_transform(
            world_name, obstacle, invisible=has_mesh and render == "models"
        )
    ]
    if has_mesh:
        transforms.append(visual_transform(world_name, obstacle))
    return transforms


def transform_json(transform: Transform) -> dict[str, Any]:
    """The proto JSON form the store's DoCommands take (uuid in base64)."""
    return MessageToDict(transform)


def ply_bytes(
    vertices: Sequence[Sequence[float]], triangles: Sequence[Sequence[int]]
) -> bytes:
    """A binary little-endian PLY triangle mesh."""
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {len(triangles)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    body = bytearray()
    for vertex in vertices:
        body += struct.pack("<3f", float(vertex[0]), float(vertex[1]), float(vertex[2]))
    for triangle in triangles:
        body += struct.pack("<B3i", 3, int(triangle[0]), int(triangle[1]), int(triangle[2]))
    return header + bytes(body)


def _uuid_json(uuid: bytes) -> dict[str, Any]:
    return {"uuid": base64.b64encode(uuid).decode()}


# generous sizes for the small items of a command, in bytes on the wire
_REMOVE_BYTES = 128
_UPDATE_BYTES = 512
_ADD_OVERHEAD_BYTES = 1024


def _encoded_size(transform: Transform) -> int:
    """About how large a transform is as proto JSON: its bytes fields (a mesh)
    grow by a third in base64, and the rest is small."""
    return transform.ByteSize() * 4 // 3 + _ADD_OVERHEAD_BYTES


def _commands(
    removes: Sequence[dict[str, Any]],
    adds: Sequence[tuple[dict[str, Any], int]],
    updates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The DoCommands for one tick: a single command when everything fits in
    MAX_COMMAND_BYTES, otherwise the removes and updates with as many adds as
    fit, then the remaining adds in commands of their own. Removes go first so
    a transform can be replaced under the same uuid."""
    first: dict[str, Any] = {}
    if removes:
        first["remove"] = list(removes)
    if updates:
        first["update"] = list(updates)
    commands = [first]
    size = len(removes) * _REMOVE_BYTES + len(updates) * _UPDATE_BYTES
    for item, item_size in adds:
        if size + item_size > MAX_COMMAND_BYTES and commands[-1]:
            commands.append({})
            size = 0
        commands[-1].setdefault("add", []).append(item)
        size += item_size
    return [command for command in commands if command]


def _pose_moved(a: tuple[Vec3, Quat], b: tuple[Vec3, Quat]) -> bool:
    if math.dist(a[0], b[0]) > POSITION_EPSILON_M:
        return True
    dot = abs(sum(x * y for x, y in zip(a[1], b[1])))
    return 2.0 * math.acos(min(1.0, dot)) > ANGLE_EPSILON_RAD


@dataclass
class _Published:
    # the transform with its pose cleared; a change here means remove and re-add
    signature: bytes
    # the pose last sent, or None for a transform that never moves on its own
    pose: tuple[Vec3, Quat] | None


class WorldStateSync:
    """Keeps a world state store matching the sim's props.

    `snapshot` blocks (it hops to the sim thread), so it runs in a worker
    thread. `store` is a world state store client.
    """

    def __init__(
        self,
        store: Any,
        snapshot: Callable[[], Sequence[Obstacle]],
        world_name: str,
        hz: float = DEFAULT_SYNC_HZ,
        render: str = "colliders",
        verify_s: float = DEFAULT_VERIFY_S,
    ) -> None:
        if render not in RENDER_MODES:
            raise ValueError(f"render must be one of {list(RENDER_MODES)}")
        self._store = store
        self._snapshot = snapshot
        self._world_name = world_name
        self._period_s = 1.0 / hz
        self._render = render
        self._prefix = world_prefix(world_name)
        self._published: dict[bytes, _Published] = {}
        # the store may hold transforms from an earlier run, or have been
        # emptied by a viam-server rebuild: start from what it really has
        self._needs_reconcile = True
        self._verify_every = max(1, round(verify_s * hz))
        self._ticks_since_verify = 0
        self._task: asyncio.Task | None = None
        self.last_error = ""

    @property
    def store_name(self) -> str:
        return str(getattr(self._store, "name", ""))

    @property
    def published_count(self) -> int:
        return len(self._published)

    def start(self) -> None:
        """Start syncing on the running event loop."""
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self.run())

    async def run(self) -> None:
        while True:
            try:
                await self.sync_once()
                self.last_error = ""
            except asyncio.CancelledError:
                raise
            except Exception as e:
                message = f"{type(e).__name__}: {e}"
                if message != self.last_error:
                    LOGGER.warning(
                        "world state sync to %r failed, will retry: %s",
                        self.store_name,
                        message,
                    )
                self.last_error = message
            await asyncio.sleep(self._period_s)

    async def stop(self) -> None:
        """Stop syncing and take what this sync published out of the store.

        Only its own transforms: a newer sync for the same world shares the
        uuid prefix, and may already have published by the time this runs.
        Anything else left behind is cleared by the next sync's first tick.
        """
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        published, self._published = self._published, {}
        self._needs_reconcile = True
        if not published:
            return
        try:
            # one missing uuid would fail the whole remove
            leftovers = [u for u in await self._ours_in_store() if u in published]
            if leftovers:
                await self._store.do_command(
                    {"remove": [_uuid_json(uuid) for uuid in leftovers]}
                )
        except Exception as e:
            LOGGER.warning("could not clear world state store %r: %s", self.store_name, e)

    async def _ours_in_store(self) -> list[bytes]:
        return [
            bytes(u)
            for u in await self._store.list_uuids()
            if bytes(u).startswith(self._prefix)
        ]

    async def sync_once(self) -> None:
        """Bring the store in line with one snapshot of the sim."""
        obstacles = await asyncio.to_thread(self._snapshot)
        desired: dict[bytes, tuple[Transform, _Published]] = {}
        for obstacle in obstacles:
            pose = (obstacle.position, obstacle.orientation_wxyz)
            for transform in obstacle_transforms(self._world_name, obstacle, self._render):
                is_collider = transform.reference_frame == obstacle.name
                static = Transform()
                static.CopyFrom(transform)
                if is_collider:
                    static.pose_in_observer_frame.ClearField("pose")
                desired[transform.uuid] = (
                    transform,
                    _Published(
                        static.SerializeToString(deterministic=True),
                        pose if is_collider else None,
                    ),
                )

        self._ticks_since_verify += 1
        if not self._needs_reconcile and self._ticks_since_verify >= self._verify_every:
            self._ticks_since_verify = 0
            if set(await self._ours_in_store()) != set(self._published):
                LOGGER.info(
                    "world state store %r no longer matches what was published; "
                    "republishing",
                    self.store_name,
                )
                self._needs_reconcile = True

        # Work out the whole change first. The store applies a call's remove,
        # add and update lists in that order, all or nothing, so a tick that
        # fits in one command never shows half applied.
        if self._needs_reconcile:
            published: dict[bytes, _Published] = {}
            removed = await self._ours_in_store()
        else:
            published = dict(self._published)
            removed = [
                uuid
                for uuid, state in published.items()
                if uuid not in desired or desired[uuid][1].signature != state.signature
            ]
            for uuid in removed:
                del published[uuid]

        added = {u: d for u, d in desired.items() if u not in published}
        moved: dict[bytes, tuple[Vec3, Quat]] = {}
        for uuid, state in published.items():
            pose = desired[uuid][1].pose
            if pose is not None and state.pose is not None and _pose_moved(pose, state.pose):
                moved[uuid] = pose

        removes = [_uuid_json(uuid) for uuid in removed]
        # name the whole pose: proto JSON leaves out zero fields, and the
        # store would otherwise keep a stale x when a prop returns to x=0
        updates = [
            {
                **_uuid_json(uuid),
                "poseInObserverFrame": {"pose": MessageToDict(pose_proto(*pose))},
                "updatedFields": [_POSE_FIELD],
            }
            for uuid, pose in moved.items()
        ]
        adds = [(transform_json(t), _encoded_size(t)) for t, _ in added.values()]
        try:
            for command in _commands(removes, adds, updates):
                await self._store.do_command(command)
        except Exception:
            # some of it may have been applied: start again from what the
            # store really holds
            self._needs_reconcile = True
            raise

        # only now that the store has taken it
        published.update({u: state for u, (_, state) in added.items()})
        for uuid, pose in moved.items():
            published[uuid] = _Published(published[uuid].signature, pose)
        self._published = published
        self._needs_reconcile = False
        if removed or added:
            self._ticks_since_verify = 0
