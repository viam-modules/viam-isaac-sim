"""Props published to a world state store: the transform encoding, and the
sync loop run against an in-memory stand-in for rdk's builtin store."""

import asyncio
import base64
import math
import threading

import pytest
from google.protobuf.json_format import ParseDict
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName, Transform
from viam.services.worldstatestore import WorldStateStore
from viam.utils import dict_to_struct, struct_to_dict

from isaac_module.models.world import IsaacWorld
from isaac_module.sim_manager import SimManager, fan_triangles
from isaac_module.world_state import (
    Collider,
    Obstacle,
    WorldStateSync,
    collider_transform,
    obstacle_transforms,
    obstacle_uuid,
    parse_collider,
    ply_bytes,
    pose_proto,
    transform_json,
    world_prefix,
)


class FakeStore(WorldStateStore):
    """rdk's builtin store, as far as the sync uses it. One DoCommand carries
    any of remove / add / update, each a list of proto JSON transforms; they
    are applied in that order, and the call takes effect whole or not at all."""

    def __init__(self, name: str = "world_state_store") -> None:
        super().__init__(name)
        self.transforms: dict[bytes, Transform] = {}
        # the commands each call carried, e.g. [["add"], ["remove", "add"]]
        self.calls: list[list[str]] = []

    async def list_uuids(self, **kwargs):
        return list(self.transforms)

    async def get_transform(self, uuid, **kwargs):
        return self.transforms[uuid]

    async def stream_transform_changes(self, **kwargs):
        raise NotImplementedError

    async def do_command(self, command, **kwargs):
        # the same trip through a protobuf Struct that gRPC gives it
        command = struct_to_dict(dict_to_struct(command))
        assert command and set(command) <= {"remove", "add", "update"}
        self.calls.append([k for k in ("remove", "add", "update") if k in command])
        staged = dict(self.transforms)
        response: dict[str, list[str]] = {}
        for kind in ("remove", "add", "update"):
            for index, item in enumerate(command.get(kind, [])):
                item = dict(item)
                paths = item.pop("updatedFields", None)
                transform = ParseDict(item, Transform())
                where = f"{kind}[{index}]"
                if kind == "add":
                    assert transform.reference_frame
                    assert transform.pose_in_observer_frame.reference_frame
                    if transform.uuid in staged:
                        raise ValueError(f"{where}: transform already exists")
                    staged[transform.uuid] = transform
                elif transform.uuid not in staged:
                    raise ValueError(f"{where}: transform not found")
                elif kind == "remove":
                    del staged[transform.uuid]
                else:
                    assert paths == ["pose_in_observer_frame.pose"]
                    merged = Transform()
                    merged.CopyFrom(staged[transform.uuid])
                    merged.pose_in_observer_frame.pose.CopyFrom(
                        transform.pose_in_observer_frame.pose
                    )
                    staged[transform.uuid] = merged
                response.setdefault(kind, []).append(item["uuid"])
        self.transforms = staged
        return response

    def by_frame(self) -> dict[str, Transform]:
        return {t.reference_frame: t for t in self.transforms.values()}


def _box(name: str, x: float = 0.0, **kwargs) -> Obstacle:
    return Obstacle(
        name=name,
        position=(x, 0.0, 0.0),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        collider=Collider("box", (0.1, 0.2, 0.3)),
        **kwargs,
    )


# ----------------------------------------------------------------------
# encoding
# ----------------------------------------------------------------------


def test_uuids_are_stable_and_share_the_world_prefix():
    a = obstacle_uuid("sim-world", "block", "collider")
    assert a == obstacle_uuid("sim-world", "block", "collider")
    assert len(a) == 16
    assert a.startswith(world_prefix("sim-world"))
    assert a != obstacle_uuid("sim-world", "block", "visual")
    assert a != obstacle_uuid("sim-world", "pad", "collider")
    assert not obstacle_uuid("other", "block", "collider").startswith(
        world_prefix("sim-world")
    )


def test_pose_is_millimeters_and_orientation_vector_degrees():
    identity = pose_proto((1.0, -0.5, 0.25))
    assert (identity.x, identity.y, identity.z) == (1000.0, -500.0, 250.0)
    assert (identity.o_x, identity.o_y, identity.o_z, identity.theta) == (0, 0, 1, 0)

    yaw90 = pose_proto((0, 0, 0), (math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)))
    assert yaw90.o_z == pytest.approx(1.0)
    assert yaw90.theta == pytest.approx(90.0)


def test_collider_transform_carries_geometry_and_color():
    obstacle = _box("block", x=2.5, color=(1.0, 0.0, 0.5))
    transform = collider_transform("sim-world", obstacle)
    assert transform.reference_frame == "block"
    assert transform.pose_in_observer_frame.reference_frame == "world"
    assert transform.pose_in_observer_frame.pose.x == 2500.0
    dims = transform.physical_object.box.dims_mm
    assert (dims.x, dims.y, dims.z) == pytest.approx((100.0, 200.0, 300.0))
    metadata = struct_to_dict(transform.metadata)
    assert base64.b64decode(metadata["colors"]) == bytes([255, 0, 128])
    assert "invisible" not in metadata


def test_sphere_and_capsule_geometry():
    sphere = collider_transform(
        "w",
        Obstacle("s", (0, 0, 0), (1, 0, 0, 0), Collider("sphere", (0.05,), (0, 0, 0.1))),
    ).physical_object
    assert sphere.sphere.radius_mm == pytest.approx(50.0)
    assert sphere.center.z == pytest.approx(100.0)
    capsule = collider_transform(
        "w", Obstacle("c", (0, 0, 0), (1, 0, 0, 0), Collider("capsule", (0.05, 0.4)))
    ).physical_object
    assert capsule.capsule.radius_mm == pytest.approx(50.0)
    assert capsule.capsule.length_mm == pytest.approx(400.0)


def test_transform_json_is_proto_json():
    encoded = transform_json(collider_transform("sim-world", _box("block", x=1.0)))
    assert base64.b64decode(encoded["uuid"]) == obstacle_uuid(
        "sim-world", "block", "collider"
    )
    assert encoded["referenceFrame"] == "block"
    assert encoded["poseInObserverFrame"]["pose"]["x"] == 1000.0


def test_parse_collider():
    assert parse_collider({"type": "box", "size": [1, 2, 3]}, "p") == Collider(
        "box", (1.0, 2.0, 3.0)
    )
    assert parse_collider(
        {"type": "sphere", "radius": 0.5, "center": [0, 0, 1]}, "p"
    ) == Collider("sphere", (0.5,), (0.0, 0.0, 1.0))
    assert parse_collider(
        {"type": "capsule", "radius": 0.1, "length": 1}, "p"
    ) == Collider("capsule", (0.1, 1.0))


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("box", "must be an object"),
        ({"type": "cone"}, "box, sphere or capsule"),
        ({"type": "box", "size": [1, 2]}, "list of 3"),
        ({"type": "box", "size": [1, 2, 0]}, "positive"),
        ({"type": "sphere"}, "radius must be a number"),
        ({"type": "capsule", "radius": 1, "length": 1}, "twice the radius"),
    ],
)
def test_parse_collider_rejects(raw, match):
    with pytest.raises(ValueError, match=match):
        parse_collider(raw, "p")


def test_render_modes():
    mesh = ply_bytes([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)])
    with_mesh = _box("chair", mesh_ply=mesh)

    (only,) = obstacle_transforms("w", with_mesh, "colliders")
    assert only.physical_object.WhichOneof("geometry_type") == "box"

    collider, visual = obstacle_transforms("w", with_mesh, "both")
    assert "invisible" not in struct_to_dict(collider.metadata)
    assert visual.reference_frame == "chair_visual"
    # a child of the collider's frame, so it follows without its own updates
    assert visual.pose_in_observer_frame.reference_frame == "chair"
    assert visual.physical_object.mesh.content_type == "ply"
    assert visual.physical_object.mesh.mesh == mesh

    collider, visual = obstacle_transforms("w", with_mesh, "models")
    assert struct_to_dict(collider.metadata)["invisible"] is True

    # nothing to show instead of the primitive, so it stays visible
    (cube,) = obstacle_transforms("w", _box("cube"), "models")
    assert "invisible" not in struct_to_dict(cube.metadata)


def test_ply_and_fan_triangles():
    assert fan_triangles([4, 3], [0, 1, 2, 3, 4, 5, 6]) == [
        (0, 1, 2),
        (0, 2, 3),
        (4, 5, 6),
    ]
    ply = ply_bytes([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)])
    header, body = ply.split(b"end_header\n")
    assert b"element vertex 3" in header and b"element face 1" in header
    assert len(body) == 3 * 12 + 13


# ----------------------------------------------------------------------
# the sync loop
# ----------------------------------------------------------------------


def _sync(store, obstacles, **kwargs) -> WorldStateSync:
    return WorldStateSync(store, lambda: list(obstacles), "sim-world", **kwargs)


def test_sync_adds_then_only_sends_what_moved():
    async def scenario():
        store = FakeStore()
        obstacles = [_box("block"), _box("pad", x=1.0)]
        sync = _sync(store, obstacles)

        await sync.sync_once()
        assert set(store.by_frame()) == {"block", "pad"}
        assert store.calls == [["add"]]

        await sync.sync_once()
        assert store.calls == [["add"]]  # nothing moved: no traffic

        obstacles[0] = _box("block", x=0.0001)  # under the threshold
        await sync.sync_once()
        assert store.calls == [["add"]]

        obstacles[0] = _box("block", x=0.5)
        await sync.sync_once()
        assert store.calls == [["add"], ["update"]]
        assert store.by_frame()["block"].pose_in_observer_frame.pose.x == 500.0
        assert store.by_frame()["block"].physical_object.HasField("box")

        # back to x=0: proto JSON drops zero fields, the update must not
        obstacles[0] = _box("block", x=0.0)
        await sync.sync_once()
        assert store.by_frame()["block"].pose_in_observer_frame.pose.x == 0.0

    asyncio.run(scenario())


def test_sync_removes_props_that_go_away_and_readds_changed_geometry():
    async def scenario():
        store = FakeStore()
        obstacles = [_box("block"), _box("pad")]
        sync = _sync(store, obstacles)
        await sync.sync_once()

        del obstacles[1]
        await sync.sync_once()
        assert set(store.by_frame()) == {"block"}

        # new geometry under the same uuid: replaced within one call
        obstacles[0] = Obstacle(
            "block", (0, 0, 0), (1, 0, 0, 0), Collider("sphere", (0.1,))
        )
        await sync.sync_once()
        assert store.by_frame()["block"].physical_object.HasField("sphere")
        assert store.calls[-1] == ["remove", "add"]


def test_one_tick_is_one_call_and_a_rejected_call_is_retried_whole():
    async def scenario():
        store = FakeStore()
        obstacles = [_box("block"), _box("pad"), _box("gone")]
        sync = _sync(store, obstacles)
        await sync.sync_once()

        obstacles[:] = [_box("block", x=1.0), _box("pad"), _box("new")]
        await sync.sync_once()
        assert store.calls[-1] == ["remove", "add", "update"]
        assert set(store.by_frame()) == {"block", "pad", "new"}

        # someone else removed one of ours: the next change to it is refused,
        # and the store keeps everything else that call carried out
        del store.transforms[collider_transform("sim-world", _box("block")).uuid]
        obstacles[:] = [_box("block", x=2.0), _box("pad"), _box("new"), _box("more")]
        with pytest.raises(ValueError, match="not found"):
            await sync.sync_once()
        assert set(store.by_frame()) == {"pad", "new"}
        assert sync.published_count == 3  # nothing recorded as sent

    asyncio.run(scenario())

    asyncio.run(scenario())


def test_sync_clears_what_an_earlier_run_left_and_leaves_others_alone():
    async def scenario():
        store = FakeStore()
        stale = collider_transform("sim-world", _box("removed_from_config"))
        foreign = collider_transform("someone-else", _box("theirs"))
        store.transforms = {stale.uuid: stale, foreign.uuid: foreign}

        sync = _sync(store, [_box("block")])
        await sync.sync_once()
        assert set(store.by_frame()) == {"block", "theirs"}
        assert store.calls == [["remove", "add"]]

        await sync.stop()
        assert set(store.by_frame()) == {"theirs"}

    asyncio.run(scenario())


def test_sync_recovers_when_the_store_is_emptied():
    async def scenario():
        store = FakeStore()
        obstacles = [_box("block")]
        sync = _sync(store, obstacles, hz=200.0)
        sync.start()
        await _until(lambda: "block" in store.by_frame())

        store.transforms = {}  # viam-server rebuilt the store
        obstacles[0] = _box("block", x=1.0)
        await _until(lambda: "block" in store.by_frame())
        assert store.by_frame()["block"].pose_in_observer_frame.pose.x == 1000.0
        assert sync.last_error == ""

        await sync.stop()
        assert store.transforms == {}

    asyncio.run(scenario())


async def _until(condition, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, "timed out"
        await asyncio.sleep(0.01)


# ----------------------------------------------------------------------
# the world component, mock sim
# ----------------------------------------------------------------------


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


WORLD_ATTRS = {
    "mock": True,
    "world_state_sync_hz": 100,
    "props": [
        {
            "name": "pick_block",
            "type": "cube",
            "position": [2.5, 0.0, 0.02],
            "size": 0.1,
            "scale": [1.0, 0.5, 0.25],
            "color": [1, 0, 0],
        },
        {"name": "pad", "type": "cube", "fixed": True, "position": [0.3, 0, 0]},
        {"name": "hidden", "type": "cube", "world_state": False},
        {"name": "table", "type": "cube", "from_toggler": True},
        {
            "name": "barrel",
            "type": "usd",
            "usd_path": "barrel.usd",
            "position": [1, 1, 0],
            "collider": {"type": "capsule", "radius": 0.3, "length": 0.9},
        },
        {"name": "unmeasured", "type": "usd", "usd_path": "thing.usd"},
    ],
}


@pytest.fixture(scope="module")
def sim():
    previous = SimManager._instance
    manager = SimManager()
    SimManager._instance = manager
    t = threading.Thread(target=manager.main_loop, daemon=True)
    t.start()
    yield manager
    manager.request_stop()
    t.join(timeout=5)
    SimManager._instance = previous


def test_validate_config_names_the_store_as_optional():
    assert IsaacWorld.validate_config(_config("w", {})) == ([], ["world_state_store"])
    assert IsaacWorld.validate_config(
        _config("w", {"world_state_store": "obstacles"})
    ) == ([], ["obstacles"])
    with pytest.raises(ValueError, match="world_state_render"):
        IsaacWorld.validate_config(_config("w", {"world_state_render": "meshes"}))
    with pytest.raises(ValueError, match="world_state_sync_hz"):
        IsaacWorld.validate_config(_config("w", {"world_state_sync_hz": 0}))


def test_sim_snapshot_selects_and_measures_props(sim):
    IsaacWorld.new(_config("sim-world", WORLD_ATTRS), {})  # no store, no loop
    obstacles = {o.name: o for o in sim.world_state_obstacles()}
    # flagged off, a toggler mirror, and a usd prop mock mode can't measure
    assert set(obstacles) == {"pick_block", "pad", "barrel"}
    assert obstacles["pick_block"].collider == Collider("box", (0.1, 0.05, 0.025))
    assert obstacles["pick_block"].position == (2.5, 0.0, 0.02)
    assert obstacles["pick_block"].color == (1.0, 0.0, 0.0)
    assert obstacles["barrel"].collider == Collider("capsule", (0.3, 0.9))


def test_world_publishes_props_and_follows_them(sim):
    async def scenario():
        store = FakeStore()
        deps = {ResourceName(name="world_state_store"): store}
        world = IsaacWorld.new(_config("sim-world", WORLD_ATTRS), deps)
        await _until(lambda: len(store.transforms) == 3)
        assert set(store.by_frame()) == {"pick_block", "pad", "barrel"}

        await world.do_command(
            {
                "command": "set_prop_pose",
                "prop": "pick_block",
                "position": [0.5, 0.25, 0.1],
                "orientation_wxyz": [1, 0, 0, 0],
            }
        )

        def block_pose():
            return store.by_frame()["pick_block"].pose_in_observer_frame.pose

        await _until(lambda: block_pose().x == pytest.approx(500.0))
        assert (block_pose().y, block_pose().z) == pytest.approx((250.0, 100.0))

        await world.do_command(
            {
                "command": "add_usd",
                "usd_path": "crate.usd",
                "prim_path": "/World/crate",
                "position": [3, 0, 0],
                "collider": {"type": "box", "size": [1, 1, 1]},
            }
        )
        await _until(lambda: "crate" in store.by_frame())
        assert store.by_frame()["crate"].pose_in_observer_frame.pose.x == 3000.0

        status = await world.do_command({"command": "status"})
        assert status["world_state"]["store"] == "world_state_store"
        assert status["world_state"]["transforms"] == 4

        # the store leaves the machine config: the world lets go of it
        world.reconfigure(_config("sim-world", WORLD_ATTRS), {})
        await _until(lambda: not store.transforms)
        status = await world.do_command({"command": "status"})
        assert "world_state" not in status
        await world.close()

    asyncio.run(scenario())
