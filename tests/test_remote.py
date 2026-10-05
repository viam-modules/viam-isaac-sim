"""The module against the extension's real server code, with a fake backend
standing in for Isaac: covers the wire protocol, threading, error mapping,
reconnects and re-attaching."""

import asyncio
import filecmp
import math
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from viam.components.arm import JointPositions
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Vector3
from viam.utils import dict_to_struct

from isaac_module.client import NotReadyError
from isaac_module.models.arm import IsaacArm
from isaac_module.models.base import IsaacBase
from isaac_module.models.camera import IsaacCamera
from isaac_module.models.world import IsaacWorld
from isaac_module.sim_manager import MockArmHandle, SimManager, pose_mismatch
from isaac_module.spatial import look_at_quat
from viam_isaac_server import images
from viam_isaac_server.protocol import MIME_RAW_RGB
from viam_isaac_server.server import Binary, NotAttached, NotReady, Server

_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def _wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.01)


class FakeIsaac:
    """Same rpc surface as IsaacBackend; mock arms, synthetic frames."""

    def __init__(self):
        self.playing = True
        self.poses = {}  # prim path -> authored position
        self.arms = {}
        self.bases = {}
        self.cameras = {}
        self.spawned_props = []
        self.raw_frames = False

    def _attach(self, registry, name, attrs, entry):
        prim_path = attrs.get("prim_path") or f"/World/{name}"
        spawned = prim_path not in self.poses
        if spawned:
            self.poses[prim_path] = list(attrs.get("position") or [0.0, 0.0, 0.0])
        registry.setdefault(name, entry)
        return {
            "prim_path": prim_path,
            "spawned": spawned,
            "position": self.poses[prim_path],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        }

    @staticmethod
    def _lookup(registry, kind, name):
        if name not in registry:
            raise NotAttached(f"{kind} {name!r} is not attached")
        return registry[name]

    # world
    def rpc_world_status(self, ctx):
        return {
            "playing": self.playing,
            "stopped": not self.playing,
            "meters_per_unit": 1.0,
        }

    def rpc_world_play(self, ctx):
        self.playing = True

    def rpc_world_pause(self, ctx):
        self.playing = False

    async def rpc_world_reset(self, ctx):
        self.playing = False
        await asyncio.sleep(0)
        self.playing = True

    def rpc_world_spawn_prop(self, ctx, prop):
        prim_path = f"/World/{prop['name']}"
        if prim_path in self.poses:
            return {"prim_path": prim_path, "spawned": False}
        self.poses[prim_path] = prop.get("position")
        self.spawned_props.append(prop["name"])
        return {"prim_path": prim_path, "spawned": True}

    # arms
    def rpc_arm_attach(self, ctx, name, attrs):
        if attrs.get("asset") == "bogus":
            raise ValueError("unknown asset 'bogus'")
        return self._attach(self.arms, name, attrs, (MockArmHandle(name, attrs), attrs))

    async def rpc_arm_get_joint_positions(self, ctx, name):
        arm, _ = self._lookup(self.arms, "arm", name)
        if not self.playing:
            raise NotReady("the simulation is stopped")
        return arm.get_joint_positions()

    def rpc_arm_set_joint_targets(self, ctx, name, positions):
        self._lookup(self.arms, "arm", name)[0].set_joint_targets(positions)

    def rpc_arm_stop(self, ctx, name):
        self._lookup(self.arms, "arm", name)[0].stop()

    def rpc_arm_get_end_pose(self, ctx, name):
        _, attrs = self._lookup(self.arms, "arm", name)
        if not attrs.get("end_effector_prim"):
            raise NotImplementedError("set end_effector_prim in the arm config")
        return {"position": [0.3, 0.0, 0.3], "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]}

    # cameras
    def rpc_camera_attach(self, ctx, name, attrs):
        return self._attach(self.cameras, name, attrs, attrs)

    def rpc_camera_get_image(self, ctx, name, mime_type="image/jpeg"):
        attrs = self._lookup(self.cameras, "camera", name)
        shape = (int(attrs.get("height", 480)), int(attrs.get("width", 640)), 3)
        rgb = np.zeros(shape, np.uint8)
        rgb[:, :, 0] = 200
        if self.raw_frames:
            h, w = rgb.shape[:2]
            return Binary(
                {"mime_type": MIME_RAW_RGB, "width": w, "height": h}, rgb.tobytes()
            )
        return Binary(encode=lambda: images.encode_rgb(rgb, mime_type))

    # bases
    def rpc_base_attach(self, ctx, name, attrs):
        info = self._attach(self.bases, name, attrs, {"cmd": (0.0, 0.0), "owner": None})
        info.update(wheel_radius=0.03, wheel_base=0.1125)
        return info

    def rpc_base_set_velocity(self, ctx, name, linear, angular):
        base = self._lookup(self.bases, "base", name)
        base.update(cmd=(linear, angular), owner=ctx.id)

    def rpc_base_stop(self, ctx, name):
        self._lookup(self.bases, "base", name)["cmd"] = (0.0, 0.0)

    def rpc_base_is_moving(self, ctx, name):
        return self._lookup(self.bases, "base", name)["cmd"] != (0.0, 0.0)

    def on_disconnect(self, ctx):
        for base in self.bases.values():
            if base["owner"] == ctx.id:
                base["cmd"] = (0.0, 0.0)


class Extension:
    """The server on its own event loop thread, standing in for Kit's."""

    def __init__(self, port=0):
        self.backend = FakeIsaac()
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()
        self.server = Server(self.backend, self.loop, "127.0.0.1", port)
        self.server.start()
        self.port = self.server.address[1]

    def stop(self):
        if self.loop.is_closed():
            return
        self.server.stop()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)
        self.loop.close()


@pytest.fixture
def ext():
    extension = Extension()
    yield extension
    extension.stop()
    SimManager.get().close()


def _world(ext, **attrs):
    return IsaacWorld.new(
        _config("sim-world", {"address": f"127.0.0.1:{ext.port}", **attrs}), {}
    )


def test_protocol_copies_match():
    assert filecmp.cmp(
        os.path.join(_ROOT, "src", "isaac_module", "protocol.py"),
        os.path.join(
            _ROOT, "exts", "viam_isaac_server", "viam_isaac_server", "protocol.py"
        ),
        shallow=False,
    ), "src/isaac_module/protocol.py must stay identical to the extension's copy"


def test_world_status_and_controls(ext):
    world = _world(ext)

    async def scenario():
        status = await world.do_command({"command": "status"})
        assert status["connected"] is True and status["playing"] is True
        await world.do_command({"command": "pause"})
        assert ext.backend.playing is False
        await world.do_command({"command": "reset"})
        assert ext.backend.playing is True

    asyncio.run(scenario())


def test_unreachable_extension_is_a_clear_error():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # nothing listens once closed
    with pytest.raises(ConnectionError, match="viam_isaac_server"):
        IsaacWorld.new(
            _config(
                "sim-world", {"address": f"127.0.0.1:{port}", "connect_timeout_sec": 1}
            ),
            {},
        )


def test_props_spawn_once(ext):
    props = [{"name": "pick_cube", "type": "cube", "position": [0.7, 0.25, 0.03]}]
    _world(ext, props=props)
    _world(ext, props=props)  # reconfigure: already there, not spawned again
    assert ext.backend.spawned_props == ["pick_cube"]


def test_arm_moves(ext):
    _world(ext)
    arm = IsaacArm.new(
        _config(
            "my-arm",
            {"world": "sim-world", "asset": "ur20", "robot_control_freq_hz": 200},
        ),
        {},
    )

    async def scenario():
        start = await arm.get_joint_positions()
        assert start.values == pytest.approx([0.0] * 6)

        await arm.move_to_joint_positions(
            JointPositions(values=[10, -20, 30, 0, 5, -5])
        )
        end = await arm.get_joint_positions()
        assert end.values == pytest.approx([10, -20, 30, 0, 5, -5], abs=0.5)

        await arm.move_through_joint_positions(
            [
                JointPositions(values=[5, -10, 15, 0, 2, -2]),
                JointPositions(values=[0, 0, 0, 0, 0, 0]),
            ]
        )
        end = await arm.get_joint_positions()
        assert end.values == pytest.approx([0] * 6, abs=0.5)

        with pytest.raises(NotImplementedError, match="end_effector_prim"):
            await arm.get_end_position()

    asyncio.run(scenario())


def test_concurrent_calls_share_one_connection(ext):
    _world(ext)
    handle = SimManager.get().create_arm("busy-arm", {"asset": "ur20"})
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda _: handle.get_joint_positions(), range(200)))
    assert all(r == pytest.approx([0.0] * 6) for r in results)


def test_errors_map_to_python_exceptions(ext):
    _world(ext)
    with pytest.raises(ValueError, match="unknown asset"):
        IsaacArm.new(_config("bad-arm", {"world": "sim-world", "asset": "bogus"}), {})

    arm = IsaacArm.new(_config("my-arm", {"world": "sim-world", "asset": "ur20"}), {})
    ext.backend.playing = False
    with pytest.raises(NotReadyError, match="stopped"):
        asyncio.run(arm.get_joint_positions())


def test_camera_images(ext):
    _world(ext)
    cam = IsaacCamera.new(
        _config("my-cam", {"world": "sim-world", "width": 320, "height": 240}), {}
    )

    async def scenario():
        from io import BytesIO

        from PIL import Image

        for mime in ("", "image/png"):
            img = await cam.get_image(mime_type=mime)
            assert img.mime_type == (mime or "image/jpeg")
            assert Image.open(BytesIO(img.data)).size == (320, 240)

        # an extension without PIL ships raw RGB; the module encodes it
        ext.backend.raw_frames = True
        img = await cam.get_image()
        assert img.mime_type == "image/jpeg"
        assert Image.open(BytesIO(img.data)).size == (320, 240)

    asyncio.run(scenario())


def test_camera_poses_are_resolved_for_the_extension(ext):
    _world(ext)
    IsaacCamera.new(
        _config(
            "aimed-cam",
            {
                "world": "sim-world",
                "position": [1.0, 0.0, 1.0],
                "target": [0.0, 0.0, 0.0],
            },
        ),
        {},
    )
    sent = ext.backend.cameras["aimed-cam"]
    assert sent["world_pose"] == {
        "position": [1.0, 0.0, 1.0],
        "orientation_wxyz": pytest.approx(
            list(look_at_quat((1.0, 0.0, 1.0), (0.0, 0.0, 0.0)))
        ),
    }
    assert sent["spawn_pose"] == {"position": [1.0, 0.0, 1.0]}
    assert "local_pose" not in sent


def test_base_drives_and_stops_when_module_disconnects(ext):
    _world(ext)
    base = IsaacBase.new(
        _config("my-base", {"world": "sim-world", "asset": "jetbot"}), {}
    )

    async def scenario():
        assert not await base.is_moving()
        await base.set_velocity(Vector3(x=0, y=200, z=0), Vector3(x=0, y=0, z=45))
        assert await base.is_moving()
        assert ext.backend.bases["my-base"]["cmd"] == pytest.approx(
            (0.2, math.radians(45))
        )
        props = await base.get_properties()
        assert props.wheel_circumference_meters == pytest.approx(2 * math.pi * 0.03)

    asyncio.run(scenario())
    SimManager.get().close()  # the module going away
    _wait_for(lambda: ext.backend.bases["my-base"]["cmd"] == (0.0, 0.0))


def test_reattaches_after_isaac_restarts(ext):
    props = [{"name": "pick_cube", "type": "cube", "position": [0.7, 0.25, 0.03]}]
    _world(ext, props=props)
    arm = IsaacArm.new(_config("my-arm", {"world": "sim-world", "asset": "ur20"}), {})
    asyncio.run(arm.move_to_joint_positions(JointPositions(values=[10, 0, 0, 0, 0, 0])))

    # Isaac Sim restarts: same port, fresh extension that knows nothing
    port = ext.port
    ext.stop()
    client = SimManager.get()._client
    _wait_for(lambda: not client.connected)
    restarted = Extension(port=port)
    try:
        positions = asyncio.run(arm.get_joint_positions())
        assert positions.values == pytest.approx([0.0] * 6)  # a fresh arm
        assert "my-arm" in restarted.backend.arms
        assert restarted.backend.spawned_props == ["pick_cube"]
    finally:
        restarted.stop()


def test_server_drops_non_protocol_clients(ext):
    with socket.create_connection(("127.0.0.1", ext.port)) as s:
        s.sendall(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
        s.settimeout(5)
        assert s.recv(1) == b""  # closed on us
    _world(ext)  # and still serves everyone else
    assert SimManager.get().status()["connected"] is True


def test_pose_mismatch():
    info = {"position": [1.0, 0.0, 0.0], "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]}
    assert pose_mismatch({"position": [1.0, 0.0, 0.0005]}, info) is None
    assert "position" in pose_mismatch({"position": [0.0, 0.0, 0.0]}, info)
    # q and -q are the same rotation
    assert pose_mismatch({"orientation_wxyz": [-1.0, 0.0, 0.0, 0.0]}, info) is None
    assert "orientation" in pose_mismatch(
        {"orientation_wxyz": [0.0, 0.0, 0.0, 1.0]}, info
    )
    assert pose_mismatch({}, info) is None
