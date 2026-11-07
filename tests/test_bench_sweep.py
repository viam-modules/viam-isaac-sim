from examples import bench_sweep


def test_sweep_waypoints_alternates_then_returns():
    wps = bench_sweep.sweep_waypoints([10, 20, 30, 40, 50, 60], 60, 2)
    assert [w[0] for w in wps] == [70, -50, 70, -50, 10]
    assert all(w[1:] == [20, 30, 40, 50, 60] for w in wps)


def test_sweep_waypoints_zero_cycles_only_returns_to_start():
    assert bench_sweep.sweep_waypoints([10, 20], 60, 0) == [[10, 20]]


class FakeArm:
    def __init__(self):
        self.cmds = []

    async def do_command(self, cmd):
        self.cmds.append(cmd)


class FakeWorld:
    def __init__(self, fail=()):
        self.fail, self.cmds = fail, []

    async def do_command(self, cmd):
        self.cmds.append(cmd["command"])
        if cmd["command"] in self.fail:
            raise RuntimeError("boom")
        if cmd["command"] == "get_prop_pose":
            return {"pose": {"position": [0, 0, 0], "orientation_wxyz": [1, 0, 0, 0]}}
        return {}


def _run(world, arm, monkeypatch):
    import asyncio
    from argparse import Namespace

    async def nosleep(_):
        pass

    monkeypatch.setattr(asyncio, "sleep", nosleep)
    args = Namespace(
        output_dir="o", cycles=1, amplitude_deg=90.0, settle_sec=5, hold_sec=3,
        arm="a", gripper="g", fixed_camera="f", wrist_camera="w", prop="p",
    )
    return asyncio.run(bench_sweep._run(args, world, arm))


def test_run_latches_radians_from_zero_pose(monkeypatch):
    import math

    arm, world = FakeArm(), FakeWorld()
    out = _run(world, arm, monkeypatch)
    vals = [c["values"][0] for c in arm.cmds]
    assert arm.cmds[0]["values"] == [0.0] * 6
    assert vals == [0.0, math.pi / 2, -math.pi / 2, 0.0]
    assert out["error"] is None and out["waypoints"] == 3
    assert "stop_episode" in world.cmds


def test_run_no_stop_when_start_fails(monkeypatch):
    world = FakeWorld(fail={"start_episode"})
    out = _run(world, FakeArm(), monkeypatch)
    assert "stop_episode" not in world.cmds
    assert out["error"] == "RuntimeError: boom"


def test_run_stop_failure_becomes_error(monkeypatch):
    out = _run(FakeWorld(fail={"stop_episode"}), FakeArm(), monkeypatch)
    assert out["error"] == "RuntimeError: boom"
