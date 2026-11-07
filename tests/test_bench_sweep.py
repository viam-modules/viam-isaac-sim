from examples import bench_sweep


def test_sweep_waypoints_alternates_then_returns():
    wps = bench_sweep.sweep_waypoints([10, 20, 30, 40, 50, 60], 60, 2)
    assert [w[0] for w in wps] == [70, -50, 70, -50, 10]
    assert all(w[1:] == [20, 30, 40, 50, 60] for w in wps)


def test_sweep_waypoints_zero_cycles_only_returns_to_start():
    assert bench_sweep.sweep_waypoints([10, 20], 60, 0) == [[10, 20]]
