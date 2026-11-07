import logging
import sys
import types

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.world import IsaacWorld
from isaac_module.sim_manager import SimManager, kit_settings_argv, validate_kit_settings


def _config(attrs):
    return ComponentConfig(name="w", attributes=dict_to_struct(attrs))


def test_argv_formats_bools_and_integral_floats():
    # viam structs deliver every number as a float
    assert kit_settings_argv(
        {
            "/rtx/reflections/enabled": False,
            "/rtx/post/dlss/execMode": 0.0,
            "/a/b": 0.5,
            "/big": 1e20,
        }
    ) == [
        "--/rtx/reflections/enabled=false",
        "--/rtx/post/dlss/execMode=0",
        "--/a/b=0.5",
        "--/big=1e+20",
    ]


@pytest.mark.parametrize(
    "bad",
    [
        [],
        {"rtx/x": True},
        {"/a=b": 1},
        {"/a b": 1},
        {"/x": "has space"},
        {"/x": "a=b"},
        {"/x": float("nan")},
        {"/x": None},
        {"/x": [1]},
    ],
)
def test_validate_rejects(bad):
    with pytest.raises(ValueError):
        validate_kit_settings(bad)


def test_validate_accepts():
    assert validate_kit_settings({"/x": True, "/y": 2.0, "/z": "fast"}) == {
        "/x": True,
        "/y": 2.0,
        "/z": "fast",
    }


def test_world_validate_rejects_bad_kit_settings():
    with pytest.raises(ValueError):
        IsaacWorld.validate_config(_config({"kit_settings": {"no/slash": 1}}))


def test_apply_kit_settings_per_key(monkeypatch, caplog):
    store = {"/a": 1, "/bad": 2, "/c": 3}

    class Fake:
        def get(self, path):
            return store.get(path)

        def set(self, path, value):
            if path == "/bad":
                raise RuntimeError("nope")
            store[path] = value

    carb = types.ModuleType("carb")
    carb.settings = types.ModuleType("carb.settings")
    carb.settings.get_settings = lambda: Fake()
    monkeypatch.setitem(sys.modules, "carb", carb)
    monkeypatch.setitem(sys.modules, "carb.settings", carb.settings)

    manager = SimManager()
    with caplog.at_level(logging.INFO):
        SimManager._apply_kit_settings(
            manager, {"/a": 10.0, "/bad": 5, "/unknown": True, "/c": 4}
        )

    assert store["/c"] == 4  # applied after the failing key
    text = caplog.text
    assert "could not apply kit setting /bad" in text
    assert "unknown kit setting path, Kit may ignore it: /unknown" in text
    assert "kit settings applied (read back)" in text
    assert "'/a': {'before': 1, 'after': 10}" in text
