import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.world import IsaacWorld
from isaac_module.sim_manager import kit_settings_argv, validate_kit_settings


def _config(attrs):
    return ComponentConfig(name="w", attributes=dict_to_struct(attrs))


def test_argv_formats_bools_and_integral_floats():
    # viam structs deliver every number as a float
    assert kit_settings_argv(
        {"/rtx/reflections/enabled": False, "/rtx/post/dlss/execMode": 0.0, "/a/b": 0.5}
    ) == [
        "--/rtx/reflections/enabled=false",
        "--/rtx/post/dlss/execMode=0",
        "--/a/b=0.5",
    ]


@pytest.mark.parametrize(
    "bad",
    [
        [],
        {"rtx/x": True},
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
