import os
import sys

_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_ROOT, "src"))
# the extension's pure-python parts (server, protocol) run in the tests too
sys.path.insert(0, os.path.join(_ROOT, "exts", "viam_isaac_server"))

import pytest  # noqa: E402

from isaac_module.sim_manager import SimManager  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_mock_handles():
    """Mock handles outlive reconfigures by design (like prims in a stage), so
    tests reusing a component name would otherwise inherit each other's joint
    state."""
    SimManager.get()._mock_handles.clear()
    yield
