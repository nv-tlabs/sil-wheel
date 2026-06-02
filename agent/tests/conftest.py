"""Test path setup + shared fixtures for the public agent tree.

Adds the agent/ root to sys.path so `import sil_wheel_agent` resolves, and the
package dir so the legacy `import wheel_client` also works. Provides a `client`
fixture wired to an in-process mock SIL Wheel server for end-to-end tests with
no real server and no network.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                       # `from sil_wheel_agent import ...`
sys.path.insert(0, str(ROOT / "sil_wheel_agent"))   # `import wheel_client`
sys.path.insert(0, str(Path(__file__).resolve().parent))  # `import mock_wheel_server`


@pytest.fixture(scope="session")
def mock_server():
    from mock_wheel_server import MockWheel
    with MockWheel() as wheel:
        os.environ["WHEEL_SERVER_URL"] = wheel.url
        os.environ.setdefault("WHEEL_USERNAME", "demo")
        os.environ.setdefault("WHEEL_PASSWORD", "demo")
        yield wheel


@pytest.fixture
def client(mock_server):
    from sil_wheel_agent import WheelClient
    c = WheelClient()
    assert c.login(), "login against mock server failed"
    return c
