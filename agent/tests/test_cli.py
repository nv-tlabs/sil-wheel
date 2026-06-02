"""CLI smoke against the mock server.

Runs `python sil_wheel_agent/wheel_client.py <cmd>` as a subprocess - the way a
user runs the CLI - pointed at the in-process mock via env. Confirms the
documented CLI subcommands actually work end to end.

    pytest tests/test_cli.py -q
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLI = ROOT / "sil_wheel_agent" / "wheel_client.py"


def _run(args, server_url, cwd=None):
    env = dict(os.environ)
    env.update({
        "WHEEL_SERVER_URL": server_url,
        "WHEEL_USERNAME": "demo",
        "WHEEL_PASSWORD": "demo",
    })
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        env=env, cwd=cwd, capture_output=True, text=True, timeout=60,
    )


def test_cli_search_caption(mock_server):
    r = _run(["search", "--caption", "rain", "-n", "5"], mock_server.url)
    assert r.returncode == 0, r.stderr
    assert "c000" in r.stdout, r.stdout


def test_cli_scenario(mock_server):
    r = _run(["scenario", "construction zone in rain", "--data-source", "MADS-1M"],
             mock_server.url)
    assert r.returncode == 0, r.stderr


def test_cli_export_writes_file(mock_server, tmp_path):
    # CLI export writes the -o filename relative to the working directory.
    r = _run(["export", "--caption", "rain", "-o", "cli_clips.txt"],
             mock_server.url, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    out = tmp_path / "cli_clips.txt"
    assert out.exists() and out.read_text().strip() != ""


def test_cli_version_graceful(mock_server):
    # The public skill URL may 404 until the repo is public; `version` must
    # degrade gracefully - print the local version, no traceback.
    r = _run(["version"], mock_server.url)
    assert "Local:" in r.stdout, r.stdout
    assert "Traceback" not in r.stderr, r.stderr
