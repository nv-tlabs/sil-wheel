# SIL Wheel Agent

An AI agent interface to a **SIL Wheel** deployment - search and curate
autonomous-driving video clips through natural language, visual/trajectory
similarity, trained classifiers, annotations, and clustering. Open this folder
in **Cursor** or **Claude Code** and the agent picks up the skill automatically.

This is a **client**. It ships no credentials and no hardcoded server - you
point it at a SIL Wheel instance you run or have access to (`WHEEL_SERVER_URL`).

## Quickstart (end users)

```bash
cd agent
pip install -r requirements.txt
cp .env.template .env        # set WHEEL_SERVER_URL + WHEEL_USERNAME + WHEEL_PASSWORD
```

Then either:

- **Open `agent/` in Cursor or Claude Code** - the editor auto-discovers
  `.cursor/skills/sil-wheel/SKILL.md` (and the `.claude/` mirror), and you can
  just ask: *"find clips of construction zones in rain"*.
- **Or use it directly**:

```python
from sil_wheel_agent import WheelClient

client = WheelClient()
client.login()
total, results = client.search(search="construction zone in rain", data_source="MADS-1M")
print(total, [r.clip_id for r in results[:5]])
```

The full usage guide an agent reads is [`SKILL.md`](SKILL.md); deeper docs are
in [`knowledge/`](knowledge/).

## How the skill is delivered

The skill is just files an agent reads. There are two ways to deliver it:

1. **Clone / download the repo** (this folder). The `SKILL.md` and `knowledge/`
   docs are right here; the editor finds them. Zero hosting needed.
2. **Hosted manifest** (for operators serving many users): host `SKILL.md` at a
   URL and have your users' agents read it - it lists the other files to fetch
   (`knowledge/*.md`, the SDK). Whoever runs a SIL Wheel deployment sets this up
   once for their org and serves the files (over plain HTTP, no auth needed for
   the docs). This is how a fleet of agents onboard without each user cloning.

Either way the agent ends up with `SKILL.md` + `knowledge/` + the SDK.

## Testing it works (no real server)

```bash
bash tests/run_clean_room.sh
```

Copies only the public files into a throwaway dir, installs deps into a fresh
venv, and runs (1) an offline unit smoke and (2) a clean-room end-to-end against
a bundled **mock SIL Wheel server** - proving the documented workflows drive the
API with no NVIDIA network and no live server. This is the harness you point at
a real (even minimal) deployment to validate end to end.

## Layout

```
agent/
  SKILL.md              # usage skill - an agent reads this to set up + drive the API
  README.md             # this file
  requirements.txt
  .env.template         # WHEEL_SERVER_URL / WHEEL_USERNAME / WHEEL_PASSWORD
  sil_wheel_agent/
    __init__.py         # from sil_wheel_agent import WheelClient
    wheel_client.py     # the SDK (Python + CLI, all search modes)
  knowledge/            # agent context docs (search modes, anti-patterns, calibration, ...)
  examples/
    quickstart.py
  tests/
    test_public_smoke.py    # offline unit smoke (no network)
    mock_wheel_server.py    # stdlib mock SIL Wheel server
    clean_room_smoke.py     # end-to-end usage workflows vs the mock
    run_clean_room.sh       # fresh-venv clean-room runner
  .cursor/skills/sil-wheel/SKILL.md   # editor auto-discovery (Cursor)
  .claude/skills/sil-wheel/SKILL.md   # editor auto-discovery (Claude Code)
```

> **Package name:** the agent SDK ships as `sil_wheel_agent` (an HTTP client to
> a running server) to avoid colliding with the platform's own `sil_wheel`
> package, which exposes a different in-process `WheelClient`. Import the agent
> client with `from sil_wheel_agent import WheelClient`.

## License

Part of the [SIL-Wheel](../) repository; released under the same Apache 2.0
license. See the repository root [`LICENSE`](../LICENSE).
