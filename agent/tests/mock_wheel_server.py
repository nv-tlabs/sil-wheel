#!/usr/bin/env python3
"""A tiny, dependency-free mock SIL Wheel server.

It speaks just enough of the real protocol for the agent's usage workflows to
run end to end with NO real server and NO network access - the point is to
prove the usage-skill drives the API correctly off-VPN, exactly like a fresh
external user would experience against their own deployment.

Implements:
  POST /                 login (body: ``user_login::user::pass``) -> session cookie
  GET  /whoami           auth status
  GET  /videos           search (filters a small in-memory corpus)
  GET  /classifiers_status   list of trained classifiers
  GET  /data_stats_list  data-source inventory

Run standalone:
    python tests/mock_wheel_server.py --port 8765
Or use the context manager in clean_room_smoke.py.

Uses only the Python standard library.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# A small, believable corpus. Each clip's caption drives substring search;
# classifier_select assigns a score so classifier_search returns something.
CORPUS = [
    {"clip_id": "c0001", "data_source": "MADS-1M", "caption": "a car driving through heavy rain on a highway"},
    {"clip_id": "c0002", "data_source": "MADS-1M", "caption": "construction zone with cones and a worker"},
    {"clip_id": "c0003", "data_source": "MADS-1M", "caption": "snow covered road at night with low visibility"},
    {"clip_id": "c0004", "data_source": "MADS-1M", "caption": "urban intersection with pedestrians crossing"},
    {"clip_id": "c0005", "data_source": "MADS-1M", "caption": "construction zone in the rain near an overpass"},
    {"clip_id": "c0006", "data_source": "MADS", "caption": "clear sunny day on an empty rural road"},
    {"clip_id": "c0007", "data_source": "MADS", "caption": "heavy rain and hard braking at a traffic light"},
    {"clip_id": "c0008", "data_source": "MADS-1M", "caption": "tunnel with bright headlights and reflections"},
    {"clip_id": "c0009", "data_source": "MADS-1M", "caption": "pedestrian stepping off the curb in fog"},
    {"clip_id": "c0010", "data_source": "MADS-1M", "caption": "snow and ice on a mountain pass switchback"},
    {"clip_id": "c0011", "data_source": "MADS-1M", "caption": "highway merge in light rain at dusk"},
    {"clip_id": "c0012", "data_source": "MADS-1M", "caption": "construction crane over a city street at night"},
]

CLASSIFIERS = ["Snow", "Heavy rain", "Construction zone", "Night", "Pedestrian", "interesting"]


def _video_dict(item: dict, classifier_score: float | None = None) -> dict:
    v = {
        "clip_id": item["clip_id"],
        "data_source": item["data_source"],
        "captions": {"qwen2.5-7b": [{"caption": item["caption"]}]},
        "has_trajectories": True,
        "has_embeddings": True,
    }
    if classifier_score is not None:
        v["classifier_score"] = classifier_score
    return v


def _search(params: dict[str, list[str]]) -> dict:
    """Handle the full /videos filter surface the SDK builds.

    Real enough that every documented search mode returns sensible results:
    text modes (caption/semantic/visual/caption-embed) substring-match captions;
    clip-similarity modes return the rest of the corpus; classifier modes attach
    a score; exact clip-id returns that clip; unknown filters fall back to all.
    """
    def first(k: str) -> str | None:
        vals = params.get(k)
        return vals[0] if vals else None

    data_source = first("data_source")
    n = int(first("n") or 20)

    pool = CORPUS
    if data_source:
        pool = [c for c in pool if c["data_source"] == data_source]

    classifier = first("classifier_select")
    # Any of the free-text search modes.
    text = next((first(k) for k in (
        "search", "semantic_search_text", "visual_search_text",
        "caption_embed_search", "search_comments") if first(k)), None)
    # Clip-similarity / exact-id modes.
    seed_clip = next((first(k) for k in (
        "semantic_search_clipid", "trajectory_shape_clipid") if first(k)), None)
    exact_id = first("search_clipid")
    wm_class = first("wm_class_name")
    country = first("search_country")

    hits: list[dict] = []
    if exact_id:
        hits = [_video_dict(c) for c in pool if c["clip_id"] == exact_id]
    elif classifier:
        key = classifier.lower().split()[0]
        for c in pool:
            if key in c["caption"].lower() or classifier.lower() == "interesting":
                hits.append(_video_dict(c, classifier_score=0.82))
    elif text:
        terms = [t for t in text.lower().replace(" or ", " ").split() if t]
        for c in pool:
            if any(t in c["caption"].lower() for t in terms):
                hits.append(_video_dict(c))
    elif seed_clip:
        hits = [_video_dict(c) for c in pool if c["clip_id"] != seed_clip]
    elif wm_class:
        kw = "pedestrian" if "PEDESTRIAN" in wm_class.upper() else ""
        hits = [_video_dict(c) for c in pool if kw in c["caption"].lower()] or \
               [_video_dict(c) for c in pool[:3]]
    elif country:
        hits = [_video_dict(c) for c in pool[:5]]
    else:
        hits = [_video_dict(c) for c in pool]

    return {"num_videos": len(hits), "videos": hits[:n]}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence noisy default logging
        pass

    def _json(self, obj: dict, status: int = 200, cookie: str | None = None):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8", "replace") if length else ""
        if self.path.rstrip("/") == "" and body.startswith("user_login::"):
            self._json({"ok": True}, cookie="session_id=mock-session-123; Path=/")
            return
        if body.strip() == "logout":
            self._json({"ok": True})
            return
        self._json({"error": "unknown POST"}, status=404)

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)
        if route == "/whoami":
            self._json({"authenticated": True, "user": "demo"})
        elif route == "/videos":
            self._json(_search(params))
        elif route == "/classifiers_status":
            # Server schema (2026-Q2+): trained_by_type maps embed backend -> labels.
            self._json({"trained_by_type": {"cosmos": CLASSIFIERS, "caption": CLASSIFIERS},
                        "untrained": []})
        elif route == "/data_stats_list":
            self._json({"datasets": [
                {"dataset": "MADS-1M", "num_clips": 1071385},
                {"dataset": "MADS", "num_clips": 142746},
            ]})
        elif route == "/api/vlm_judge/status":
            # Advertise VLM Judge as unavailable so callers exercise the
            # graceful "feature off" path (no NV_INFERENCE_API_KEY here).
            self._json({"enabled": False})
        else:
            self._json({"error": f"unknown route {route}"}, status=404)


class MockWheel:
    """Context manager that runs the mock server on a background thread."""

    def __init__(self, port: int = 0):
        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "MockWheel":
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Mock SIL Wheel server on http://127.0.0.1:{args.port}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()
