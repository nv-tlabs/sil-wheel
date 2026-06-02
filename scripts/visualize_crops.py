#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Debug viewer for Florence2+SigCLIP crop embeddings.

Starts a local HTTP server that lets you browse the cropped regions produced
by extract_florence2_sigclip_embeddings.py in the browser.  Videos are
fetched on demand from Wheel's S3.

Usage
-----
    python visualize_crops.py /lustre/.../visual_embeddings/ \\
        --bucket my-bucket \\
        --s3-key-template "some/prefix/{clip_id}.mp4" \\
        --port 8765
"""
import argparse
import io
import json
import pickle
import sys
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from decord import VideoReader
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# LRU cache (copied from utils.py to keep this file self-contained)
# ---------------------------------------------------------------------------
class LRUDict(dict):
    def __init__(self, *args, **kwargs):
        self.cache_size = kwargs.pop("size", 10)
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if key in self:
            self.pop(key)
        if len(self) >= self.cache_size:
            self.pop(next(iter(self)))
        super().__setitem__(key, value)

    def __getitem__(self, key):
        value = self.pop(key)
        super().__setitem__(key, value)
        return value

    def keys(self):
        return list(super().keys())

    def values(self):
        return list(super().values())

    def items(self):
        return list(super().items())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
class PklStore:
    def __init__(self):
        self._items = []
        self._lock = threading.Lock()

    def _print_status(self):
        n_clips = len({it["clip_id"] for it in self._items})
        print(
            f"Loaded {len(self._items)} items from "
            f"{n_clips} clips",
            flush=True,
        )

    def extend(self, new_items):
        with self._lock:
            self._items.extend(new_items)

    def labels(self):
        with self._lock:
            return sorted({it.get("label", "") for it in self._items})

    def query(self, label=None, clip_id=None, page=1, per_page=50):
        with self._lock:
            rows = self._items
            if label:
                rows = [it for it in rows if it.get("label") == label]
            if clip_id:
                rows = [
                    it for it in rows
                    if clip_id in it.get("clip_id", "")
                ]
            total = len(rows)
            start = (page - 1) * per_page
            return rows[start : start + per_page], total

    def clip_ids_by_label(self, label=None, clip_id=None):
        groups = {}
        with self._lock:
            for it in self._items:
                lbl = it.get("label", "")
                cid = it.get("clip_id", "")
                if label and lbl != label:
                    continue
                if clip_id and clip_id not in cid:
                    continue
                groups.setdefault(lbl, set()).add(cid)
        return {lbl: sorted(cids) for lbl, cids in sorted(groups.items())}


# ---------------------------------------------------------------------------
# S3 video + frame cache
# ---------------------------------------------------------------------------
class S3VideoCache:
    def __init__(
        self, sources, video_cache_size, frame_cache_size,
        clip_to_key=None,
    ):
        """sources: list of (bucket, profile, endpoint) tuples."""
        self._clients = []
        self._buckets = []
        for bucket, profile, endpoint in sources:
            sess = boto3.Session(
                profile_name=profile, region_name="us-east-1"
            )
            self._clients.append(sess.client(
                "s3",
                endpoint_url=endpoint,
                config=Config(
                    max_pool_connections=50,
                    read_timeout=60,
                    connect_timeout=10,
                ),
            ))
            self._buckets.append(bucket)
        self._clip_to_key = clip_to_key or {}
        self._videos = LRUDict(size=video_cache_size)
        self._frames = LRUDict(size=frame_cache_size)
        self._lock = threading.Lock()

    def _resolve(self, clip_id):
        """Return (client, bucket, key) for a clip_id."""
        if clip_id in self._clip_to_key:
            key, src_idx = self._clip_to_key[clip_id]
            return self._clients[src_idx], self._buckets[src_idx], key
        return self._clients[0], self._buckets[0], clip_id

    def _fetch_video(self, clip_id):
        with self._lock:
            if clip_id in self._videos:
                return self._videos[clip_id]
        client, bucket, key = self._resolve(clip_id)
        resp = client.get_object(Bucket=bucket, Key=key)
        data = resp["Body"].read()
        with self._lock:
            self._videos[clip_id] = data
        return data

    def get_frame(self, clip_id, frame_index):
        cache_key = f"{clip_id}:{frame_index}"
        with self._lock:
            if cache_key in self._frames:
                return self._frames[cache_key]

        data = self._fetch_video(clip_id)
        reader = VideoReader(io.BytesIO(data))
        idx = min(frame_index, len(reader) - 1)
        arr = reader.get_batch([idx]).asnumpy()[0]
        img = Image.fromarray(arr).convert("RGB")
        with self._lock:
            self._frames[cache_key] = img
        return img


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class CropServer(BaseHTTPRequestHandler):
    def __init__(self, *args, store=None, cache=None, wheel_url=None, **kwargs):
        self.store = store
        self.cache = cache
        self.wheel_url = wheel_url
        super().__init__(*args, **kwargs)

    def log_message(self, fmt, *args):
        print(fmt % args, flush=True)

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _send_jpeg(self, img):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        body = buf.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", len(body))
        self.send_header("Cache-Control", "max-age=3600, public")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        path = parsed.path

        try:
            if path in ("/", "/index.html"):
                self._route_html()
            elif path == "/clip":
                self._route_clip(qs)
            elif path == "/api/labels":
                self._route_labels()
            elif path == "/api/items":
                self._route_items(qs)
            elif path == "/api/clip_ids":
                self._route_clip_ids(qs)
            elif path == "/api/crop":
                self._route_crop(qs)
            else:
                self.send_error(404)
        except Exception as e:
            print(f"Error handling {path}: {e}", flush=True)
            self.send_error(500, str(e))

    def _route_html(self):
        body = EMBEDDED_HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _route_clip(self, qs):
        try:
            clip_id = qs["clip_id"][0]
            frame_index = int(qs["frame_index"][0])
            bbox = qs["bbox"][0]
            [float(v) for v in bbox.split(",")]
        except (KeyError, ValueError, IndexError):
            self.send_error(400, "Missing or invalid parameters")
            return

        label = qs.get("label", [""])[0]
        camera = qs.get("camera", [""])[0]
        body = render_clip_page(
            clip_id=clip_id,
            frame_index=frame_index,
            bbox=bbox,
            label=label,
            camera=camera,
            wheel_url=self.wheel_url,
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _route_labels(self):
        self._send_json(self.store.labels())

    def _route_clip_ids(self, qs):
        label = qs.get("label", [None])[0] or None
        clip_id = qs.get("clip_id", [None])[0] or None
        if not label and not clip_id:
            self.send_error(400, "Specify at least one of label or clip_id")
            return
        groups = self.store.clip_ids_by_label(label=label, clip_id=clip_id)
        body = json.dumps(groups, indent=2).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header(
            "Content-Disposition",
            'attachment; filename="clip_ids.json"',
        )
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _route_items(self, qs):
        def _int(key, default):
            try:
                return int(qs[key][0])
            except (KeyError, ValueError, IndexError):
                return default

        label = qs.get("label", [None])[0] or None
        clip_id = qs.get("clip_id", [None])[0] or None
        page = max(1, _int("page", 1))
        per_page = max(1, min(_int("per_page", 50), 500))

        rows, total = self.store.query(
            label=label, clip_id=clip_id, page=page, per_page=per_page,
        )
        self._send_json({
            "total": total,
            "page": page,
            "per_page": per_page,
            "items": [
                {
                    "clip_id": it["clip_id"],
                    "camera": it.get("camera"),
                    "frame_index": int(it["frame_index"]),
                    "bbox_xyxy": [float(v) for v in it["bbox_xyxy"]],
                    "label": it.get("label", ""),
                }
                for it in rows
            ],
        })

    def _route_crop(self, qs):
        try:
            clip_id = qs["clip_id"][0]
            frame_index = int(qs["frame_index"][0])
            bbox = [float(v) for v in qs["bbox"][0].split(",")]
            full = qs.get("full", ["0"])[0] == "1"
        except (KeyError, ValueError, IndexError):
            self.send_error(400, "Missing or invalid parameters")
            return

        try:
            frame = self.cache.get_frame(clip_id, frame_index)
        except ClientError as e:
            self.log_error("S3 error for %s: %s", clip_id, e)
            self.send_error(404, "Video not found in S3")
            return
        except Exception as e:
            self.log_error(
                "Frame extraction failed for %s:%d: %s", clip_id, frame_index, e
            )
            self.send_error(500, "Frame extraction failed")
            return

        x1, y1, x2, y2 = (int(v) for v in bbox)
        if full:
            img = frame.copy()
            draw = ImageDraw.Draw(img)
            for w in range(3):
                draw.rectangle(
                    [x1 - w, y1 - w, x2 + w, y2 + w], outline=(255, 80, 0),
                )
        else:
            img = frame.crop((x1, y1, x2, y2))
            if img.width < 1 or img.height < 1:
                self.send_error(400, "Empty crop region")
                return

        self._send_jpeg(img)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


# ---------------------------------------------------------------------------
# Clip detail page (rendered per-request so clip_id can be linked to Wheel)
# ---------------------------------------------------------------------------
def render_clip_page(clip_id, frame_index, bbox, label, camera, wheel_url):
    import html as _html
    from urllib.parse import quote as _quote

    crop_params = (
        f"clip_id={_quote(clip_id)}&frame_index={frame_index}"
        f"&bbox={_quote(bbox)}"
    )
    full_url = f"/api/crop?{crop_params}&full=1"
    crop_url = f"/api/crop?{crop_params}"

    wheel_link_html = ""
    if wheel_url:
        base = wheel_url.rstrip("/")
        href = f"{base}/#page=0&search_clipid={_quote(clip_id)}"
        wheel_link_html = (
            f'<a class="wheel-link" href="{_html.escape(href)}" '
            f'target="_blank" rel="noopener noreferrer">Open in SIL-Wheel &rarr;</a>'
        )

    bbox_list = [round(float(v)) for v in bbox.split(",")]
    rows = [
        ("clip_id", clip_id),
        ("camera", camera or "—"),
        ("frame_index", str(frame_index)),
        ("label", label or "—"),
        ("bbox_xyxy", "[" + ", ".join(str(v) for v in bbox_list) + "]"),
    ]
    meta_html = "".join(
        f"<dt>{_html.escape(k)}</dt><dd>{_html.escape(v)}</dd>"
        for k, v in rows
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_html.escape(clip_id)} &mdash; Crop Viewer</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui,sans-serif;background:#111;color:#eee;padding:20px;min-height:100vh}}
header{{display:flex;align-items:center;gap:18px;flex-wrap:wrap;margin-bottom:18px;border-bottom:1px solid #2a2a2a;padding-bottom:14px}}
h1{{font-size:16px;font-weight:600;color:#ffa040;word-break:break-all}}
.wheel-link{{background:#ff5000;color:#fff;padding:7px 14px;border-radius:4px;text-decoration:none;font-size:13px;font-weight:600}}
.wheel-link:hover{{background:#ff6a1a}}
#images{{display:flex;gap:14px;flex-wrap:wrap;justify-content:center;margin-bottom:18px}}
#images figure{{display:flex;flex-direction:column;align-items:center;gap:6px;flex:1;min-width:260px}}
#images img{{max-width:100%;max-height:65vh;object-fit:contain;border-radius:4px;border:1px solid #333;background:#1e1e1e}}
#images figcaption{{font-size:11px;color:#777}}
dl{{font-size:13px;line-height:1.9;background:#1a1a1a;border-radius:6px;padding:14px 18px;max-width:720px}}
dt{{color:#888;float:left;clear:left;width:110px}}
dd{{margin-left:120px;color:#ddd;word-break:break-all}}
</style>
</head>
<body>
<header>
  <h1>{_html.escape(clip_id)}</h1>
  {wheel_link_html}
</header>
<div id="images">
  <figure>
    <img src="{_html.escape(full_url)}" alt="full frame">
    <figcaption>Full frame (box = detected region)</figcaption>
  </figure>
  <figure>
    <img src="{_html.escape(crop_url)}" alt="crop">
    <figcaption>Crop</figcaption>
  </figure>
</div>
<dl>{meta_html}</dl>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Embedded HTML / CSS / JS
# ---------------------------------------------------------------------------
EMBEDDED_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Crop Viewer</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#111;color:#eee;min-height:100vh}

/* ---- filter bar ---- */
#bar{
  display:flex;gap:8px;align-items:center;flex-wrap:wrap;
  padding:10px 16px;background:#1a1a1a;
  position:sticky;top:0;z-index:10;border-bottom:1px solid #2a2a2a;
}
#bar select,#bar input,#bar button{
  background:#252525;color:#eee;border:1px solid #444;
  padding:5px 10px;border-radius:4px;font-size:13px;
}
#bar input{width:200px}
#bar button{cursor:pointer}
#bar button:hover:not(:disabled){background:#333}
#bar button:disabled{opacity:.4;cursor:not-allowed}
#bar .sep{flex:1}
#count{font-size:13px;color:#777}

/* ---- grid ---- */
#grid{
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(210px,1fr));
  gap:10px;padding:14px;
}
.tile{
  background:#1e1e1e;border-radius:6px;overflow:hidden;
  cursor:pointer;transition:transform .12s,box-shadow .12s;
}
.tile:hover{transform:scale(1.03);box-shadow:0 0 0 2px #ff5000}
.tile img{
  width:100%;height:155px;object-fit:cover;display:block;
  background:#2a2a2a;
}
.tile-meta{padding:7px 9px;font-size:12px;line-height:1.6}
.t-label{font-weight:600;color:#ffa040;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.t-clip{color:#999;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.t-frame{color:#555;font-size:11px}

/* ---- pagination ---- */
#pager{
  display:flex;justify-content:center;align-items:center;
  gap:14px;padding:16px;
}
#pager button{padding:5px 14px}
#pager span{font-size:13px;color:#aaa}

</style>
</head>
<body>

<div id="bar">
  <label for="sel-label" style="font-size:13px;color:#aaa">Label</label>
  <select id="sel-label"><option value="">All labels</option></select>
  <input id="inp-clip" type="text" placeholder="Filter by clip_id…">
  <label for="sel-pp" style="font-size:13px;color:#aaa">Per page</label>
  <select id="sel-pp">
    <option value="25">25</option>
    <option value="50" selected>50</option>
    <option value="100">100</option>
    <option value="200">200</option>
  </select>
  <button id="btn-apply">Apply</button>
  <button id="btn-download" disabled title="Apply a label or clip_id filter first">Download clip_ids</button>
  <span class="sep"></span>
  <span id="count"></span>
</div>

<div id="grid"></div>

<div id="pager">
  <button id="btn-prev">&#8592; Prev</button>
  <span id="pager-info"></span>
  <button id="btn-next">Next &#8594;</button>
</div>

<script>
const state = { page: 1, perPage: 50, label: "", clipId: "", total: 0 };

const $ = id => document.getElementById(id);

const observer = new IntersectionObserver(entries => {
  for (const e of entries) {
    if (e.isIntersecting) {
      const img = e.target;
      if (img.dataset.src) { img.src = img.dataset.src; observer.unobserve(img); }
    }
  }
}, { rootMargin: "300px" });

function cropUrl(item, full = false) {
  const b = item.bbox_xyxy.join(",");
  let u = `/api/crop?clip_id=${encodeURIComponent(item.clip_id)}`
    + `&frame_index=${item.frame_index}&bbox=${encodeURIComponent(b)}`;
  if (full) u += "&full=1";
  return u;
}

function shortId(id) {
  return id.length > 28 ? "…" + id.slice(-27) : id;
}

function renderTiles(items) {
  const grid = $("grid");
  grid.innerHTML = "";
  for (const it of items) {
    const tile = document.createElement("div");
    tile.className = "tile";

    const img = document.createElement("img");
    img.alt = it.label;
    img.dataset.src = cropUrl(it);
    observer.observe(img);

    const meta = document.createElement("div");
    meta.className = "tile-meta";
    meta.innerHTML =
      `<div class="t-label">${escHtml(it.label)}</div>` +
      `<div class="t-clip" title="${escHtml(it.clip_id)}">${escHtml(shortId(it.clip_id))}</div>` +
      `<div class="t-frame">frame ${it.frame_index}${it.camera ? " &middot; " + escHtml(it.camera) : ""}</div>`;

    tile.appendChild(img);
    tile.appendChild(meta);
    tile.addEventListener("click", () => openDetail(it));
    grid.appendChild(tile);
  }
}

function openDetail(item) {
  const params = new URLSearchParams({
    clip_id: item.clip_id,
    frame_index: item.frame_index,
    bbox: item.bbox_xyxy.join(","),
    label: item.label || "",
    camera: item.camera || "",
  });
  window.open("/clip?" + params.toString(), "_blank", "noopener");
}

function escHtml(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

async function loadItems() {
  const params = new URLSearchParams({
    page: state.page,
    per_page: state.perPage,
  });
  if (state.label)  params.set("label", state.label);
  if (state.clipId) params.set("clip_id", state.clipId);

  const resp = await fetch("/api/items?" + params);
  const data = await resp.json();
  state.total = data.total;

  $("count").textContent = `${data.total.toLocaleString()} items`;
  renderTiles(data.items);

  const pages = Math.max(1, Math.ceil(data.total / state.perPage));
  $("pager-info").textContent = `Page ${state.page} / ${pages}`;
  $("btn-prev").disabled = state.page <= 1;
  $("btn-next").disabled = state.page >= pages;

  const filtered = Boolean(state.label || state.clipId);
  const btn = $("btn-download");
  btn.disabled = !filtered;
  btn.title = filtered
    ? "Download matching clip_ids grouped by label"
    : "Apply a label or clip_id filter first";
}

async function downloadClipIds() {
  if (!state.label && !state.clipId) return;
  const params = new URLSearchParams();
  if (state.label)  params.set("label", state.label);
  if (state.clipId) params.set("clip_id", state.clipId);

  const resp = await fetch("/api/clip_ids?" + params);
  if (!resp.ok) {
    alert("Download failed: " + resp.status);
    return;
  }
  const blob = await resp.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  const suffix = state.label
    ? state.label.replace(/[^a-z0-9_-]+/gi, "_")
    : state.clipId.replace(/[^a-z0-9_-]+/gi, "_");
  a.download = `clip_ids_${suffix}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

async function init() {
  const resp = await fetch("/api/labels");
  const labels = await resp.json();
  const sel = $("sel-label");
  for (const lbl of labels) {
    const opt = document.createElement("option");
    opt.value = lbl;
    opt.textContent = lbl;
    sel.appendChild(opt);
  }
  loadItems();
}

// ---- event listeners ----
$("btn-apply").addEventListener("click", () => {
  state.page = 1;
  state.label = $("sel-label").value;
  state.clipId = $("inp-clip").value.trim();
  state.perPage = parseInt($("sel-pp").value);
  loadItems();
});

$("inp-clip").addEventListener("keydown", e => {
  if (e.key === "Enter") $("btn-apply").click();
});

$("btn-prev").addEventListener("click", () => { state.page--; loadItems(); });
$("btn-next").addEventListener("click", () => { state.page++; loadItems(); });
$("btn-download").addEventListener("click", downloadClipIds);

init();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# S3 pkl download
# ---------------------------------------------------------------------------
def update_clip_to_key(items, stem_to_key, clip_to_key):
    """Extend clip_to_key with any newly-seen clip_ids that match stem_to_key."""
    if clip_to_key is None or stem_to_key is None:
        return
    for it in items:
        cid = it["clip_id"]
        if cid not in clip_to_key and cid in stem_to_key:
            clip_to_key[cid] = stem_to_key[cid]


def load_s3_pkls(s3_uri, store, profile, endpoint=None,
                 max_pkls=8, initial_pkls=10,
                 stem_to_key=None, clip_to_key=None):
    """Load pkl files from S3; first initial_pkls inline, rest on a daemon thread."""
    path = s3_uri[len("s3://"):]
    bucket, _, prefix = path.partition("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    sess = boto3.Session(profile_name=profile, region_name="us-east-1")
    client_kwargs = {
        "config": Config(
            max_pool_connections=16,
            read_timeout=60,
            connect_timeout=10,
        ),
    }
    if endpoint:
        client_kwargs["endpoint_url"] = endpoint
    client = sess.client("s3", **client_kwargs)

    print(f"Listing pkls under s3://{bucket}/{prefix}",
          flush=True)
    pkl_keys = []
    kwargs = {"Bucket": bucket, "Prefix": prefix}
    while len(pkl_keys) < max_pkls:
        resp = client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            if obj["Key"].endswith(".pkl"):
                pkl_keys.append(obj["Key"])
                if len(pkl_keys) >= max_pkls:
                    break
        if not resp.get("IsTruncated"):
            break
        kwargs["ContinuationToken"] = resp["NextContinuationToken"]

    n_inline = min(initial_pkls, len(pkl_keys))
    print(
        f"Loading {len(pkl_keys)} pkl files "
        f"(first {n_inline} inline, {len(pkl_keys) - n_inline} in background)...",
        flush=True,
    )

    def load_one(key):
        resp = client.get_object(Bucket=bucket, Key=key)
        data = pickle.load(io.BytesIO(resp["Body"].read()))
        items = data.get("items", [])
        store.extend(items)
        update_clip_to_key(items, stem_to_key, clip_to_key)
        print(f"  {key} ({len(store._items)} items)", flush=True)

    for key in pkl_keys[:n_inline]:
        load_one(key)

    remaining = pkl_keys[n_inline:]
    if remaining:
        def bg():
            for key in remaining:
                try:
                    load_one(key)
                except Exception as e:
                    print(f"Warning: skipping {key}: {e}",
                          file=sys.stderr)
            print(
                f"Background pkl loading complete "
                f"({len(store._items)} items total).",
                flush=True,
            )
        threading.Thread(target=bg, daemon=True).start()


def load_local_pkls(pkl_dir, store, initial_pkls=10,
                    stem_to_key=None, clip_to_key=None):
    """Load local pkl files; first initial_pkls inline, rest on a daemon thread."""
    pkl_paths = sorted(Path(pkl_dir).glob("*.pkl"))
    n_inline = min(initial_pkls, len(pkl_paths))
    print(
        f"Loading {len(pkl_paths)} pkl files "
        f"(first {n_inline} inline, {len(pkl_paths) - n_inline} in background)...",
        flush=True,
    )

    def load_one(pkl_p):
        try:
            with open(pkl_p, "rb") as f:
                data = pickle.load(f)
            items = data.get("items", [])
            store.extend(items)
            update_clip_to_key(items, stem_to_key, clip_to_key)
        except Exception as e:
            print(f"Warning: skipping {pkl_p.name}: {e}",
                  file=sys.stderr)

    for pkl_p in pkl_paths[:n_inline]:
        load_one(pkl_p)
    store._print_status()

    remaining = pkl_paths[n_inline:]
    if remaining:
        def bg():
            for pkl_p in remaining:
                load_one(pkl_p)
            print(
                f"Background pkl loading complete "
                f"({len(store._items)} items total).",
                flush=True,
            )
            store._print_status()
        threading.Thread(target=bg, daemon=True).start()


def _build_clip_to_key(video_paths_file, clip_ids):
    """Build a clip_id -> S3 key mapping from a video paths txt file."""
    with open(video_paths_file, "r") as f:
        keys = [line.strip().lstrip("/") for line in f if line.strip()]

    # Index all stems for fast lookup
    stem_to_key = {}
    for key in keys:
        if key.endswith(".mp4"):
            stem_to_key[Path(key).stem] = key

    clip_to_key = {}
    for cid in clip_ids:
        if cid in stem_to_key:
            clip_to_key[cid] = stem_to_key[cid]

    print(
        f"Mapped {len(clip_to_key)}/{len(clip_ids)} clip_ids "
        f"to video keys",
        flush=True,
    )
    return clip_to_key


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Browse Florence2+SigCLIP crop embeddings in the browser."
    )
    parser.add_argument(
        "pkl_dir",
        help=(
            "Local directory or S3 URI (s3://bucket/prefix/) "
            "containing *.pkl shard files."
        ),
    )
    parser.add_argument(
        "--bucket", required=True, action="append",
        help="S3 bucket for videos (repeatable, one per source).",
    )
    parser.add_argument(
        "--profile", action="append",
        help="AWS profile for videos (repeatable, default: sil-wheel).",
    )
    parser.add_argument(
        "--endpoint", action="append",
        help="S3 endpoint for videos (repeatable, default: https://s3.example.com).",
    )
    parser.add_argument(
        "--pkl-profile",
        default="sil-wheel",
        dest="pkl_profile",
        help="AWS profile for pkl S3 (default: sil-wheel).",
    )
    parser.add_argument(
        "--pkl-endpoint",
        default=None,
        dest="pkl_endpoint",
        help="S3 endpoint for pkl files (default: standard AWS).",
    )
    parser.add_argument(
        "--video-paths",
        action="append",
        dest="video_paths",
        help=(
            "Path to a .txt file listing S3 video keys "
            "(repeatable, one per source). "
            "Matched by position with --bucket/--profile/--endpoint."
        ),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--video-cache-size",
        type=int,
        default=10,
        dest="video_cache_size",
        help="Max videos to hold in memory (default 10).",
    )
    parser.add_argument(
        "--frame-cache-size",
        type=int,
        default=200,
        dest="frame_cache_size",
        help="Max decoded frames to hold in memory (default 200).",
    )
    parser.add_argument(
        "--max-pkls",
        type=int,
        default=8,
        dest="max_pkls",
        help="Max pkl files to load from S3 (default 8).",
    )
    parser.add_argument(
        "--initial-pkls",
        type=int,
        default=10,
        dest="initial_pkls",
        help=(
            "Pkls loaded before the server starts; the rest load "
            "in the background (default 10)."
        ),
    )
    parser.add_argument(
        "--wheel-url",
        default="http://localhost:8000",
        dest="wheel_url",
        help=(
            "Base URL of the SIL-Wheel server. "
            "A per-clip link is shown on the detail page "
            "(default http://localhost:8000). "
            "Pass an empty string to disable."
        ),
    )
    args = parser.parse_args()

    buckets = args.bucket
    profiles = args.profile or []
    endpoints = args.endpoint or []
    video_paths = args.video_paths or []
    n_sources = len(buckets)
    if len(profiles) != n_sources or len(endpoints) != n_sources:
        parser.error(
            f"--bucket, --profile, and --endpoint must all appear "
            f"the same number of times (got {n_sources}, "
            f"{len(profiles)}, {len(endpoints)})"
        )
    if video_paths and len(video_paths) != n_sources:
        parser.error(
            f"--video-paths must appear once per source "
            f"(got {len(video_paths)}, expected {n_sources})"
        )

    pkl_path = args.pkl_dir
    store = PklStore()

    stem_to_key = None
    clip_to_key = None
    if video_paths:
        stem_to_key = {}
        clip_to_key = {}
        for src_idx, vp in enumerate(video_paths):
            with open(vp, "r") as f:
                keys = [
                    l.strip().lstrip("/") for l in f if l.strip()
                ]
            for k in keys:
                if k.endswith(".mp4"):
                    stem = Path(k).stem
                    entry = (k, src_idx)
                    stem_to_key[stem] = entry
                    # Also index by UUID prefix for filenames like
                    # UUID.camera_front_wide_120fov.mp4
                    prefix = stem.split(".")[0]
                    if prefix != stem:
                        stem_to_key.setdefault(prefix, entry)

    if pkl_path.startswith("s3://"):
        load_s3_pkls(
            pkl_path,
            store,
            args.pkl_profile,
            args.pkl_endpoint,
            max_pkls=args.max_pkls,
            initial_pkls=args.initial_pkls,
            stem_to_key=stem_to_key,
            clip_to_key=clip_to_key,
        )
    else:
        load_local_pkls(
            pkl_path,
            store,
            initial_pkls=args.initial_pkls,
            stem_to_key=stem_to_key,
            clip_to_key=clip_to_key,
        )

    if video_paths:
        initial_clip_ids = {it["clip_id"] for it in store._items}
        print(
            f"Mapped {len(clip_to_key)}/{len(initial_clip_ids)} "
            f"clip_ids to video keys "
            f"(updates as background loading continues)",
            flush=True,
        )

    sources = list(zip(buckets, profiles, endpoints))
    cache = S3VideoCache(
        sources=sources,
        video_cache_size=args.video_cache_size,
        frame_cache_size=args.frame_cache_size,
        clip_to_key=clip_to_key,
    )
    handler = partial(
        CropServer, store=store, cache=cache,
        wheel_url=(args.wheel_url or None),
    )

    with _ThreadingHTTPServer((args.host, args.port), handler) as httpd:
        print(f"Crop viewer running at http://{args.host}:{args.port}/", flush=True)
        print("Press Ctrl-C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
