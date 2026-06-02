// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Single state object for the currently-open clustering run; null when
// no run is open. All reads/writes funnel through this so the load /
// close lifecycle has one source of truth instead of seven parallel
// globals. Shape:
//   {
//     data: { clusters, umap, topics },     // server response
//     selectedCids: number[],                // highlighted on overview
//     zoomedCid: null | number,             // current beeswarm view
//     zoomedMembers: null | { clip_ids, distances, cluster_id },
//     centroidPositions: [],                // overview hit-test cache
//     resizeObserver: ResizeObserver | null,
//   }
window._clusteringRun = null;

function renderClusteringRuns(runs) {
    const sel = document.getElementById("clustering-runs-select");
    const runBtn = document.getElementById("run-clustering-button");
    const viewBtn = document.getElementById("view-clustering-run-button");
    const delBtn = document.getElementById("delete-clustering-run-button");
    if (!sel) return;

    const prevSelected = sel.value;
    sel.innerHTML = "";

    if (runs.length === 0) {
        sel.innerHTML = '<option disabled value="">No runs yet</option>';
        if (viewBtn) viewBtn.disabled = true;
        if (delBtn) delBtn.disabled = true;
        return;
    }

    let pendingCount = 0;
    runs.forEach(run => {
        const opt = document.createElement("option");
        opt.value = run.run_id;
        opt.dataset.embedType = run.embed_type || "cosmos";
        const icon = run.status === "done" ? "✓" : run.status === "pending" ? "⏳" : "✗";
        const embedLabel = embedTypeLabel(run.embed_type);
        // Render search filters without the always-noisy `page` param.
        const searchFull = parseSearchParams(run.search_params)
            .map(([k, v]) => `${k}=${v}`).join("&");
        const searchShort = searchFull ? ` · ${searchFull.slice(0, 40)}…` : "";
        opt.textContent = `${icon} #Clusters: ${run.n_clusters} · ${embedLabel} · ${run.run_id} · ${run.n_clips.toLocaleString()} clips${searchShort}`;
        if (searchFull) {
            opt.title = searchFull;
        }
        opt.disabled = run.status !== "done";
        if (run.run_id === prevSelected) opt.selected = true;
        sel.appendChild(opt);
        if (run.status === "pending") pendingCount++;
    });

    if (runBtn && !runBtn.dataset.locked) runBtn.disabled = pendingCount >= 3;
    if (viewBtn) viewBtn.disabled = !sel.value || sel.options[sel.selectedIndex]?.disabled;
    if (delBtn) delBtn.disabled = !sel.value;

    sel.onchange = function () {
        if (viewBtn) viewBtn.disabled = !sel.value || sel.options[sel.selectedIndex]?.disabled;
        if (delBtn) delBtn.disabled = !sel.value;
        const embedSel = document.getElementById("cluster-embed-type");
        const embedType = sel.options[sel.selectedIndex]?.dataset.embedType;
        if (embedSel && embedType) embedSel.value = embedType;
    };
}

function deleteClusteringRun() {
    const sel = document.getElementById("clustering-runs-select");
    if (!sel || !sel.value) return;
    const run_id = sel.value;
    fetch("", {
        method: "POST",
        headers: { "Content-Type": "text/plain" },
        body: `delete_clustering::${run_id}`,
    })
        .then(r => r.json())
        .then(() => checkClusteringState())
        .catch(e => console.error("Failed to delete clustering run", e));
}

function viewSelectedClusteringRun() {
    const sel = document.getElementById("clustering-runs-select");
    if (!sel || !sel.value) return;
    // Re-issue the search after the panel loads so the per-clip
    // cluster_membership enrichment fires server-side; the result grid
    // then renders distance + the "Show Cluster" button on every card.
    loadClusteringResults(sel.value).then(() => search());
}

// Parse a clustering run's `search_params` string ("&page=0&data_source=Foo&
// project_source=Bar") into [["data_source","Foo"], ["project_source","Bar"]].
// `page` is always pagination noise from the source URL; drop it.
function parseSearchParams(s) {
    if (!s) return [];
    let qs = s;
    while (qs.startsWith("&") || qs.startsWith("?")) qs = qs.slice(1);
    if (!qs) return [];
    const params = new URLSearchParams(qs);
    const out = [];
    for (const [k, v] of params.entries()) {
        if (k === "page") continue;
        out.push([k, v]);
    }
    return out;
}

function renderClusteringBanner(banner, run_id, data) {
    const meta = data.metadata || {};
    const tmeta = data.topics_meta || {};
    const embedLabel = embedTypeLabel(meta.embed_type);
    const searchPairs = parseSearchParams(meta.search_params);

    banner.replaceChildren();

    // Compact one-line summary + a [▾] disclosure toggle that reveals the
    // detail block. Clicking anywhere on the summary row toggles it.
    const summaryRow = document.createElement("div");
    summaryRow.className = "clustering-run-banner-summary";
    summaryRow.setAttribute("role", "button");
    summaryRow.setAttribute("tabindex", "0");
    summaryRow.setAttribute("aria-expanded", "false");

    const summaryText = document.createElement("span");
    summaryText.className = "clustering-run-banner-summary-text";
    const captionTail = tmeta.caption_model ? ` · captions: ${tmeta.caption_model}` : "";
    summaryText.textContent = (
        `Run ${run_id} · ${embedLabel} ·`
        + ` ${meta.n_clusters} clusters ·`
        + ` ${(meta.n_input_clips || 0).toLocaleString()} clips`
        + captionTail
    );
    const caret = document.createElement("span");
    caret.className = "clustering-run-banner-caret";
    caret.textContent = "▾";
    summaryRow.appendChild(summaryText);
    summaryRow.appendChild(caret);
    banner.appendChild(summaryRow);

    const details = document.createElement("div");
    details.className = "clustering-run-banner-details";
    details.hidden = true;

    if (searchPairs.length > 0) {
        const searchBlock = document.createElement("div");
        searchBlock.className = "clustering-run-banner-search";
        const searchLabel = document.createElement("div");
        searchLabel.className = "clustering-run-banner-section-label";
        searchLabel.textContent = "Search filters";
        searchBlock.appendChild(searchLabel);
        const dl = document.createElement("dl");
        for (const [k, v] of searchPairs) {
            const dt = document.createElement("dt");
            dt.textContent = k;
            const dd = document.createElement("dd");
            dd.textContent = v;
            dl.appendChild(dt);
            dl.appendChild(dd);
        }
        searchBlock.appendChild(dl);
        details.appendChild(searchBlock);
    }

    if (tmeta.caption_model) {
        const topicRow = document.createElement("div");
        topicRow.className = "clustering-run-topics-meta";
        const label = document.createTextNode("Topics Extracted from Captions of ");
        const code = document.createElement("code");
        code.textContent = tmeta.caption_model;
        topicRow.appendChild(label);
        topicRow.appendChild(code);
        if (typeof tmeta.captions_found === "number"
            && typeof tmeta.captions_total === "number") {
            const pct = tmeta.captions_total > 0
                ? Math.round(100 * tmeta.captions_found / tmeta.captions_total)
                : 0;
            topicRow.appendChild(document.createTextNode(
                ` (${tmeta.captions_found.toLocaleString()} / `
                + `${tmeta.captions_total.toLocaleString()} clips, ${pct}%)`
            ));
        }
        topicRow.title = (
            "Caption model auto-selected as the one with the "
            + "most coverage of the run's clip ids."
        );
        details.appendChild(topicRow);
    }

    banner.appendChild(details);

    function toggle() {
        const isOpen = !details.hidden;
        details.hidden = isOpen;
        summaryRow.setAttribute("aria-expanded", String(!isOpen));
        caret.textContent = isOpen ? "▾" : "▴";
    }
    summaryRow.addEventListener("click", toggle);
    summaryRow.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            toggle();
        }
    });
}

function renderClusterTopics(container, topicInfo) {
    const keywords = topicInfo && Array.isArray(topicInfo.keywords)
        ? topicInfo.keywords : [];
    if (keywords.length === 0) {
        container.textContent = "";
        container.style.display = "none";
        return;
    }
    container.textContent = keywords.join(", ");
    container.style.display = "";
}

// Apply cluster selection UI state without triggering a new search.
// Used both by showClusterSection (interactive) and loadClusteringResults (restore).
// Accepts a single cluster id or an array of ids; multi-cluster selections
// just light up multiple centroids and show a count summary in the info panel.
function applyClusterSelection(cidOrIds) {
    const run = window._clusteringRun;
    if (!run) return;
    const clusters = run.data.clusters;
    const ids = (Array.isArray(cidOrIds) ? cidOrIds : [cidOrIds])
        .map(c => parseInt(c))
        .filter(c => clusters[String(c)]);
    if (ids.length === 0) return;
    run.selectedCids = ids;
    renderUMAPPlot();
    const panel = document.getElementById("cluster-info-panel");
    const text = document.getElementById("cluster-info-text");
    if (panel) panel.style.display = "flex";

    const kwEl = document.getElementById("cluster-keywords");
    const zoomBtn = document.getElementById("zoom-into-cluster-button");
    const addBtn = document.getElementById("add-nearest-clusters-button");
    const isSingle = ids.length === 1;
    const singleAndZoomable = isSingle && !window.disableClusterZoom;
    if (ids.length === 1) {
        const cid = ids[0];
        const size = (clusters[String(cid)] || {}).cluster_size || 0;
        const topicInfo = (run.data.topics || {})[String(cid)];
        let label = `Cluster ${cid} — ${size} clips`;
        if (topicInfo && topicInfo.description) {
            label += ` — ${topicInfo.description}`;
        } else if (topicInfo && topicInfo.keywords && topicInfo.keywords.length > 0) {
            label += ` — ${topicInfo.keywords.slice(0, 3).join(", ")}`;
        }
        if (text) text.textContent = label;
        if (kwEl) renderClusterTopics(kwEl, topicInfo);
    } else {
        const total = ids.reduce(
            (s, cid) => s + ((clusters[String(cid)] || {}).cluster_size || 0), 0,
        );
        if (text) text.textContent = `${ids.length} clusters — ${total} clips`;
        if (kwEl) { kwEl.textContent = ""; kwEl.style.display = "none"; }
    }
    if (zoomBtn) zoomBtn.style.display = singleAndZoomable ? "" : "none";
    if (addBtn) addBtn.style.display = isSingle ? "" : "none";
    window.currentClusterSearch.cluster_ids = ids.map(String);
    renderBreadcrumb();
}

// Add the K nearest clusters in UMAP space to the current selection.
// Distance is measured between centroids in the visible UMAP plot — what
// the user is looking at — rather than in the original embedding space.
function addNearestClusters(k = 5) {
    const run = window._clusteringRun;
    if (!run || (run.selectedCids || []).length !== 1) return;
    const seedCid = run.selectedCids[0];
    const centroids = run.data.umap?.centroids || {};
    const seed = centroids[String(seedCid)];
    if (!seed) return;
    const ranked = [];
    for (const [cid, xy] of Object.entries(centroids)) {
        const id = parseInt(cid);
        if (id === seedCid) continue;
        const dx = xy[0] - seed[0];
        const dy = xy[1] - seed[1];
        ranked.push({ id, d2: dx * dx + dy * dy });
    }
    ranked.sort((a, b) => a.d2 - b.d2);
    const additions = ranked.slice(0, k).map(r => r.id);
    applyClusterSelection([seedCid, ...additions]);
    search();
}

// Explicit "Zoom in →" affordance from the cluster info panel — only
// available when exactly one cluster is selected.
function zoomIntoSelectedCluster() {
    const run = window._clusteringRun;
    if (!run || window.disableClusterZoom) return;
    if ((run.selectedCids || []).length !== 1) return;
    const cid = run.selectedCids[0];
    setZoom(cid);
    fetchClusterMembers(cid);
}

function loadClusteringResults(run_id, restoreClusterIds = null, restoreZoom = false) {
    // Normalize: scalar -> [scalar], anything truthy stays a list.
    const restoreList = restoreClusterIds == null
        ? []
        : (Array.isArray(restoreClusterIds) ? restoreClusterIds : [restoreClusterIds])
            .filter(v => v !== null && v !== undefined && v !== "");
    const applyRestore = () => {
        if (restoreList.length === 0) return;
        applyClusterSelection(restoreList.map(c => parseInt(c)));
        if (restoreZoom && restoreList.length === 1) {
            const cid = parseInt(restoreList[0]);
            setZoom(cid);
            fetchClusterMembers(cid);
        }
    };
    // Idempotent: if the panel is already populated for this run, just
    // restore the requested cluster selection (if any) and return.
    if (
        window.currentClusterSearch.run_id === run_id &&
        window._clusteringRun
    ) {
        applyRestore();
        return Promise.resolve();
    }
    showLoading("Loading clustering results…");
    return fetch(`/clustering_results?run_id=${encodeURIComponent(run_id)}`)
        .then(r => {
            if (!r.ok) {
                // Run was deleted or never completed — fall back to the runs panel.
                hideLoading();
                window.currentClusterSearch = { run_id: null, cluster_ids: [] };
                closeClusteringResultsUI();
                checkClusteringState();
                return null;
            }
            return r.json();
        })
        .then(data => {
            if (!data) return;
            window._clusteringRun = {
                data: {
                    clusters: data.clusters || {},
                    umap: data.umap || { centroids: {}, reps: {} },
                    topics: data.topics || {},
                },
                selectedCids: [],
                zoomedCid: null,
                zoomedMembers: null,
                centroidPositions: [],
                resizeObserver: null,
            };
            window.currentClusterSearch = { run_id: run_id, cluster_ids: [] };
            document.getElementById("cluster-embed-type").value = data.metadata.embed_type;

            const banner = document.getElementById("clustering-run-banner");
            if (banner) {
                renderClusteringBanner(banner, run_id, data);
            }

            const warn = document.getElementById("clustering-subset-warning");
            if (warn) warn.style.display = "none";

            setViewingRun(true);
            renderBreadcrumb();
            hideLoading();

            // Defer render until after the browser has laid out the newly-visible
            // canvas, so clientWidth reflects the actual CSS display width.
            requestAnimationFrame(() => {
                renderUMAPPlot();
                applyRestore();

                // Redraw when the container is resized (e.g. window resize or sidebar toggle)
                const canvas = document.getElementById("umap-canvas");
                if (canvas && window.ResizeObserver && window._clusteringRun) {
                    if (window._clusteringRun.resizeObserver) {
                        window._clusteringRun.resizeObserver.disconnect();
                    }
                    window._clusteringRun.resizeObserver = new ResizeObserver(() => {
                        renderUMAPPlot();
                    });
                    window._clusteringRun.resizeObserver.observe(canvas);
                }
            });
        })
        .catch(e => {
            hideLoading();
            console.error("Failed to load clustering results", e);
        });
}

// Top-level dispatcher: sizes the canvas, then hands off to the right
// view. The two views (zoomed beeswarm vs full overview) share nothing
// meaningful — keep them as separate, navigable functions.
function renderUMAPPlot() {
    const canvas = document.getElementById("umap-canvas");
    if (!canvas) return;

    // Match the canvas internal resolution to its displayed CSS size.
    const displayW = canvas.clientWidth || 700;
    const displayH = Math.round(displayW * (5 / 7));
    if (canvas.width !== displayW || canvas.height !== displayH) {
        canvas.width = displayW;
        canvas.height = displayH;
    }
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const run = window._clusteringRun;
    const tsne = run?.data?.umap || { centroids: {}, reps: {} };
    if (Object.keys(tsne.centroids || {}).length === 0) {
        drawCenterMsg(ctx, "No t-SNE data available", H);
        return;
    }
    if (run.zoomedCid != null) {
        renderZoomedView(ctx, canvas, W, H);
    } else {
        renderClusterOverview(ctx, canvas, W, H, tsne);
    }
}

// Zoom mode: beeswarm of one cluster's clips, x-coded by distance to
// centroid. Members fetched on demand via fetchClusterMembers; until
// they arrive, draw a placeholder.
function renderZoomedView(ctx, canvas, W, H) {
    const run = window._clusteringRun;
    const zoomedCid = run.zoomedCid;
    const members = run.zoomedMembers;
    const ready = members && members.cluster_id === zoomedCid;
    const cluster_clip_ids = ready ? members.clip_ids : [];
    const cluster_distances = ready ? members.distances : [];
    const dotPositions = [];
    let hitRadius = 8;
    const padZ = 30;

    if (!ready) {
        drawCenterMsg(ctx, `Loading cluster ${zoomedCid}…`, H);
    } else if (cluster_clip_ids.length === 0) {
        drawCenterMsg(ctx, `No clips in cluster ${zoomedCid}`, H);
    } else {
        // Beeswarm: x = distance to centroid, y = jitter that bumps
        // overlapping dots away from the midline. Clips arrive sorted
        // by distance, so x is monotonic and the collision check can
        // early-break on x-distance.
        const n = cluster_clip_ids.length;
        const dotR = 3;
        const step = 2 * dotR + 1;
        const xRange = W - 2 * padZ;
        const yMid = H / 2;
        const yMaxDelta = H / 2 - padZ;
        const minD = cluster_distances[0];
        const maxD = cluster_distances[n - 1];
        hitRadius = dotR + 5;

        const distToX = (d) => maxD > minD
            ? padZ + ((d - minD) / (maxD - minD)) * xRange
            : W / 2;

        const range = window.currentClusterDistanceRange || { lo: 0, hi: 100 };
        const loIdx = Math.floor(n * range.lo / 100);
        const hiIdx = Math.ceil(n * range.hi / 100);

        const placed = [];
        const collisionD2 = step * step;
        const maxK = Math.floor(yMaxDelta / step);
        for (let i = 0; i < n; i++) {
            const d = cluster_distances[i];
            const x = distToX(d);
            let y = yMid;
            outer: for (let k = 0; k <= maxK; k++) {
                const dirs = k === 0 ? [0] : [-1, 1];
                for (const dir of dirs) {
                    const cy = yMid + dir * k * step;
                    let collides = false;
                    for (let j = placed.length - 1; j >= 0; j--) {
                        const p = placed[j];
                        const dx = x - p.x;
                        if (dx > step) break;
                        const dy = cy - p.y;
                        if (dx * dx + dy * dy < collisionD2) {
                            collides = true; break;
                        }
                    }
                    if (!collides) { y = cy; break outer; }
                }
            }
            placed.push({ x, y });

            const t = maxD > minD ? (d - minD) / (maxD - minD) : 0;
            const beyond = i < loIdx || i >= hiIdx;
            ctx.globalAlpha = beyond ? 0.25 : 0.9;
            // Sequential cool→warm: close (blue) → far (orange).
            const hue = Math.round(220 - 200 * t);
            const lightness = Math.round(55 - 10 * t);
            ctx.fillStyle = `hsl(${hue},80%,${lightness}%)`;
            if (!beyond) {
                dotPositions.push({ cx: x, cy: y, clip_id: cluster_clip_ids[i] });
            }
            ctx.beginPath();
            ctx.arc(x, y, dotR, 0, 2 * Math.PI);
            ctx.fill();
            ctx.strokeStyle = "#fff";
            ctx.lineWidth = 0.8;
            ctx.stroke();
        }
        ctx.globalAlpha = 1.0;

        // x-axis baseline + min/max distance labels for context.
        ctx.strokeStyle = "#ccc";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padZ, H - padZ / 2);
        ctx.lineTo(W - padZ, H - padZ / 2);
        ctx.stroke();
        ctx.fillStyle = "#666";
        ctx.font = "11px sans-serif";
        ctx.fillText(`d = ${minD.toFixed(3)}`, padZ, H - 6);
        const farLabel = `d = ${maxD.toFixed(3)}`;
        const farW = ctx.measureText(farLabel).width;
        ctx.fillText(farLabel, W - padZ - farW, H - 6);
    }

    const tooltip = document.getElementById("umap-tooltip");
    const tooltipText = tooltip?.querySelector(".umap-tooltip-text");
    const tooltipVideo = tooltip?.querySelector(".umap-tooltip-video");
    const pickHit = (e) => {
        const h = pickNearest(canvas, e, dotPositions, () => hitRadius);
        return (h && h.clip_id) ? h : null;
    };
    const clearZoom = () => { canvas.style.cursor = ""; clearTooltip(); };

    // Box-select state local to this render. Coordinates are kept in
    // canvas-CSS-pixel space so the overlay (anchored inside
    // .umap-canvas-wrap) tracks the canvas through page scroll.
    let boxStart = null;
    const boxOverlay = () => {
        let el = document.getElementById("cluster-box-select-overlay");
        if (!el) {
            el = document.createElement("div");
            el.id = "cluster-box-select-overlay";
            (canvas.parentElement || document.body).appendChild(el);
        }
        return el;
    };
    const canvasCoords = (e) => {
        const r = canvas.getBoundingClientRect();
        return { x: e.clientX - r.left, y: e.clientY - r.top };
    };

    canvas.onmousedown = function (e) {
        if (window._clusterSelectMode === "box") {
            boxStart = canvasCoords(e);
        }
    };
    canvas.onmousemove = function (e) {
        if (window._clusterSelectMode === "box") {
            if (!boxStart) return;
            const c = canvasCoords(e);
            const el = boxOverlay();
            el.style.display = "block";
            el.style.left = Math.min(boxStart.x, c.x) + "px";
            el.style.top = Math.min(boxStart.y, c.y) + "px";
            el.style.width = Math.abs(c.x - boxStart.x) + "px";
            el.style.height = Math.abs(c.y - boxStart.y) + "px";
            return;
        }
        if (!tooltip) return;
        const hit = pickHit(e);
        if (hit) {
            tooltip.style.display = "block";
            tooltip.style.left = (e.clientX + 12) + "px";
            tooltip.style.top = (e.clientY - 10) + "px";
            canvas.style.cursor = "pointer";
            if (tooltipText) {
                tooltipText.textContent = `${hit.clip_id}  ·  click to copy`;
            }
            if (window._umapPreviewKey !== hit.clip_id) {
                window._umapPreviewKey = hit.clip_id;
                if (tooltipVideo) {
                    tooltipVideo.src = `/video/${hit.clip_id}.mp4`;
                    tooltipVideo.style.display = "block";
                    tooltipVideo.play().catch(() => { });
                }
            }
        } else {
            clearZoom();
        }
    };
    canvas.onmouseleave = function () {
        if (window._clusterSelectMode === "box") {
            // Only cancel the in-progress drag; leave any committed box visible.
            if (boxStart) {
                boxStart = null;
                const el = document.getElementById("cluster-box-select-overlay");
                if (el) el.style.display = "none";
            }
        } else {
            clearZoom();
        }
    };
    canvas.onmouseup = function (e) {
        if (window._clusterSelectMode !== "box" || !boxStart) return;
        const rect = canvas.getBoundingClientRect();
        const W2 = canvas.width;
        const xRange = W2 - 2 * padZ;
        const c = canvasCoords(e);
        const px1 = Math.min(boxStart.x, c.x) * (W2 / rect.width);
        const px2 = Math.max(boxStart.x, c.x) * (W2 / rect.width);
        boxStart = null;
        // Count which clips' x-positions actually fall inside the box,
        // then convert their indices to clip-percentile. Pixel-fraction
        // would mis-map when the distance distribution is non-uniform.
        const n = cluster_distances.length;
        let firstIdx = -1, lastIdx = -1;
        if (n > 0) {
            const minD = cluster_distances[0];
            const maxD = cluster_distances[n - 1];
            const distToX = (d) => maxD > minD
                ? padZ + ((d - minD) / (maxD - minD)) * xRange
                : W2 / 2;
            for (let i = 0; i < n; i++) {
                const x = distToX(cluster_distances[i]);
                if (x >= px1 && x <= px2) {
                    if (firstIdx === -1) firstIdx = i;
                    lastIdx = i;
                }
            }
        }
        if (firstIdx === -1) {
            const el = document.getElementById("cluster-box-select-overlay");
            if (el) el.style.display = "none";
            return;
        }
        const lo = (firstIdx / n) * 100;
        const hi = ((lastIdx + 1) / n) * 100;
        setClusterDistanceSliderUI(Math.round(lo), Math.round(hi));
        commitClusterDistanceRange();
    };
    canvas.onclick = async function (e) {
        if (window._clusterSelectMode === "box") return;
        const hit = pickHit(e);
        if (!hit) return;
        const showStatus = (msg) => {
            if (!tooltipText) return;
            tooltipText.textContent = msg;
            setTimeout(() => {
                if (tooltipText && window._umapPreviewKey === hit.clip_id) {
                    tooltipText.textContent = `${hit.clip_id}  ·  click to copy`;
                }
            }, 1200);
        };
        try {
            await navigator.clipboard.writeText(hit.clip_id);
            showStatus(`Copied: ${hit.clip_id}`);
        } catch (err) {
            // Insecure-context fallback (HTTP / non-localhost).
            const temp = document.createElement("input");
            temp.value = hit.clip_id;
            document.body.appendChild(temp);
            temp.select();
            try {
                document.execCommand("copy");
                showStatus(`Copied: ${hit.clip_id}`);
            } catch (e2) {
                showStatus("Copy failed");
            }
            document.body.removeChild(temp);
        }
    };
    clearTooltip();
    canvas.style.cursor = window._clusterSelectMode === "box" ? "crosshair" : "";
}

// Wire the overview canvas's mouse handlers. Click-vs-drag is decided
// by movement past a small CSS-pixel threshold; a click selects (and
// zooms in on a second click of the only-selected centroid), a drag
// box-selects every centroid inside the rectangle.
function bindOverviewHandlers(canvas, run) {
    const tooltip = ensureUmapTooltip();
    const tooltipText = tooltip.querySelector(".umap-tooltip-text");
    const tooltipVideo = tooltip.querySelector(".umap-tooltip-video");
    const pickCentroid = e => pickNearest(canvas, e, run.centroidPositions, d => d.r + 5);
    const DRAG_THRESHOLD = 5;
    let dragStart = null;
    let isDragging = false;
    const canvasCoords = (e) => {
        const r = canvas.getBoundingClientRect();
        return { x: e.clientX - r.left, y: e.clientY - r.top };
    };
    const boxOverlay = () => {
        let el = document.getElementById("cluster-box-select-overlay");
        if (!el) {
            el = document.createElement("div");
            el.id = "cluster-box-select-overlay";
            (canvas.parentElement || document.body).appendChild(el);
        }
        return el;
    };
    const hideBox = () => {
        const el = document.getElementById("cluster-box-select-overlay");
        if (el) el.style.display = "none";
    };

    canvas.onmousedown = (e) => {
        dragStart = canvasCoords(e);
        isDragging = false;
    };
    canvas.onmousemove = (e) => {
        if (dragStart) {
            const c = canvasCoords(e);
            if (Math.abs(c.x - dragStart.x) > DRAG_THRESHOLD
                || Math.abs(c.y - dragStart.y) > DRAG_THRESHOLD) {
                isDragging = true;
                const el = boxOverlay();
                el.style.display = "block";
                el.style.left = Math.min(dragStart.x, c.x) + "px";
                el.style.top = Math.min(dragStart.y, c.y) + "px";
                el.style.width = Math.abs(c.x - dragStart.x) + "px";
                el.style.height = Math.abs(c.y - dragStart.y) + "px";
                clearTooltip();
                return;
            }
        }
        const nearest = pickCentroid(e);
        if (nearest) {
            tooltip.style.display = "block";
            tooltip.style.left = (e.clientX + 12) + "px";
            tooltip.style.top = (e.clientY - 10) + "px";
            const ti = (run.data.topics || {})[String(nearest.cid)];
            const tipLines = [`Cluster ${nearest.cid} — ${nearest.size} clips`];
            if (ti && ti.description) {
                tipLines.push(ti.description);
            } else if (ti && Array.isArray(ti.keywords) && ti.keywords.length > 0) {
                tipLines.push(ti.keywords.slice(0, 5).join(", "));
            }
            tooltipText.textContent = tipLines.join("\n");
            tooltipText.style.whiteSpace = "pre-line";
            if (window._umapPreviewKey !== nearest.cid) {
                window._umapPreviewKey = nearest.cid;
                if (nearest.repClipId) {
                    tooltipVideo.src = `/video/${nearest.repClipId}.mp4`;
                    tooltipVideo.style.display = "block";
                    tooltipVideo.play().catch(() => { });
                } else {
                    tooltipVideo.pause();
                    tooltipVideo.removeAttribute("src");
                    tooltipVideo.load();
                    tooltipVideo.style.display = "none";
                }
            }
        } else {
            clearTooltip();
        }
    };
    canvas.onmouseleave = () => {
        if (dragStart) {
            dragStart = null;
            isDragging = false;
            hideBox();
        }
        clearTooltip();
    };
    canvas.onmouseup = (e) => {
        const start = dragStart;
        const wasDragging = isDragging;
        dragStart = null;
        isDragging = false;
        if (!start) return;
        if (wasDragging) {
            const c = canvasCoords(e);
            hideBox();
            const x1 = Math.min(start.x, c.x);
            const x2 = Math.max(start.x, c.x);
            const y1 = Math.min(start.y, c.y);
            const y2 = Math.max(start.y, c.y);
            const rect = canvas.getBoundingClientRect();
            const sx = canvas.width / rect.width;
            const sy = canvas.height / rect.height;
            const inside = run.centroidPositions
                .filter(p => p.cx >= x1 * sx && p.cx <= x2 * sx
                          && p.cy >= y1 * sy && p.cy <= y2 * sy)
                .map(p => p.cid);
            if (inside.length === 0) return;
            applyClusterSelection(inside);
            search();
            return;
        }
        const nearest = pickCentroid(e);
        if (!nearest) return;
        const sels = window._clusteringRun?.selectedCids || [];
        const isOnlySelected = sels.length === 1 && sels[0] === nearest.cid;
        if (isOnlySelected && !window.disableClusterZoom) {
            setZoom(nearest.cid);
            fetchClusterMembers(nearest.cid);
        } else {
            showClusterSection(nearest.cid);
        }
    };
    canvas.onclick = null;
}

// Full UMAP scatter: every cluster's clip dots + size-encoded centroid
// markers. Selecting one centroid dims the rest. Click to zoom in.
function renderClusterOverview(ctx, canvas, W, H, tsne) {
    const run = window._clusteringRun;
    const clusters = run.data.clusters;
    const selectedSet = new Set(run.selectedCids || []);
    const centroidKeys = Object.keys(tsne.centroids);

    // Gather all coordinates to compute scale.
    const clips = tsne.clips || {};
    const allX = [], allY = [];
    for (const cid of centroidKeys) {
        const [x, y] = tsne.centroids[cid];
        allX.push(x); allY.push(y);
        for (const [cx2, cy2] of (clips[cid] || [])) {
            allX.push(cx2); allY.push(cy2);
        }
    }
    const minX = Math.min(...allX), maxX = Math.max(...allX);
    const minY = Math.min(...allY), maxY = Math.max(...allY);
    const pad = 30;
    const sx = (W - 2 * pad) / (maxX - minX || 1);
    const sy = (H - 2 * pad) / (maxY - minY || 1);
    const toCanvas = (x, y) => [pad + (x - minX) * sx, pad + (y - minY) * sy];

    const K = centroidKeys.length;
    const palette = centroidKeys.map((_, i) => `hsl(${Math.round(i * 360 / K)},70%,55%)`);

    // Size-encoded centroid radius: scale sqrt(size) into [7, 18].
    const sizes = centroidKeys.map(cid => (clusters[cid] || {}).cluster_size || 0);
    const maxSize = Math.max(...sizes, 1);
    const centroidRadius = size => Math.max(7, Math.min(18, 7 + 11 * Math.sqrt(size / maxSize)));

    const hasSelection = selectedSet.size > 0;

    // Draw clip dots — dim all clusters except the selected ones.
    for (let i = 0; i < centroidKeys.length; i++) {
        const cid = centroidKeys[i];
        const isSelected = selectedSet.has(parseInt(cid));
        ctx.globalAlpha = hasSelection ? (isSelected ? 0.75 : 0.12) : 0.6;
        ctx.fillStyle = palette[i];
        for (const [px, py] of (clips[cid] || [])) {
            const [cx2, cy2] = toCanvas(px, py);
            ctx.beginPath();
            ctx.arc(cx2, cy2, 3, 0, 2 * Math.PI);
            ctx.fill();
        }
    }
    ctx.globalAlpha = 1.0;

    // Draw centroid dots — size-encoded radius, dimmed when not selected.
    const centroidPositions = [];
    for (let i = 0; i < centroidKeys.length; i++) {
        const cid = centroidKeys[i];
        const [x, y] = tsne.centroids[cid];
        const [cx, cy] = toCanvas(x, y);
        const clusterSize = (clusters[cid] || {}).cluster_size || 0;
        const repClipId = (clusters[cid] || {}).representative_clip_id || null;
        const r = centroidRadius(clusterSize);
        centroidPositions.push({ cid: parseInt(cid), cx, cy, size: clusterSize, r, repClipId });

        const isSelected = selectedSet.has(parseInt(cid));
        ctx.globalAlpha = hasSelection && !isSelected ? 0.3 : 1.0;
        ctx.fillStyle = palette[i];
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
        ctx.strokeStyle = "#333";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.globalAlpha = 1.0;

        if (isSelected) {
            ctx.strokeStyle = "#fff";
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(cx, cy, r + 4, 0, 2 * Math.PI);
            ctx.stroke();
            ctx.strokeStyle = "#333";
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(cx, cy, r + 4, 0, 2 * Math.PI);
            ctx.stroke();
        }

        ctx.globalAlpha = hasSelection && !isSelected ? 0.3 : 1.0;
        ctx.font = "bold 11px sans-serif";
        const labelX = cx + r + 3;
        const labelY = cy + 4;
        // White halo for contrast.
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 3;
        ctx.lineJoin = "round";
        ctx.strokeText(cid, labelX, labelY);
        ctx.fillStyle = "#111";
        ctx.fillText(cid, labelX, labelY);
        ctx.globalAlpha = 1.0;
    }
    run.centroidPositions = centroidPositions;
    bindOverviewHandlers(canvas, run);
    canvas.style.cursor = "";
}

function fetchClusterMembers(cid) {
    const runId = window.currentClusterSearch.run_id;
    if (!runId) return;
    fetch(`/cluster_members?run_id=${encodeURIComponent(runId)}&cluster_id=${cid}`)
        .then(r => r.json())
        .then(data => {
            const run = window._clusteringRun;
            if (!run || run.zoomedCid !== cid) return;
            if (data.error) {
                console.error("cluster_members error:", data.error);
                return;
            }
            run.zoomedMembers = {
                cluster_id: cid,
                clip_ids: data.clip_ids || [],
                distances: data.distances || [],
            };
            renderUMAPPlot();
        })
        .catch(e => console.error("cluster_members fetch failed:", e));
}

// Single source of truth for zoom state. The `.zoomed` class on
// #clustering-results-view drives all zoom-aware UI visibility via CSS,
// so adding new zoom-aware elements is a CSS-only change.
function setZoom(cid) {
    const run = window._clusteringRun;
    const wasZoomed = run?.zoomedCid != null;
    const willZoom = cid != null;
    if (run) {
        run.zoomedCid = cid;
        run.zoomedMembers = null;
    }
    const view = document.getElementById("clustering-results-view");
    if (view) view.classList.toggle("zoomed", cid !== null);
    const overlay = document.getElementById("cluster-box-select-overlay");
    if (overlay) overlay.style.display = "none";
    // Cross-fade the canvas when transitioning between overview / zoomed
    // so the view-swap doesn't feel instant.
    if (wasZoomed !== willZoom) {
        const canvas = document.getElementById("umap-canvas");
        if (canvas) {
            canvas.style.opacity = "0";
            requestAnimationFrame(() => {
                requestAnimationFrame(() => { canvas.style.opacity = "1"; });
            });
        }
        // Push the new zoom flag into the URL via a search().
        if (typeof search === "function") search();
    }
    renderBreadcrumb();
}

// Single source of truth for "are we viewing a run?". The `.viewing-run`
// class on #clustering-container drives all run-view-aware visibility via
// CSS — config form / runs list / run-results panel swap declaratively.
function setViewingRun(on) {
    const c = document.getElementById("clustering-container");
    if (c) c.classList.toggle("viewing-run", on);
}

// Render the navigational breadcrumb at the top of the run-results view.
// Segments: Run › Cluster › Range (each appears only when the relevant
// state is set). The deepest segment is the "current location" and is
// non-clickable; segments above it navigate up to that level.
function renderBreadcrumb() {
    const el = document.getElementById("clustering-breadcrumb");
    if (!el) return;
    const run = window._clusteringRun;
    const runId = window.currentClusterSearch?.run_id;
    if (!run || !runId) { el.innerHTML = ""; return; }
    const range = window.currentClusterDistanceRange || { lo: 0, hi: 100 };
    const sels = run.selectedCids || [];
    const hasSelection = sels.length > 0;
    const isMulti = sels.length > 1;
    const isZoomed = run.zoomedCid != null;
    const hasRange = range.lo > 0 || range.hi < 100;

    const seg = (label, current, onclick) => {
        const cls = current ? "breadcrumb-segment current" : "breadcrumb-segment";
        const handler = current ? "" : ` onclick="${onclick}"`;
        return `<span class="${cls}"${handler}>${label}</span>`;
    };

    const parts = [];
    const runLabel = `Run ${runId.slice(0, 8)}`;
    parts.push(seg(runLabel, !hasSelection, "resetClusterSearch()"));
    if (hasSelection) {
        if (isMulti) {
            // Multi-cluster: this is the deepest level; range/zoom not
            // available with multi.
            parts.push(seg(`${sels.length} clusters`, true, ""));
        } else {
            const clusterCurrent = !isZoomed && !hasRange;
            parts.push(seg(`Cluster ${sels[0]}`, clusterCurrent, "exitClusterZoom()"));
            if (isZoomed || hasRange) {
                const tail = hasRange ? `Range ${range.lo}–${range.hi}%` : "Zoomed";
                parts.push(seg(tail, true, ""));
            }
        }
    }
    el.innerHTML = parts.join('<span class="breadcrumb-sep">›</span>');
}

// Display name for a clustering run's embed_type, used in run labels
// and the run-detail banner.
function embedTypeLabel(t) {
    return t === "caption" ? "Caption"
        : t === "visual" ? "Visual"
            : "Text-to-Video";
}

// Draw a placeholder/loading message centred vertically on the canvas.
function drawCenterMsg(ctx, msg, H) {
    ctx.fillStyle = "#999";
    ctx.font = "14px sans-serif";
    ctx.fillText(msg, 20, H / 2);
}

// Find the nearest dot under a mouse event, accepting if within the
// caller-supplied hit radius. Used by both UMAP modes (zoom and full).
function pickNearest(canvas, e, dots, getR) {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
    const my = (e.clientY - rect.top) * (canvas.height / rect.height);
    let nearest = null, minDist = Infinity;
    for (const d of dots) {
        const dist = Math.hypot(mx - d.cx, my - d.cy);
        if (dist < minDist) { minDist = dist; nearest = d; }
    }
    return (nearest && minDist <= getR(nearest)) ? nearest : null;
}

// Lazy-create the singleton hover card used by both UMAP modes.
function ensureUmapTooltip() {
    let tooltip = document.getElementById("umap-tooltip");
    if (tooltip) return tooltip;
    tooltip = document.createElement("div");
    tooltip.id = "umap-tooltip";
    const video = document.createElement("video");
    video.className = "umap-tooltip-video";
    video.muted = true;
    video.loop = true;
    video.playsInline = true;
    video.preload = "metadata";
    const text = document.createElement("div");
    text.className = "umap-tooltip-text";
    tooltip.appendChild(video);
    tooltip.appendChild(text);
    document.body.appendChild(tooltip);
    return tooltip;
}

// Reset the umap tooltip + its preview video to a hidden idle state.
// Used by both UMAP modes; _umapPreviewKey holds whatever id (clip or
// cluster) the tooltip is currently bound to.
function clearTooltip() {
    const tooltip = document.getElementById("umap-tooltip");
    if (!tooltip) return;
    tooltip.style.display = "none";
    window._umapPreviewKey = null;
    const video = tooltip.querySelector(".umap-tooltip-video");
    if (video) {
        video.pause();
        video.removeAttribute("src");
        video.load();
        video.style.display = "none";
    }
}

function exitClusterZoom() {
    // Reset the range first so the search() fired by setZoom carries
    // the cleaned-up state into the URL, not the stale range.
    setClusterDistanceSliderUI(0, 100);
    setZoom(null);
    renderUMAPPlot();
}

function umapGotoCluster() {
    const input = document.getElementById("umap-goto-input");
    if (!input) return;
    const cid = parseInt(input.value);
    if (isNaN(cid)) return;
    const clusters = window._clusteringRun?.data?.clusters || {};
    if (!clusters[String(cid)]) {
        input.setCustomValidity("Unknown cluster ID");
        input.reportValidity();
        return;
    }
    input.setCustomValidity("");
    showClusterSection(cid);
}

function findClosestClusters() {
    const input = document.getElementById("closest-clusters-input");
    const out = document.getElementById("closest-clusters-results");
    const summary = document.getElementById("closest-clusters-summary");
    const list = document.getElementById("closest-clusters-list");
    if (!input || !out || !summary || !list) return;
    const query = input.value.trim();
    const runId = window.currentClusterSearch.run_id;
    if (!query || !runId) return;
    out.style.display = "block";
    summary.textContent = `Searching for "${query}"…`;
    list.innerHTML = "";
    const url = `/closest_clusters?run_id=${encodeURIComponent(runId)}` +
                `&query=${encodeURIComponent(query)}&k=10`;
    fetch(url)
        .then(r => r.json())
        .then(data => renderClosestClusters(out, data))
        .catch(e => {
            summary.textContent = "";
            list.innerHTML = `<p class="form-warning">Search failed: ${e}</p>`;
        });
}

function renderClosestClusters(container, data) {
    const summary = document.getElementById("closest-clusters-summary");
    const list = document.getElementById("closest-clusters-list");
    container.classList.remove("closest-clusters-collapsed");
    list.innerHTML = "";

    if (data.error) {
        summary.textContent = "";
        list.innerHTML = `<p class="form-warning">${data.error}</p>`;
        return;
    }
    if (data.warning) {
        const warn = document.createElement("p");
        warn.className = "form-warning";
        warn.textContent = data.warning;
        list.appendChild(warn);
    }
    const results = data.results || [];
    if (results.length === 0) {
        summary.textContent = `No clusters matched "${data.query}"`;
        return;
    }
    summary.textContent =
        `Top ${results.length} clusters for "${data.query}"`;

    const clusters = window._clusteringRun?.data?.clusters || {};
    const topics = window._clusteringRun?.data?.topics || {};
    for (const r of results) {
        const cid = r.cluster_id;
        const size = (clusters[String(cid)] || {}).cluster_size || 0;
        const kw = ((topics[String(cid)] || {}).keywords || []).slice(0, 4);
        const row = document.createElement("div");
        row.className = "closest-cluster-row";
        row.innerHTML =
            `<span class="closest-cluster-id">Cluster ${cid}</span> ` +
            `<span class="closest-cluster-meta">${size} clips · distance to centroid: ${r.distance.toFixed(2)}</span>` +
            (kw.length ? `<div class="closest-cluster-kw">${kw.join(", ")}</div>` : "");
        row.onclick = () => showClusterSection(parseInt(cid));
        list.appendChild(row);
    }
}

function toggleClosestClustersList() {
    const container = document.getElementById("closest-clusters-results");
    if (container) container.classList.toggle("closest-clusters-collapsed");
}

function clearClosestClusters() {
    const input = document.getElementById("closest-clusters-input");
    const out = document.getElementById("closest-clusters-results");
    if (input) input.value = "";
    if (out) out.style.display = "none";
}

function showClusterSection(cid) {
    const clusters = window._clusteringRun?.data?.clusters || {};
    if (!clusters[String(cid)]) return;
    applyClusterSelection(cid);
    search();
}


function downloadUMAPPlot() {
    const canvas = document.getElementById("umap-canvas");
    if (!canvas) return;
    const a = document.createElement("a");
    a.href = canvas.toDataURL("image/png");
    a.download = `umap_${window.currentClusterSearch.run_id || "plot"}.png`;
    a.click();
}

// Reset the clustering panel UI to its initial state without triggering a search.
// Called from both closeClusteringResults() and resetClusterSearch().
function closeClusteringResultsUI() {
    const run = window._clusteringRun;
    if (run?.resizeObserver) run.resizeObserver.disconnect();
    setZoom(null);
    setViewingRun(false);
    const infoPanel = document.getElementById("cluster-info-panel");
    if (infoPanel) infoPanel.style.display = "none";
    window._clusteringRun = null;
    setClusterDistanceSliderUI(0, 100);
    const tooltip = document.getElementById("umap-tooltip");
    if (tooltip) tooltip.style.display = "none";
    renderBreadcrumb();
}

function closeClusteringResults() {
    window.currentClusterSearch = { run_id: null, cluster_ids: [] };
    search();
    closeClusteringResultsUI();
}

function showClusteringHelp() {
    document.getElementById("clustering-help-content").style.display = "block";
}

function hideClusteringHelp() {
    document.getElementById("clustering-help-content").style.display = "none";
}

function showClosestClustersHelp() {
    document.getElementById("closest-clusters-help-content").style.display = "block";
}

function hideClosestClustersHelp() {
    document.getElementById("closest-clusters-help-content").style.display = "none";
}
