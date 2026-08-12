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

// Global variable to keep track of the currently selected annotation for time logging
// Structure: { videoId: '...', annotationKey: '...', startTime: null, endTime: null }
window.selectedAnnotation = null;

const CAPTION_QUERY_RE = /"([^"]+)"|(\S+)/g;
window.showAllTrajectories = false;
window.showAllMetrics = false;
window.showAllBEV = false;

window.visibleLabelTypes = {
    manual: true,
    autolabel: true,
    numeric: true
};

window.reconstructionStatus = { running: [], ready: [] };

window._loadingTimer = null;
window._searchAbortController = null;
window._clipIDTimestamp = 0;
window._otherFilterTimestamp = 0;
function cancelSearch() {
    if (window._searchAbortController) {
        window._searchAbortController.abort();
        window._searchAbortController = null;
    }
    hideLoading();
}
// Track which clip's InstantNuRec was last pressed (to gate which shows "View")
window.currentInstantNuRecClip = null;
window.currentNurecClip = null;

// Global variable to keep track of the some shortcuts for quick annotation
window.labelShortcuts = [];

// Global playback rate used for all videos on the page
window.globalPlaybackRate = 1.0;


const VPP_VALUES = [4, 6, 8, 10, 12, 16, 20];

function increasePlaybackSpeed() {
    // Cycle through discrete speeds: 1x → 2x → 4x → 6x → 1x → ...
    const steps = [1, 2, 4, 6];
    const curr = Number(window.globalPlaybackRate || 1);
    const currIdx = Math.max(0, steps.indexOf(Math.round(curr)));
    const next = steps[(currIdx + 1) % steps.length];
    window.globalPlaybackRate = next;

    // Apply to all existing videos on the page
    document.querySelectorAll('video').forEach(v => {
        v.playbackRate = next;
    });

    const btn = document.getElementById('increase-speed-btn');
    btn.textContent = `🏃 Playback Speed + (x${next})`;
}

function renderJsonCaption(obj, regex) {
    const esc = (t) =>
        String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const highlight = (t) => {
        if (!regex) return esc(t);
        const START = "\uE001HL_START\uE001";
        const END   = "\uE001HL_END\uE001";
        return esc(String(t).replace(regex, `${START}$1${END}`))
            .replaceAll(START, '<span class="search-highlight">')
            .replaceAll(END, '</span>');
    };
    const renderValue = (v) => {
        if (v === null) return '<span class="json-null">null</span>';
        if (typeof v === "boolean") return `<span class="json-bool">${v}</span>`;
        if (typeof v === "number") return `<span class="json-number">${v}</span>`;
        if (typeof v === "string") return `<span class="json-string">${highlight(v)}</span>`;
        if (Array.isArray(v)) {
            const items = v.map(item => `<li>${renderValue(item)}</li>`).join("");
            return `<ul class="json-array">${items}</ul>`;
        }
        if (typeof v === "object") return renderJsonCaption(v, regex);
        return esc(String(v));
    };
    const rows = Object.entries(obj).map(([k, v]) =>
        `<tr><th class="json-key">${esc(k)}</th><td class="json-val">${renderValue(v)}</td></tr>`
    ).join("");
    return `<table class="json-caption">${rows}</table>`;
}

function renderCaption(raw, regex) {
    // 0) Normalize input -> string
    let s = String(raw ?? "")
        .replace(/\r\n/g, "\n")
        .replace(/\\\s*\n/g, "\n")          // turn trailing "\" line-wraps into real newlines
        .trim();

    // 0a) If the string is valid JSON object, render as a structured table
    if (s.startsWith("{") || s.startsWith("[")) {
        try {
            const parsed = JSON.parse(s);
            if (parsed && typeof parsed === "object") {
                return renderJsonCaption(parsed, regex);
            }
        } catch (_) { /* not valid JSON, fall through */ }
    }

    // 1) Quick heuristic: does it *look* like markdown that we should render?
    const looksLikeMarkdown = (t) => {
        if (/^\s*```/.test(t)) return true;                         // fenced code
        if (/(^|\n)\s*#{1,6}\s/.test(t)) return true;               // headings
        if (/(^|\n)\s*[-*]\s+/.test(t)) return true;                // unordered list
        if (/(^|\n)\s*\d+\.\s+/.test(t)) return true;               // ordered list
        if (/(^|\n)\s*>\s+/.test(t)) return true;                   // blockquote
        if (/\*\*[A-Za-z][^*]{0,50}:\*\*/.test(t)) return true;     // **Label:** pattern
        const pipeLines = t.split("\n").filter(l => /^\s*\|/.test(l)).length;
        if (pipeLines >= 1) return true;                            // pipe-style sections/tables
        if ((t.match(/\|/g) || []).length >= 6) return true;        // dense single-line pipes
        return false;
    };

    // 2) Safe escape helper (used for plain text path)
    const esc = (t) =>
        t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

    // 3) Plain text path (no markdown cues) — keep it simple & safe
    if (!looksLikeMarkdown(s)) {
        if (regex) {
            const START = "\uE001HL_START\uE001";
            const END = "\uE001HL_END\uE001";
            s = s.replace(regex, `${START}$1${END}`);
            s = esc(s).replaceAll(START, '<span class="search-highlight">')
                .replaceAll(END, '</span>');
        } else {
            s = esc(s);
        }
        return s.replace(/\n/g, "<br>");
    }

    // 4) Markdown path — repair common problems *before* rendering

    // Strip single outer ``` fences so we render content, not a code block
    {
        const m = s.match(/^\s*```(?:markdown)?\s*\n([\s\S]*?)\n```[\s]*$/);
        if (m) s = m[1];
    }

    // 4a) Convert pipe-style “sections” into real headings + body.
    // Handles lines like: "| Title | body ..." (even if all content was a single long line).
    s = s.split("\n").map(line => {
        if (!/^\s*\|/.test(line)) return line;
        // Split by pipes, trim, drop empties
        const parts = line.split("|").map(t => t.trim()).filter(Boolean);
        // Turn consecutive pairs into sections: [title, body], [title, body], ...
        const out = [];
        for (let i = 0; i + 1 < parts.length; i += 2) {
            const title = parts[i], body = parts[i + 1];
            if (title && body) out.push(`### ${title}\n\n${body}`);
        }
        return out.length ? out.join("\n\n") : line;
    }).join("\n");

    // 4b) Normalize time-range brackets and small label glitches
    // Merge split ranges like:
    // [ 0
    //   - 5 seconds]
    s = s.replace(/\[\s*([0-9]+(?:\.[0-9]+)?)\s*\n+\s*-\s*([0-9]+(?:\.[0-9]+)?)([^\]]*)\]/g, "[$1 - $2$3]");
    // Ensure "seconds" formatting consistency
    s = s.replace(/\[\s*([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)\s*seconds?\s*\]/gi, "[$1 - $2 seconds]");
    // "**Label:*" -> "**Label:**"
    s = s.replace(/\*\*([A-Za-z0-9 _()"'./-]+):\*/g, "**$1:**");
    // "Action:******" -> "**Action:**"; same for Reason
    s = s.replace(/\b(Action|Reason)\s*:\s*\*{2,}/gi, (_, w) => `**${w[0].toUpperCase() + w.slice(1).toLowerCase()}:**`);
    // Turn "Label: text" lines into list items when not already bolded
    s = s.replace(/(^|\n)\s*([A-Z][A-Za-z ()/"']+):\s*(?!\*\*)([^\n]+)/g,
        (_, pre, label, rest) => `${pre}- **${label}:** ${rest}`);
    // Collapse excessive blank lines
    s = s.replace(/\n{3,}/g, "\n\n");

    // 4c) Search highlight BEFORE markdown rendering (sanitization happens after)
    if (regex) {
        s = s.replace(regex, '<span class="search-highlight">$1</span>');
    }

    // 5) Render Markdown (prefer marked + DOMPurify if available)
    if (window.marked) {
        const html = window.marked.parse(s, { mangle: false, headerIds: false });
        return window.DOMPurify ? window.DOMPurify.sanitize(html) : html;
    }

    // 6) Tiny fallback renderer (in case marked isn’t loaded)
    let h = s
        .replace(/^######\s?(.*)$/gm, "<h6>$1</h6>")
        .replace(/^#####\s?(.*)$/gm, "<h5>$1</h5>")
        .replace(/^####\s?(.*)$/gm, "<h4>$1</h4>")
        .replace(/^###\s?(.*)$/gm, "<h3>$1</h3>")
        .replace(/^##\s?(.*)$/gm, "<h2>$1</h2>")
        .replace(/^#\s?(.*)$/gm, "<h1>$1</h1>")
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/\*([^*]+)\*/g, "<em>$1</em>")
        .replace(/(^|\n)-\s+(.+)(?=\n|$)/g, (_, pre, item) => `${pre}<li>${item}</li>`);
    h = h.replace(/(?:<li>[\s\S]*?<\/li>)/g, (block) => `<ul>${block}</ul>`);
    h = h.replace(/(^|\n)(?!<h\d|<ul>|<li>|<\/ul>|<\/li>|<p>)([^\n][^\n]*)/g,
        (_, pre, line) => `${pre}<p>${line}</p>`);
    return h;
}


function toggleAllTrajectories(el) {
    window.showAllTrajectories = !window.showAllTrajectories;

    const allPlots = document.querySelectorAll(".trajectory-plot");
    allPlots.forEach(plotDiv => {
        if (window.showAllTrajectories) {
            plotDiv.style.display = "block";
            if (!plotDiv.hasChildNodes()) {
                const clipId = plotDiv.dataset.clipId;
                drawD3TrajectoryPlot(clipId, plotDiv);
            }
        } else {
            plotDiv.style.display = "none";
        }

        let activate = plotDiv.style.display === "block";
        el.classList.toggle("selected", activate);
    });

    // Update the eye icon state instead of text label
    const eyeTraj = document.getElementById('eye-trajectories');
    if (eyeTraj) {
        if (window.showAllTrajectories) {
            eyeTraj.classList.remove('hidden');
            eyeTraj.classList.add('visible');
        } else {
            eyeTraj.classList.remove('visible');
            eyeTraj.classList.add('hidden');
        }
    }
}

function toggleAllBEV(el) {
    window.showAllBEV = !window.showAllBEV;

    const allBEVs = document.querySelectorAll(".bev-container");
    allBEVs.forEach(bevDiv => {
        if (window.showAllBEV) {
            bevDiv.style.display = "block";
            if (!bevDiv.hasChildNodes()) {
                const clipId = bevDiv.dataset.clipId;
                initializeBEVForClip(clipId, bevDiv);
            }
        } else {
            bevDiv.style.display = "none";
        }

        let activate = bevDiv.style.display === "block";
        el.classList.toggle("selected", activate);
    });

    // Update the eye icon state
    const eyeBEV = document.getElementById('eye-bev');
    if (eyeBEV) {
        if (window.showAllBEV) {
            eyeBEV.classList.remove('hidden');
            eyeBEV.classList.add('visible');
        } else {
            eyeBEV.classList.remove('visible');
            eyeBEV.classList.add('hidden');
        }
    }
}

function toggleAllMetrics(el) {
    window.showAllMetrics = !window.showAllMetrics;

    const allMetrics = document.querySelectorAll(".metrics-container");
    allMetrics.forEach(metricsDiv => {
        if (window.showAllMetrics) {
            metricsDiv.style.display = "block";
            if (!metricsDiv.hasChildNodes()) {
                const clipId = metricsDiv.dataset.clipId;
                initializeMetricsForClip(clipId, metricsDiv);
            }
        } else {
            metricsDiv.style.display = "none";
        }

        let activate = metricsDiv.style.display === "block";
        el.classList.toggle("selected", activate);
    });

    // Update the eye icon state
    const eyeMetrics = document.getElementById('eye-metrics');
    if (eyeMetrics) {
        if (window.showAllMetrics) {
            eyeMetrics.classList.remove('hidden');
            eyeMetrics.classList.add('visible');
        } else {
            eyeMetrics.classList.remove('visible');
            eyeMetrics.classList.add('hidden');
        }
    }
}

function initializeMetricsForClip(clipId, metricsContainer) {
    // Find the video element for this clip
    const videoElement = document.querySelector(`#video-tile-${clipId} video`);
    if (!videoElement) {
        console.error(`Could not find video element for clip ${clipId}`);
        return;
    }

    // Initialize metrics using the global manager
    if (window.metricsTileManager) {
        window.metricsTileManager.initializeMetrics(clipId, metricsContainer, videoElement);
    }
}

function initializeBEVForClip(clipId, bevContainer) {
    // Find the video element for this clip
    const videoTile = bevContainer.closest('.tile');
    const videoElement = videoTile ? videoTile.querySelector('video') : null;

    if (!videoElement) {
        console.error(`Could not find video element for clip ${clipId}`);
        return;
    }

    // Initialize BEV using the global manager
    if (window.bevTileManager) {
        window.bevTileManager.initializeBEV(clipId, bevContainer, videoElement)
            .catch(error => {
                console.error(`Failed to initialize BEV for ${clipId}:`, error);
            });
    } else {
        console.error('BEVTileManager not initialized');
    }
}

function drawD3TrajectoryPlot(clipId, plotDiv) {
    const video = window.currentVideos.find(v => v.annotations.clip_id === clipId);
    if (!video || !video.positions || video.positions.length < 2) {
        return; // Exit if no valid data
    }
    const data = video.positions.map(([x, y]) => ({ x, y }));
    const margin = { top: 20, right: 20, bottom: 40, left: 50 };
    const width = plotDiv.clientWidth - margin.left - margin.right;
    const height = plotDiv.clientHeight - margin.top - margin.bottom;

    d3.select(plotDiv).html(""); // Clear previous SVG content

    // Store a reference to the main SVG element
    const mainSvg = d3.select(plotDiv)
        .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom);

    // Append the group 'g' element to the main SVG
    const gElement = mainSvg.append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // Canonical ego frame: d.x = forward, d.y = right (so left = -y).
    // Plot forward on the vertical axis (up = forward) and lateral on the
    // horizontal axis (right = +y).
    let lateralExtent = d3.extent(data, d => d.y);
    let lateralMean = (lateralExtent[0] + lateralExtent[1]) / 2;
    let lateralSize = lateralExtent[1] - lateralExtent[0];
    let forwardExtent = d3.extent(data, d => d.x);
    let forwardMean = (forwardExtent[0] + forwardExtent[1]) / 2;
    let forwardSize = forwardExtent[1] - forwardExtent[0];
    lateralSize = Math.max(forwardSize * 0.3, lateralSize);
    forwardSize = Math.max(lateralSize * 0.3, forwardSize);
    lateralExtent[0] = lateralMean - lateralSize / 2;
    lateralExtent[1] = lateralMean + lateralSize / 2;
    forwardExtent[0] = forwardMean - forwardSize / 2;
    forwardExtent[1] = forwardMean + forwardSize / 2;

    const xScale = d3.scaleLinear().domain(lateralExtent).range([0, width]);
    const yScale = d3.scaleLinear().domain(forwardExtent).range([height, 0]);

    // Smooth interpolated spline
    const line = d3.line()
        .x(d => xScale(d.y))
        .y(d => yScale(d.x))
        .curve(d3.curveCatmullRom.alpha(0.5));

    const path = gElement.append("path") // Append to gElement, not mainSvg
        .datum(data)
        .attr("fill", "none")
        .attr("stroke", "blue")
        .attr("stroke-width", 2)
        .attr("d", line);

    const pathNode = path.node();
    const pathTotalLength = pathNode.getTotalLength();

    // Raw observed points (red)
    if (window.currentTrajectoryShapeClipID == clipId) {
        // When trajectory shape search is active, render points with opacity
        const startT = window.currentTrajectoryShapeStartT !== null ? window.currentTrajectoryShapeStartT : 0;
        const endT = window.currentTrajectoryShapeEndT !== null ? window.currentTrajectoryShapeEndT : 20;

        gElement.selectAll(".trajectory-point")
            .data(data.map((d, i) => ({ ...d, i })))
            .enter()
            .append("circle")
            .attr("class", "trajectory-point")
            .attr("cx", d => xScale(d.y))
            .attr("cy", d => yScale(d.x))
            .attr("r", 3)
            .attr("fill", d => {
                const totalLength = data.length - 1;
                const percentage = d.i / totalLength;
                return percentage >= startT / 20 && percentage <= endT / 20 ? "green" : "red";
            })
            .attr("opacity", d => {
                const totalLength = data.length - 1;
                const percentage = d.i / totalLength;
                return percentage >= startT / 20 && percentage <= endT / 20 ? 1 : 0.5;
            });
    } else {
        // Normal rendering without opacity effects
        gElement.selectAll(".trajectory-point")
            .data(data)
            .enter()
            .append("circle")
            .attr("class", "trajectory-point")
            .attr("cx", d => xScale(d.y))
            .attr("cy", d => yScale(d.x))
            .attr("r", 3)
            .attr("fill", "red");
    }

    // Axes
    gElement.append("g")
        .attr("transform", `translate(0,${height})`)
        .call(d3.axisBottom(xScale))
        .append("text")
        .attr("x", width / 2)
        .attr("y", 35)
        .attr("fill", "#000")
        .style("text-anchor", "middle")
        .text("Lateral (m)");

    gElement.append("g")
        .call(d3.axisLeft(yScale))
        .append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", -height / 2)
        .attr("y", -40)
        .attr("fill", "#000")
        .style("text-anchor", "middle")
        .text("Forward (m)");

    if (typeof window.drawTrajectoryCompass === "function") {
        window.drawTrajectoryCompass(gElement, width, height);
    }

    // The end marker (black square) remains static
    const end = data[data.length - 1];
    gElement.append("rect")
        .attr("x", xScale(end.y) - 5)
        .attr("y", yScale(end.x) - 5)
        .attr("width", 10)
        .attr("height", 10)
        .attr("fill", "black");

    // Define the car icon
    const car = gElement.append("text")
        .attr("font-size", "24px")
        .attr("text-anchor", "middle")
        .attr("alignment-baseline", "middle")
        .text("🚗")
        .style("cursor", "grab");

    const videoElement = document.getElementById("video-tile-" + clipId).querySelector("video");

    // Update car position based on video time
    function updateCarPosition(time) {
        const duration = videoElement.duration;
        if (isNaN(duration) || duration === 0 || pathTotalLength === 0) {
            car.style("display", "none");
            return;
        }
        car.style("display", "block");

        const percentage = time / duration;
        const pointIndex = Math.floor(percentage * (data.length - 1));
        const nextIndex = Math.min(pointIndex + 1, data.length - 1);
        const currentPoint = [xScale(data[pointIndex].y), yScale(data[pointIndex].x)];
        const nextPoint = [xScale(data[nextIndex].y), yScale(data[nextIndex].x)];
        const prevPerc = pointIndex / (data.length - 1);
        const nextPerc = nextIndex / (data.length - 1);
        const t = (percentage - prevPerc) / (nextPerc - prevPerc + 1e-6);
        const drawPointX = t * nextPoint[0] + (1 - t) * currentPoint[0];
        const drawPointY = t * nextPoint[1] + (1 - t) * currentPoint[1];
        const xscale = (nextIndex > pointIndex && nextPoint[0] != currentPoint[0]) ? -Math.sign(nextPoint[0] - currentPoint[0]) : -1;

        car.attr("transform", `translate(${drawPointX}, ${drawPointY}) scale(${xscale}, 1)`);
    }

    // Video timeupdate listener (Car moves with video)
    videoElement.addEventListener("timeupdate", () => {
        if (!car.node().__isDragging__) { // Only update car if not currently being dragged by user
            updateCarPosition(videoElement.currentTime);
        }
    });

    // Drag Behavior for the Car
    const drag = d3.drag()
        .on("start", function (event) {
            this.__isDragging__ = true;
            videoElement.pause();
            d3.select(this).style("cursor", "grabbing");
        })
        .on("drag", function (event) {
            const svgDomElement = mainSvg.node();
            const svgPoint = svgDomElement.createSVGPoint();
            svgPoint.x = event.sourceEvent.clientX;
            svgPoint.y = event.sourceEvent.clientY;
            const transformedPoint = svgPoint.matrixTransform(svgDomElement.getScreenCTM().inverse())
                .matrixTransform(gElement.node().getCTM().inverse());

            // Snap to the closest data-point index — same parameterization
            // as updateCarPosition, so the car shown during drag and the
            // video frame at the time we set below correspond to the same
            // real-world point.
            let closestIdx = 0;
            let minDistSq = Infinity;
            for (let i = 0; i < data.length; i++) {
                const dx = transformedPoint.x - xScale(data[i].y);
                const dy = transformedPoint.y - yScale(data[i].x);
                const distSq = dx * dx + dy * dy;
                if (distSq < minDistSq) {
                    minDistSq = distSq;
                    closestIdx = i;
                }
            }

            const newTime = (closestIdx / Math.max(1, data.length - 1)) * videoElement.duration;
            videoElement.currentTime = newTime;
            updateCarPosition(newTime);
        })
        .on("end", function () {
            this.__isDragging__ = false;
            videoElement.play();
            d3.select(this).style("cursor", "grab");
        });

    car.call(drag);

    // Set initial car position
    videoElement.addEventListener("loadedmetadata", () => {
        updateCarPosition(videoElement.currentTime);
    }, { once: true });

    if (videoElement.readyState >= 2) {
        updateCarPosition(videoElement.currentTime);
    }

}


function makeVideoTile(video_data, options) {
    let clip_id = video_data["annotations"]["clip_id"];
    let clip_options = video_data["annotations"]["annotations"];
    let clip_speed = video_data["speed"];
    let clip_acceleration = video_data["acceleration"];
    let clip_curvature = video_data["curvature"];
    let clip_jerk = video_data["jerk"];
    let clip_country = video_data["country"];
    let clip_country_name = video_data["country_name"];
    let data_source = video_data["data_source"];

    let has_embeddings = video_data["has_embeddings"];
    let has_trajectories = video_data["has_trajectories"];

    let semantic_video_score = video_data["semantic_video_score"];
    let semantic_text_score = video_data["semantic_text_score"];
    let trajectory_shape_score = video_data["trajectory_shape_score"];
    let classification_score = video_data["classification_score"];
    let clip_score = video_data["clip_score"];
    let clip_image_score = video_data["clip_image_score"];
    let numeric_scores = video_data["numeric_scores"];
    let cluster_distance_score = video_data["cluster_distance_score"];
    let caption_embed_score = video_data["caption_embed_score"];
    let rrf_score = video_data["rrf_score"];

    let video_tile = document.getElementById("tile-template").cloneNode(true);
    video_tile.id = "video-tile-" + clip_id;
    video_tile.style.display = "";

    let videoElement = video_tile.querySelector("video");
    let sourceElement = video_tile.querySelector("source");
    let videoSlider = video_tile.querySelector(".video-slider");
    let playPauseBtn = video_tile.querySelector(".play-pause-btn");
    let currentTimeSpan = video_tile.querySelector(".current-time");
    let durationSpan = video_tile.querySelector(".duration");
    let startLogBtn = video_tile.querySelector(".start-log-btn");
    let endLogBtn = video_tile.querySelector(".end-log-btn");
    let speedDisplay = video_tile.querySelector(".speed-display")
    let sessionIdSpan = video_tile.querySelector(".session-id-container span");

    let semanticTextScore = video_tile.querySelector(".semantic-text-score-container");
    let trajectoryShapeScore = video_tile.querySelector(".trajectory-shape-score-container");
    let classificationScore = video_tile.querySelector(".classification-score-container");
    let clipScore = video_tile.querySelector(".clip-score-container");
    let numericScore = video_tile.querySelector(".numeric-score-container");
    let clusterDistanceScore = video_tile.querySelector(".cluster-distance-score-container");
    let captionEmbedScore = video_tile.querySelector(".caption-embed-score-container");
    let rrfScore = video_tile.querySelector(".rrf-score-container");

    // Helper to render the Clip ID UI: link + copy button
    function createClipIdWidget(clip_id_value) {
        const container = document.createElement("span");

        // Clickable Clip ID: open search in a new tab with this clip id
        let linkUrl = "/#page=0";
        linkUrl += "&search_clipid=" + encodeURIComponent(clip_id_value);
        linkUrl += "&project_source=" + encodeURIComponent(window.currentProjectSource.join("||"));
        const idLink = document.createElement("a");
        idLink.className = "clip-id-pill";
        idLink.textContent = clip_id_value;
        idLink.href = linkUrl;
        idLink.target = "_blank";
        idLink.rel = "noopener noreferrer";
        idLink.title = "Open search in new tab";
        container.appendChild(idLink);

        // Copy-to-clipboard icon (two-squares) with tooltip
        const copyBtn = document.createElement("button");
        copyBtn.type = "button";
        copyBtn.className = "copy-clipid-btn";
        copyBtn.title = "Copy to clipboard";
        copyBtn.setAttribute("aria-label", "Copy Clip ID");
        copyBtn.textContent = "⧉";
        copyBtn.addEventListener("click", async (e) => {
            e.stopPropagation();
            try {
                await navigator.clipboard.writeText(clip_id_value);
                copyBtn.title = "Copied!";
                setTimeout(() => { copyBtn.title = "Copy to clipboard"; }, 1500);
            } catch (err) {
                // Fallback for older browsers without navigator.clipboard
                const temp = document.createElement("input");
                temp.value = clip_id_value;
                document.body.appendChild(temp);
                temp.select();
                try {
                    document.execCommand("copy");
                    copyBtn.title = "Copied!";
                    setTimeout(() => { copyBtn.title = "Copy to clipboard"; }, 1500);
                } catch (e2) {
                    // ignore
                }
                document.body.removeChild(temp);
            }
        });
        container.appendChild(copyBtn);

        // Link to high-resolution video player. Only NVIDIA-internal datasets
        // are hosted on cdn-prod.nvda.ai, so skip the link otherwise.
        const dsList = Array.isArray(data_source)
            ? data_source
            : String(data_source || "").split(",").map(s => s.trim()).filter(Boolean);
        const meta = window.currentDatasetMetadata || {};
        const isInternalAV = dsList.some(ds =>
            meta[ds]?.license === "internal" &&
            meta[ds]?.category === "Autonomous Driving (AV)"
        );
        if (isInternalAV) {
            const hdLink = document.createElement("a");
            hdLink.className = "copy-clipid-btn";
            hdLink.href = `https://cdn-prod.nvda.ai/v1/player/${encodeURIComponent(clip_id_value)}/camera_front_wide_120fov`;
            hdLink.target = "_blank";
            hdLink.rel = "noopener noreferrer";
            hdLink.title = "Open high-resolution video in new tab";
            hdLink.setAttribute("aria-label", "Open high-resolution video");
            hdLink.textContent = "🎬";
            container.appendChild(hdLink);
        }

        return container;
    }

    // Ensure playback speed follows global setting
    if (videoElement) {
        try { videoElement.playbackRate = window.globalPlaybackRate; } catch (e) { /* ignore */ }
    }

    // Add the clip id to the tile
    sourceElement.src = `/video/${clip_id}.mp4`;
    // video_tile.getElementsByTagName("p")[0].getElementsByTagName("span")[0].innerText = clip_id;
    const idSpan = video_tile.querySelector(".session-id-container span");
    idSpan.replaceWith(createClipIdWidget(clip_id));

    // Add a flag in case the country information if provided
    if (clip_country) {
        let flagImg = document.createElement("img");
        flagImg.src = `https://flagsapi.com/${clip_country}/shiny/32.png`;
        flagImg.alt = `${clip_country} Flag`;
        flagImg.title = clip_country_name;
        (video_tile.querySelector(".session-id-container p") || sessionIdSpan?.parentNode)?.appendChild(flagImg);
    }

    // Show data source (link to per-dataset stats)
    const dataSourceContainer = video_tile.querySelector(".data-source-container");
    if (dataSourceContainer && data_source) {
        const dataSources = Array.isArray(data_source) ? data_source : String(data_source).split(',').map(s => s.trim()).filter(Boolean);
        dataSources.forEach(ds => {
            const link = document.createElement('a');
            link.className = 'data-source-badge';
            link.href = '/data_stats#' + encodeURIComponent('data_source') + '=' + encodeURIComponent(ds);
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            link.title = 'View data stats for ' + ds;
            link.textContent = ds;
            dataSourceContainer.appendChild(link);
        });
    }

    // Add the semantic score if provided
    if (semantic_video_score || semantic_text_score) {
        semanticTextScore.style.display = "block";
        const semParts = [];
        if (semantic_text_score) semParts.push(`Text: ${semantic_text_score.toFixed(4)}`);
        if (semantic_video_score) semParts.push(`Video: ${semantic_video_score.toFixed(4)}`);
        semanticTextScore.innerHTML = `Semantic Score — ${semParts.join(" | ")}`;
    }

    // Add the trajectory shape score if provided
    if (trajectory_shape_score != null) {
        trajectoryShapeScore.style.display = "block";
        trajectoryShapeScore.innerHTML = `Trajectory Shape Score: ${trajectory_shape_score.toFixed(4)}`;
    }

    // Add the classification score if provided
    if (classification_score) {
        classificationScore.style.display = "block";
        const runId = window.currentClassifierSearch.run_id;
        const run = runId
            ? (window.classifierStatuses?.runs || []).find(r => r.run_id === runId)
            : null;
        const labelText = run
            ? (run.positive_labels || []).slice().sort().join("&&")
            : (runId || "");
        classificationScore.innerHTML = `Classification Score for ${labelText}: ${classification_score.toFixed(4)}`;
    }

    // Add the CLIP score if provided
    if (clip_score || clip_image_score) {
        clipScore.style.display = "block";
        const parts = [];
        if (clip_score) parts.push(`Text: ${clip_score.toFixed(4)}`);
        if (clip_image_score) parts.push(`Image: ${clip_image_score.toFixed(4)}`);
        clipScore.innerHTML = `CLIP Score — ${parts.join(" | ")}`;
    }

    // Add the cluster distance score if provided
    if (cluster_distance_score != null) {
        clusterDistanceScore.style.display = "block";
        clusterDistanceScore.innerHTML = `Cluster Distance: ${cluster_distance_score.toFixed(4)}`;
    }

    if (caption_embed_score) {
        captionEmbedScore.style.display = "block";
        captionEmbedScore.innerHTML = `Caption Embed Score: ${caption_embed_score.toFixed(4)}`;
    }

    if (rrf_score != null) {
        rrfScore.style.display = "block";
        rrfScore.innerHTML = `RRF Score: ${rrf_score.toFixed(4)}`;
    }

    // Add the numeric scores if provided
    if (numeric_scores) {
        numericScore.style.display = "block";
        numericScore.innerHTML = Object.entries(numeric_scores)
            .map(([name, score]) => `Numeric Score for ${name}: ${score.toFixed(4)}`)
            .join("<br>");
    }

    // Add the caption (with tabs + search highlighting, keep comments working)
    let captionContainer = video_tile.querySelector(".caption-container");
    let captionBtn = captionContainer.getElementsByTagName("button")[0];
    let captionBox = captionContainer.querySelector(".caption-box");
    let clearCaptionBox = captionBox.querySelector(".clear-annotation-button");
    let captionTextContainer = captionBox.querySelector(".caption-text");

    // Remove only existing tabs if re-rendering, leave other children intact for comments
    let oldTabs = captionTextContainer.querySelector(".caption-tabs");
    if (oldTabs) {
        oldTabs.remove();
    }

    // Get search term
    let searchTerm = document.getElementById("search-term").value;
    var allTerms = [];
    if (searchTerm) allTerms.push(searchTerm);
    [window.currentExtraQueries, window.currentCaptionEmbedExtraQueries, window.currentSemanticExtraQueries].forEach(function(list) {
        if (list) list.forEach(function(q) { if (q) allTerms.push(q); });
    });
    const BOOL_OPS = new Set(["AND", "OR", "NOT"]);
    var regexPattern = allTerms.flatMap(function(t) {
        // Mirror Python _sanitize_fts5_query: preserve quoted phrases; pass
        // boolean operators (AND, OR, NOT) through without highlighting them;
        // group remaining consecutive unquoted tokens into a single phrase.
        const phrases = [];
        const pending = [];
        const re = new RegExp(CAPTION_QUERY_RE.source, "g");
        let m;
        while ((m = re.exec(t.replace(/-/g, " "))) !== null) {
            if (m[1]) {
                if (pending.length) { phrases.push(pending.join(" ")); pending.length = 0; }
                const phrase = m[1].trim();
                if (phrase) phrases.push(phrase);
            } else if (m[2]) {
                if (BOOL_OPS.has(m[2].toUpperCase())) {
                    if (pending.length) { phrases.push(pending.join(" ")); pending.length = 0; }
                } else {
                    pending.push(m[2]);
                }
            }
        }
        if (pending.length) phrases.push(pending.join(" "));
        return phrases;
    }).map(function(token) {
        return token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    }).join("|");
    let regex = regexPattern ? new RegExp(`(${regexPattern})`, "gi") : null;

    // Build tabs container
    let tabs = document.createElement("div");
    tabs.className = "caption-tabs";
    let tabButtons = document.createElement("div");
    tabButtons.className = "caption-tab-buttons";
    let tabContents = document.createElement("div");
    tabContents.className = "caption-tab-contents";

    // Video_data.captions is { model_name: [caption_text | {caption,start_time,end_time}, ...] }
    if (video_data.captions && Object.keys(video_data.captions).length) {
        let first = true;

        Object.entries(video_data.captions).forEach(([model, texts]) => {
            const items = Array.isArray(texts) ? texts : (texts != null ? [texts] : []);

            // Tab button
            const btn = document.createElement("button");
            btn.textContent = model;
            btn.className = "caption-tab-btn" + (first ? " active" : "");
            tabButtons.appendChild(btn);

            // Tab content
            const content = document.createElement("div");
            content.className = "caption-tab-content";
            content.style.display = first ? "block" : "none";

            const modelScores = (video_data.vlm_caption_scores || {})[model] || [];
            const html = items.map((raw, idx) => {
                const isObj = raw && typeof raw === 'object';
                const text = isObj ? raw.caption : raw;
                const rendered = renderCaption(text, regex);
                const hasTimes = isObj && raw.start_time != null && raw.end_time != null;
                const timeHtml = hasTimes
                    ? `<div class="caption-time"><span class="caption-time-pill">${Math.round(raw.start_time)}–${Math.round(raw.end_time)} s</span></div>`
                    : '';
                const cached = modelScores[idx];
                const cachedHtml = cached ? renderVlmCaptionScoresHtml(cached) : '';
                const evalRow = `<div class="vlm-judge-row">`
                    + `<button class="vlm-judge-btn" title="Score this caption with VLM Judge">Evaluate Caption</button>`
                    + `<div class="vlm-judge-result"${cachedHtml ? '' : ' style="display:none"'}>${cachedHtml}</div>`
                    + `</div>`;
                const uid = isObj && raw.uid != null ? raw.uid : '';
                return `<div class="caption-item" data-idx="${idx}" data-caption-uid="${uid}">${timeHtml}${rendered}${evalRow}</div>`;
            }).join('<hr class="caption-sep">');

            content.innerHTML = html;
            tabContents.appendChild(content);

            // Wire tab switching
            btn.onclick = () => {
                tabContents.querySelectorAll(".caption-tab-content").forEach(c => (c.style.display = "none"));
                tabButtons.querySelectorAll(".caption-tab-btn").forEach(b => b.classList.remove("active"));
                content.style.display = "block";
                btn.classList.add("active");
            };

            first = false;
        });
    }

    // Add buttons + contents to tabs
    tabs.appendChild(tabButtons);
    tabs.appendChild(tabContents);

    // Append tabs into caption-text without removing other comment-related elements
    captionTextContainer.appendChild(tabs);

    // VLM Judge: evaluate caption button handler
    tabs.addEventListener("click", function (e) {
        const btn = e.target.closest(".vlm-judge-btn");
        if (!btn) return;
        const item = btn.closest(".caption-item");
        const resultDiv = item.querySelector(".vlm-judge-result");
        const captionText = item.cloneNode(true);
        captionText.querySelectorAll(".vlm-judge-row, .vlm-judge-btn, .vlm-judge-result, .caption-time").forEach(el => el.remove());
        const plainCaption = captionText.textContent.trim();
        const uid = item.dataset.captionUid;

        btn.disabled = true;
        btn.textContent = "Evaluating…";
        resultDiv.style.display = "none";

        const params = new URLSearchParams({ clip_id: clip_id, caption: plainCaption, uid: uid });
        fetch("/api/vlm_judge/caption_score?" + params.toString())
            .then(r => r.json())
            .then(data => {
                btn.textContent = "Evaluate Caption";
                btn.disabled = false;
                if (data.error) {
                    resultDiv.innerHTML = `<span class="vlm-judge-error">Error: ${data.error}</span>`;
                } else if (data.scores) {
                    resultDiv.innerHTML = renderVlmCaptionScoresHtml(data);
                } else {
                    resultDiv.innerHTML = `<span class="vlm-judge-error">Unexpected response</span>`;
                }
                resultDiv.style.display = "block";
            })
            .catch(err => {
                btn.textContent = "Evaluate Caption";
                btn.disabled = false;
                resultDiv.innerHTML = `<span class="vlm-judge-error">${err.message}</span>`;
                resultDiv.style.display = "block";
            });
    });

    // Keep original pop-up open/close
    captionBtn.onclick = function () {
        closeCaptionBoxes();
        captionBox.style.display = "block";
    };
    clearCaptionBox.onclick = function () {
        captionBox.style.display = "none";
    };

    // Disable log buttons initially for a new tile
    startLogBtn.disabled = true;
    endLogBtn.disabled = true;

    videoElement.addEventListener("loadedmetadata", () => {
        videoSlider.max = videoElement.duration;
        durationSpan.innerText = formatTime(videoElement.duration);
    });

    videoElement.addEventListener("timeupdate", () => {
        videoSlider.value = videoElement.currentTime;
        currentTimeSpan.innerText = formatTime(videoElement.currentTime);
        let mph = document.getElementById("convert-to-mph").checked;

        if (clip_speed && clip_speed.length > 0) {
            const frameIndex = Math.round(videoElement.currentTime * (clip_speed.length - 1) / videoElement.duration);
            if (frameIndex >= 0 && frameIndex < clip_speed.length) {
                let speed = mph ? clip_speed[frameIndex] * 2.236936 : clip_speed[frameIndex] * 3.6;
                let units = mph ? "mph" : "km/h";
                let acceleration = clip_acceleration[frameIndex];
                let curvature = clip_curvature[frameIndex];
                speed = speed.toFixed(2).toString().padStart(4, " ");
                acceleration = acceleration.toFixed(2).toString().padStart(5, " ");
                curvature = curvature.toFixed(4).toString().padStart(6, " ");

                speedDisplay.innerText = `Speed: ${speed} ${units} - Accel: ${acceleration} m/s^2 - Curv: ${curvature} m^-1`;
            }
        }
    });

    videoElement.addEventListener("ended", () => {
        playPauseBtn.textContent = "Play";
    });

    playPauseBtn.addEventListener("click", () => {
        if (videoElement.paused || videoElement.ended) {
            videoElement.play();
            playPauseBtn.textContent = "Pause";
        } else {
            videoElement.pause();
            playPauseBtn.textContent = "Play";
        }
    });

    videoSlider.addEventListener("input", () => {
        videoElement.currentTime = videoSlider.value;
    });

    startLogBtn.addEventListener("click", () => {
        // Ensure an annotation is selected for THIS video
        if (window.selectedAnnotation && window.selectedAnnotation.videoId === clip_id) {
            window.selectedAnnotation.startTime = videoElement.currentTime;
            startLogBtn.textContent = "Start recorded...";
            startLogBtn.disabled = true;
            endLogBtn.disabled = false;

            var msg = "";
            msg += "Started annotation for ";
            msg += window.selectedAnnotation.annotationKey;
            msg += " (video: ";
            msg += clip_id;
            msg += ") at ";
            msg += formatTime(window.selectedAnnotation.startTime);
        }
    });

    endLogBtn.addEventListener("click", () => {
        // Ensure an annotation is selected for THIS video and start time is set
        if (window.selectedAnnotation &&
            window.selectedAnnotation.videoId === clip_id &&
            window.selectedAnnotation.startTime !== null) {
            const endTime = videoElement.currentTime;
            var msg = "";
            msg += "Ended annotation for ";
            msg += window.selectedAnnotation.annotationKey;
            msg += " (video: ";
            msg += clip_id;
            msg += ") at ";
            msg += formatTime(endTime);
            // Send update to server and re-render
            updateVideoAnnotation(
                window.selectedAnnotation.videoId,
                window.selectedAnnotation.annotationKey,
                "update_times",
                window.selectedAnnotation.uid,
                window.selectedAnnotation.startTime,
                endTime
            );
            // Reset selected annotation. Button states will be reset by renderVideos
            window.selectedAnnotation = null;
        }
    });


    let select = video_tile.getElementsByTagName("select")[0];
    select.innerHTML = "";

    let defaultOpt = document.createElement("option");
    defaultOpt.text = "Select annotation...";
    defaultOpt.value = "";
    defaultOpt.selected = true;
    defaultOpt.disabled = true;
    select.add(defaultOpt);
    for (let i = 0; i < options.length; i++) {
        let opt = document.createElement("option");
        opt.text = options[i];
        select.add(opt);
    }

    if ($(select).data("select2")) {
        $(select).select2("destroy");
    }
    $(select).select2({
        placeholder: "Select or write your own label...",
        tags: true,
        allowClear: true,
        width: "70%",
        createTag: function (params) {
            var term = $.trim(params.term);

            if (term === '') {
                return null;
            }

            return {
                id: term,
                text: term,
                newTag: true // add additional parameters
            }
        }
    });

    let annotations_div = video_tile.getElementsByClassName("annotations")[0];

    annotations_div.innerHTML = "";

    for (let j = 0; j < clip_options.length; j++) {
        // Skip hidden label types
        if (!window.visibleLabelTypes[clip_options[j].label_type]) {
            continue;
        }

        // Create a wrapper to add the annotation and the action buttons
        let wrapper = document.createElement("div");
        wrapper.className = "annotation-button-wrapper";

        // Annotation button
        let btn = document.createElement("button");
        let buttonText = clip_options[j].key;
        if (clip_options[j].label_type === "manual") {
            buttonText = "🔧 " + buttonText;
        }

        if (clip_options[j].label_type === "autolabel") {
            buttonText = "🎯 " + buttonText;
            btn.style.backgroundColor = "#FFDAB9";
        }

        // Special handling for numeric labels: distinct style and show value
        if (clip_options[j].label_type === "numeric") {
            // Light blue style to differentiate numeric annotations
            btn.style.backgroundColor = "#E6F2FF";
        }

        btn.innerText = buttonText;
        btn.value = clip_options[j].key;
        btn.onclick = selectAnnotation;
        btn.id = clip_options[j].uid;
        // Persist metadata for verify action
        btn.dataset.startTime = clip_options[j].start_time;
        btn.dataset.endTime = clip_options[j].end_time;
        btn.dataset.labelType = clip_options[j].label_type;
        btn.appendChild(makeTimeSpan(clip_options[j].start_time, clip_options[j].end_time));

        // For numeric labels, append the value next to the time span
        if (clip_options[j].label_type === "numeric" && clip_options[j].value !== null && clip_options[j].value !== undefined) {
            const val = clip_options[j].value;
            const isInt = Number.isInteger(val);
            const valText = isInt ? String(val) : Number(val).toFixed(3);
            const valueSpan = document.createElement("span");
            valueSpan.className = "numeric-value";
            valueSpan.style.marginLeft = "6px";
            valueSpan.style.padding = "1px 6px";
            valueSpan.style.borderRadius = "10px";
            valueSpan.style.background = "#DDEBFF";
            valueSpan.style.color = "#1A4099";
            valueSpan.style.fontSize = "11px";
            valueSpan.textContent = valText;
            btn.appendChild(valueSpan);
        }
        wrapper.appendChild(btn);

        // Remove annotation button
        if (clip_options[j].label_type !== "numeric") {
            let clearX = document.createElement("span");
            clearX.className = "clear-annotation-button";
            clearX.innerHTML = "&times;";
            clearX.onclick = removeAnnotation;
            wrapper.appendChild(clearX);
        }

        // Verify (✓) button: only for autolabel annotations
        if (clip_options[j].label_type === "autolabel") {
            let verifyTick = document.createElement("span");
            verifyTick.className = "verify-annotation-button";
            verifyTick.title = "Verify label";
            verifyTick.textContent = "✓";
            verifyTick.onclick = verifyAnnotation;
            wrapper.appendChild(verifyTick);
        }

        annotations_div.appendChild(wrapper);
    }

    // Add metrics container (appears under video when toggled)
    const metricsContainer = document.createElement("div");
    metricsContainer.className = "metrics-container";
    metricsContainer.dataset.clipId = clip_id;
    video_tile.appendChild(metricsContainer);

    // Add BEV container (appears above trajectory plot when toggled)
    const bevContainer = document.createElement("div");
    bevContainer.className = "bev-container";
    bevContainer.dataset.clipId = clip_id;
    video_tile.appendChild(bevContainer);

    const trajectoryPlotDiv = document.createElement("div");
    trajectoryPlotDiv.className = "trajectory-plot";
    // Store the clip ID in a data attribute, so we can retrieve it later
    trajectoryPlotDiv.dataset.clipId = clip_id;
    // Append the div to the bottom of this video tile
    video_tile.appendChild(trajectoryPlotDiv);

    // Create buttons to search for similar clips to this clip (by trajectory shape and embeddings)
    const searchButtonsDiv = document.createElement("div");
    searchButtonsDiv.className = "search-buttons-wrapper";

    video_tile.appendChild(searchButtonsDiv);

    // Conditionally add Reconstruction buttons for specific datasource
    let silAPIs = video_data["sil_apis"];
    addReconstructionButtonsIfApplicable(video_tile, data_source, clip_id, silAPIs);

    // Only make the button if we have trajectories for this clip
    if (has_trajectories) {
        let searchByTrajectoryShapeBtn = document.createElement("button");
        searchByTrajectoryShapeBtn.textContent = "Search by Trajectory Shape";
        searchByTrajectoryShapeBtn.classList.add("search-trajectory-button");
        searchByTrajectoryShapeBtn.onclick = () => {
            window._otherFilterTimestamp = Date.now();
            window.currentTrajectoryShapeClipID = clip_id;
            document.getElementById("trajectory-shape-clipid").value = clip_id;
            search();
        };
        searchButtonsDiv.appendChild(searchByTrajectoryShapeBtn);
    }

    // Only make the button if we have embeddings for this clip
    if (has_embeddings) {
        let searchBySemanticsBtn = document.createElement("button");
        searchBySemanticsBtn.textContent = "Video-to-Video Search";
        searchBySemanticsBtn.classList.add("semantic-search-button");
        searchBySemanticsBtn.onclick = () => {
            window._otherFilterTimestamp = Date.now();
            window.currentSemanticSearchClipID = clip_id;
            document.getElementById("semantic-search-clipid").value = clip_id;
            search();
        };
        searchButtonsDiv.appendChild(searchBySemanticsBtn);
    }

    // When a clustering run is active, show this clip's cluster + a button
    // to drill into it. `cluster_membership` is populated by the server
    // only when `cluster_run_id` is set on the active filters; clips that
    // weren't part of the clustering run get nothing.
    //
    // The button delegates to `showClusterSection` (defined in clustering.js)
    // so the effect is identical to clicking the cluster's centroid in the
    // UMAP: highlights the centroid, opens the cluster info panel with topic
    // keywords, and re-runs the search filtered to that cluster.
    let clusterMembership = video_data["cluster_membership"];
    if (clusterMembership && window.currentClusterSearch.run_id) {
        let showClusterBtn = document.createElement("button");
        showClusterBtn.textContent =
            `Show Cluster ${clusterMembership.cluster_id} ` +
            `(d=${clusterMembership.distance.toFixed(2)})`;
        showClusterBtn.classList.add("show-cluster-button");
        // Disable only when this clip's cluster is the sole active filter
        // (already viewing it). With multiple clusters selected, the button
        // narrows the selection down to this one, so it must stay enabled.
        const selectedIds = (window.currentClusterSearch.cluster_ids || []).map(String);
        showClusterBtn.disabled = (
            selectedIds.length === 1 &&
            selectedIds[0] === String(clusterMembership.cluster_id)
        );
        showClusterBtn.onclick = () => {
            window._otherFilterTimestamp = Date.now();
            showClusterSection(parseInt(clusterMembership.cluster_id));
        };
        searchButtonsDiv.appendChild(showClusterBtn);
    }

    let recommendedAnnotationsDiv = video_tile.querySelector(".recommended-annotations");
    if (window.labelShortcuts.length == 0) {
        recommendedAnnotationsDiv.style.display = "none";
    }
    recommendedAnnotationsDiv.innerHTML = "";
    window.labelShortcuts.forEach(searchTerm => {
        let wrapper = document.createElement("div");
        wrapper.className = "recommended-annotation-button-wrapper";
        // Annotation button
        let btn = document.createElement("button");
        btn.innerText = searchTerm;
        btn.onclick = function () {
            updateVideoAnnotation(clip_id, searchTerm, "add", null, -1, -1);
        };
        wrapper.appendChild(btn);
        recommendedAnnotationsDiv.appendChild(wrapper);
    });

    // Restore VLM search validation from cache or in-flight spinner when re-rendering
    var validationContainer = video_tile.querySelector(".vlm-search-validation");
    if (validationContainer) {
        if (window.vlmValidationCache && window.vlmValidationCache[clip_id]) {
            applyVlmValidationToContainer(validationContainer, window.vlmValidationCache[clip_id]);
        } else if (window.vlmValidatingClipIds && window.vlmValidatingClipIds.has(clip_id)) {
            validationContainer.innerHTML = '<div class="vlm-search-validation-loading"><span class="vlm-spinner"></span>Validating…</div>';
            validationContainer.style.display = "block";
        }
    }

    return video_tile;
}

function postReconstructionAction(clipId, method) {
    const payload = `reconstruction::${clipId}::${method}`;
    return fetch("/", {
        method: "POST",
        headers: { "Content-Type": "text/plain" },
        body: payload,
    }).then((res) => {
        if (!res.ok) {
            throw new Error(`Server responded ${res.status}`);
        }
        return true;
    }).catch((err) => {
        console.error("Reconstruction action failed", err);
        alert("Failed to trigger reconstruction action: " + err.message);
        return false;
    });
}

function applyVlmValidationToContainer(container, r) {
    if (!container || !r) return;
    var matchLabel = r.match ? "✓ Match" : "✗ No match";
    var reasoning = (r.reasoning || r.analysis || "").trim();
    container.innerHTML = "<div class=\"vlm-search-validation-badge vlm-search-" + (r.match ? "match" : "nomatch") + "\">" + matchLabel + "</div>" +
        (reasoning ? "<div class=\"vlm-search-validation-reasoning\" title=\"" + reasoning.replace(/"/g, "&quot;") + "\">" + reasoning + "</div>" : "");
    container.style.display = "block";
}

function getVlmJudgeTopK() {
    var el = document.getElementById("vlm-judge-modal-top-k");
    var v = el ? parseInt(String(el.value).trim(), 10) : NaN;
    if (isNaN(v) || v < 1) v = 20;
    var n = (window.searchJudgeClipIds && window.searchJudgeClipIds.length) || 0;
    if (n > 0) return Math.min(v, n);
    return v;
}

function requestVlmValidation(searchTerm, clipIds) {
    var btn = document.getElementById("vlm-judge-btn");
    var okBtn = document.getElementById("vlm-judge-query-ok");
    if (okBtn) { okBtn.disabled = true; okBtn.textContent = "Validating…"; }

    window.vlmValidatingClipIds = new Set(clipIds);
    var spinner = '<div class="vlm-search-validation-loading"><span class="vlm-spinner"></span>Validating…</div>';
    clipIds.forEach(function (cid) {
        var tile = document.getElementById("video-tile-" + cid);
        if (!tile) return;
        var container = tile.querySelector(".vlm-search-validation");
        if (!container) return;
        container.innerHTML = spinner;
        container.style.display = "block";
    });

    var params = new URLSearchParams({ search: searchTerm, clip_ids: clipIds.join(",") });
    fetch("/api/vlm_judge/validate_search?" + params.toString())
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (okBtn) { okBtn.textContent = "Run"; updateVlmJudgeRunBtn(); }
            if (data.error) {
                alert("VLM validation failed: " + data.error);
                return;
            }
            window.vlmValidatingClipIds = new Set();
            window.vlmValidationCache = window.vlmValidationCache || {};
            (data.results || []).forEach(function (r) {
                window.vlmValidationCache[r.clip_id] = { match: r.match, reasoning: (r.reasoning || r.analysis || "").trim() };
                var tile = document.getElementById("video-tile-" + r.clip_id);
                if (!tile) return;
                var container = tile.querySelector(".vlm-search-validation");
                if (!container) return;
                applyVlmValidationToContainer(container, { match: r.match, reasoning: r.reasoning, analysis: r.analysis });
            });
        })
        .catch(function (err) {
            if (okBtn) { okBtn.textContent = "Run"; updateVlmJudgeRunBtn(); }
            window.vlmValidatingClipIds = new Set();
            clipIds.forEach(function (cid) {
                var tile = document.getElementById("video-tile-" + cid);
                if (!tile) return;
                var container = tile.querySelector(".vlm-search-validation");
                if (container) container.style.display = "none";
            });
            alert("VLM validation request failed: " + err.message);
        });
}

function vlmJudgeDesc() {
    var ranked = window.searchJudgeClipIds || [];
    var available = ranked.length;
    if (!available) return "No search results available. Run a search first.";
    var kEl = document.getElementById("vlm-judge-modal-top-k");
    var k = kEl ? (parseInt(kEl.value, 10) || 20) : 20;
    var will = Math.min(k, available);
    var total = window.totalVideos || available;
    var desc = "The VLM will check the top " + will + " of " + total + " search results against your query.";
    if (total > available) {
        desc += " (VLM Judge is limited to the top " + available + " ranked results.)";
    }
    return desc;
}

function openVlmJudgeModal() {
    var box = document.getElementById("vlm-judge-query-box");
    var input = document.getElementById("vlm-judge-query-input");
    var okBtn = document.getElementById("vlm-judge-query-ok");
    var desc = document.getElementById("vlm-judge-query-desc");
    var kEl = document.getElementById("vlm-judge-modal-top-k");
    if (kEl && !kEl.value) kEl.value = "20";
    if (desc) desc.textContent = vlmJudgeDesc();
    var isValidating = window.vlmValidatingClipIds && window.vlmValidatingClipIds.size > 0;
    if (okBtn) {
        okBtn.textContent = isValidating ? "Validating…" : "Run";
        var hasResults = (window.searchJudgeClipIds || []).length > 0;
        okBtn.disabled = isValidating || !hasResults || !(input && input.value.trim());
    }
    if (box) box.style.display = "flex";
    if (input) { input.focus(); try { input.select(); } catch (e) {} }
}

function updateVlmJudgeRunBtn() {
    var input = document.getElementById("vlm-judge-query-input");
    var okBtn = document.getElementById("vlm-judge-query-ok");
    var desc = document.getElementById("vlm-judge-query-desc");
    if (desc) desc.textContent = vlmJudgeDesc();
    if (!okBtn) return;
    var hasResults = (window.searchJudgeClipIds || []).length > 0;
    okBtn.disabled = !hasResults || !(input && input.value.trim());
}

function hideVlmJudgeQueryBox() {
    var box = document.getElementById("vlm-judge-query-box");
    if (box) box.style.display = "none";
}

function runVlmJudgeFromQueryBox() {
    var input = document.getElementById("vlm-judge-query-input");
    var searchTerm = (input && input.value) ? String(input.value).trim() : "";
    if (!searchTerm) return;
    var ranked = window.searchJudgeClipIds || [];
    if (!ranked.length) return;
    var topK = getVlmJudgeTopK();
    hideVlmJudgeQueryBox();
    requestVlmValidation(searchTerm, ranked.slice(0, topK));
}

function runVlmJudge() {
    if (!window.vlmJudgeAvailable) {
        alert("VLM Judge is not available.");
        return;
    }
    openVlmJudgeModal();
}

function createReconstructionButton(clipId, method, label, viewerUrl, waitTime, postToServer) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'reconstruction-button';
    btn.textContent = label;
    btn.setAttribute('data-method', method);
    btn.setAttribute('data-clip-id', clipId);
    btn.onclick = function () {
        if (btn.textContent.startsWith("View") || !postToServer) {
            const w = window.open(viewerUrl, method);
            if (w) {
                w.focus();
            }
        } else {
            document.querySelectorAll(`[data-method=${method}]`).forEach(function (other) {
                // Reset the label that will be used by onclick to figure out
                // whether we need to open the url or talk to the server.
                if (other.getAttribute("data-clip-id") == clipId) {
                    return;
                }

                if (other.textContent.startsWith("View")) {
                    other.textContent = label;
                }
            });

            btn.disabled = true;
            btn.textContent = "Processing (Wait...)";
            postReconstructionAction(clipId, method).then(function (ok) {
                // Something went quite wrong so revert the button
                if (!ok) {
                    btn.disabled = false;
                    btn.textContent = label;
                }

                // We 're good so wait a small amount of time to let the other
                // server start up whatever is needed to start up and then make the
                // button take us there.
                else {
                    setTimeout(function () {
                        btn.disabled = false;
                        btn.textContent = "View " + label;
                    }, waitTime);
                }
            });
        }
    };
    return btn;
}

function createSauronVisualizerButton(clipId) {
    const btn = document.createElement('button');
    btn.type = 'button';
    // Match styling with InstantNuRec/NuRec buttons
    btn.className = 'reconstruction-button depth-button';
    btn.textContent = "🏷️ Sauron Autolabels";
    btn.setAttribute('data-clip-id', clipId);

    btn.onclick = function () {
        const box_url = `/boxes_video/${clipId}.mp4`;
        const pointmap_url = `/point_video/${clipId}.mp4`;
        const url = `/depth_video/${clipId}.mp4`;
        const mfmrh_url = `/mfmrh_video/${clipId}.mp4`;
        // Open a popup window with three side-by-side videos and keep them in sync
        const w = window.open('', `Sauron_${clipId}`, 'width=1280,height=720');
        if (w) {
            const doc = w.document;
            doc.open();
            doc.write(`<!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8" />
                    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
                    <title>Sauron Visualizer - ${clipId}</title>
                    <style>
                        * { box-sizing: border-box; }
                        html, body { height: 100%; }
                        body { margin: 0; background: #0b0b0b; color: #eaeaea; font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
                        .wrap { display: flex; flex-direction: column; gap: 12px; padding: 12px; height: 100%; width: 100%; }
                        .row { display: flex; gap: 12px; width: 100%; flex: 1 1 0; }
                        .col { flex: 1 1 0; display: flex; flex-direction: column; min-width: 0; }
                        .title { font-size: 14px; font-weight: 600; letter-spacing: .2px; padding: 8px 6px; text-align: center; color: #f5f5f5; background: #1a1a1a; border-radius: 6px; margin-bottom: 8px; }
                        .vid-wrap { position: relative; flex: 1 1 auto; display: flex; }
                        video { width: 100%; height: 100%; object-fit: contain; background: #000; border-radius: 6px; }
                        .hint { position: fixed; bottom: 8px; right: 12px; opacity: .7; font-size: 12px; }
                    </style>
                </head>
                <body>
                    <div class="wrap">
                        <div class="row">
                            <div class="col">
                                <div class="title">Bounding boxes</div>
                                <div class="vid-wrap"><video id="vid-bboxes" src="${box_url}" controls playsinline muted loop></video></div>
                            </div>
                            <div class="col">
                                <div class="title">Depth</div>
                                <div class="vid-wrap"><video id="vid-depth" src="${url}" controls playsinline muted loop></video></div>
                            </div>
                            <div class="col">
                                <div class="title">Pointmaps</div>
                                <div class="vid-wrap"><video id="vid-pointmaps" src="${pointmap_url}" controls playsinline muted loop></video></div>
                            </div>
                        </div>
                        <div class="row">
                            <div class="col">
                                <div class="title">Human Motion Estimation</div>
                                <div class="vid-wrap"><video id="vid-mfmrh" src="${mfmrh_url}" controls playsinline muted loop></video></div>
                            </div>
                        </div>
                    </div>
                    <div class="hint">Produced with Sauron and MFM-HMR</div>
                    <script>
                        (function() {
                            const openerDoc = window.opener && window.opener.document;
                            const parentTile = openerDoc ? openerDoc.getElementById('video-tile-${clipId}') : null;
                            const parentVideo = parentTile ? parentTile.querySelector('video') : null;

                            const vids = [
                                document.getElementById('vid-depth'),
                                document.getElementById('vid-bboxes'),
                                document.getElementById('vid-pointmaps'),
                                document.getElementById('vid-mfmrh')
                            ];

                            // Helper flags to avoid feedback loops
                            let syncingFromParent = false;
                            let syncingFromChild = false;
                            const EPS = 0.15; // seconds

                            function setChildrenPaused(paused) {
                                vids.forEach(v => {
                                    if (paused) { v.pause(); } else { v.play().catch(() => {}); }
                                });
                            }

                            function alignChildrenTime(t) {
                                vids.forEach(v => {
                                    if (Math.abs(v.currentTime - t) > EPS) {
                                        try { v.currentTime = t; } catch (e) {}
                                    }
                                });
                            }

                            function attachChildHandlers() {
                                vids.forEach(v => {
                                    v.addEventListener('play', () => {
                                        if (syncingFromParent) return;
                                        if (parentVideo) {
                                            syncingFromChild = true;
                                            try { parentVideo.play(); } catch (e) {}
                                            syncingFromChild = false;
                                        } else {
                                            // Keep siblings in sync when no parent
                                            setChildrenPaused(false);
                                        }
                                    });

                                    v.addEventListener('pause', () => {
                                        if (syncingFromParent) return;
                                        if (parentVideo) {
                                            syncingFromChild = true;
                                            try { parentVideo.pause(); } catch (e) {}
                                            syncingFromChild = false;
                                        } else {
                                            setChildrenPaused(true);
                                        }
                                    });

                                    const seekEvents = ['seeking', 'seeked'];
                                    seekEvents.forEach(ev => v.addEventListener(ev, () => {
                                        if (syncingFromParent) return;
                                        if (parentVideo) {
                                            syncingFromChild = true;
                                            try { parentVideo.currentTime = v.currentTime; } catch (e) {}
                                            syncingFromChild = false;
                                        } else {
                                            alignChildrenTime(v.currentTime);
                                        }
                                    }));
                                });
                            }

                            function attachParentHandlers() {
                                if (!parentVideo) return;
                                // Initialize children to parent's state
                                const init = () => {
                                    alignChildrenTime(parentVideo.currentTime || 0);
                                    setChildrenPaused(parentVideo.paused);
                                };
                                if (parentVideo.readyState >= 1) { init(); }
                                else { parentVideo.addEventListener('loadedmetadata', init, { once: true }); }

                                parentVideo.addEventListener('play', () => {
                                    if (syncingFromChild) return;
                                    syncingFromParent = true;
                                    setChildrenPaused(false);
                                    syncingFromParent = false;
                                });

                                parentVideo.addEventListener('pause', () => {
                                    if (syncingFromChild) return;
                                    syncingFromParent = true;
                                    setChildrenPaused(true);
                                    syncingFromParent = false;
                                });

                                const syncTime = () => {
                                    if (syncingFromChild) return;
                                    syncingFromParent = true;
                                    alignChildrenTime(parentVideo.currentTime);
                                    syncingFromParent = false;
                                };
                                parentVideo.addEventListener('timeupdate', syncTime);
                                parentVideo.addEventListener('seeked', syncTime);

                                // Clean up if popup closes
                                window.addEventListener('beforeunload', () => {
                                    parentVideo.removeEventListener('timeupdate', syncTime);
                                    parentVideo.removeEventListener('seeked', syncTime);
                                });
                            }

                            // Mute child videos to avoid multiple audio streams
                            vids.forEach(v => v.muted = true);

                            attachChildHandlers();
                            attachParentHandlers();

                            // Autoplay/align on load depending on parent state
                            const tryStart = () => {
                                if (parentVideo) {
                                    alignChildrenTime(parentVideo.currentTime || 0);
                                    if (!parentVideo.paused) setChildrenPaused(false);
                                }
                            };
                            // When all child metadata is ready, try to align
                            let readyCount = 0;
                            vids.forEach(v => v.addEventListener('loadedmetadata', () => {
                                readyCount += 1;
                                if (readyCount === vids.length) tryStart();
                            }));
                        })();
                    </script>
                </body>
                </html>`);
            doc.close();
            w.focus();
        }
    };

    return btn;
}

function createSimpleVideoPopupButton(clipId, label, videoUrl, windowName, windowTitle) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'reconstruction-button depth-button';
    btn.textContent = label;
    btn.setAttribute('data-clip-id', clipId);

    btn.onclick = function () {
        const w = window.open('', windowName, 'width=960,height=540');
        if (w) {
            const doc = w.document;
            doc.open();
            doc.write(`<!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8" />
                    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
                    <title>${windowTitle}</title>
                    <style>
                        html, body { height: 100%; }
                        body { margin: 0; background: #000; display: flex; align-items: center; justify-content: center; }
                        video { max-width: 100%; max-height: 100%; width: 100%; height: auto; }
                    </style>
                </head>
                <body>
                    <video src="${videoUrl}" controls autoplay playsinline></video>
                </body>
                </html>`);
            doc.close();
            w.focus();
        }
    };

    return btn;
}

function createDepthVisualizerButton(clipId) {
    return createSimpleVideoPopupButton(
        clipId, "📏 Depth",
        `/depth_video/${clipId}.mp4`,
        `DepthVideo_${clipId}`,
        `Depth Video - ${clipId}`
    );
}

function createVipeLabelsButton(clipId) {
    return createSimpleVideoPopupButton(
        clipId, "Vipe Labels",
        `/vipe_video/${clipId}.mp4`,
        `VipeLabels_${clipId}`,
        `Vipe Labels - ${clipId}`
    );
}

function addReconstructionButtonsIfApplicable(tileEl, dataSource, clipId, silAPIs) {
    if (!silAPIs) {
        return;
    }

    const wrap = document.createElement('div');
    wrap.className = 'reconstruction-buttons-wrapper';

    const groupTitle = document.createElement('div');
    groupTitle.className = 'reconstruction-group-title';
    groupTitle.textContent = '🧩 SIL APIs';
    wrap.appendChild(groupTitle);

    const btnRow = document.createElement('div');
    btnRow.className = 'reconstruction-button-row';

    if (silAPIs.includes("InstantNurec")) {
        btnRow.appendChild(
            createReconstructionButton(
                clipId,
                "InstantNuRec",
                "⚡ InstantNuRec",
                "http://localhost:7590/",
                5000,
                true
            )
        );
    }
    if (silAPIs.includes("Nurec")) {
        btnRow.appendChild(
            createReconstructionButton(
                clipId,
                "NuRec",
                "🐢 NuRec",
                "http://localhost:8080/",
                11000,
                true
            )
        );
    }
    //if (silAPIs.includes("Drive")) {
    //    btnRow.appendChild(
    //        createReconstructionButton(
    //            clipId,
    //            "Drive",
    //            "🚗 Drive",
    //            `http://localhost:8000/usdz=${encodeURIComponent(clipId)}.usdz`
    //        )
    //    );
    //}
    if (silAPIs.includes("Autolabels")) {
        btnRow.appendChild(createSauronVisualizerButton(clipId));
    }

    if (silAPIs.includes("Vipe")) {
        btnRow.appendChild(createVipeLabelsButton(clipId));
    }

    wrap.appendChild(btnRow);
    tileEl.appendChild(wrap);
}


// Renders all video tiles on the current page
function renderVideos(videos, options) {
    let currentPageDiv = document.getElementById("current-page");

    // Clean up all existing BEV and metrics instances before destroying DOM elements
    if (window.bevTileManager) {
        window.bevTileManager.destroyAll();
    }
    if (window.metricsTileManager) {
        window.metricsTileManager.destroyAll();
    }

    currentPageDiv.innerHTML = ""; // Clear existing tiles
    for (let i = 0; i < videos.length; i++) {
        let newTile = makeVideoTile(videos[i], options);
        currentPageDiv.appendChild(newTile);
    }

    if (window.showAllTrajectories) {
        const allPlots = document.querySelectorAll(".trajectory-plot");
        allPlots.forEach(plotDiv => {
            plotDiv.style.display = "block";
            if (!plotDiv.hasChildNodes()) {
                const clipId = plotDiv.dataset.clipId;
                drawD3TrajectoryPlot(clipId, plotDiv);
            }
        });
    }

    if (window.showAllMetrics) {
        const allMetrics = document.querySelectorAll(".metrics-container");
        allMetrics.forEach(metricsDiv => {
            metricsDiv.style.display = "block";
            if (!metricsDiv.hasChildNodes()) {
                const clipId = metricsDiv.dataset.clipId;
                initializeMetricsForClip(clipId, metricsDiv);
            }
        });
    }

    if (window.showAllBEV) {
        const allBEVs = document.querySelectorAll(".bev-container");
        allBEVs.forEach(bevDiv => {
            bevDiv.style.display = "block";
            if (!bevDiv.hasChildNodes()) {
                const clipId = bevDiv.dataset.clipId;
                initializeBEVForClip(clipId, bevDiv);
            }
        });
    }
}


function setFilterMode(mode) {
    window.currentFilterMode = mode;
    syncModeControl("filter-mode-row", "#annotation-tag", "currentFilterMode");
}

function renderFilters() {
    let options = window.currentOptions;
    addAnnotationOptions(options, window.currentFilter);
    setFilterMode(window.currentFilterMode || "any");
    addClassifierSearchOptions(options);
    addOptionsForLabelManipulation(options);
    addLabelsToExclude(options, window.currentLabelsToExclude);

    let metricNameOptions = window.currentMetricNames;
    addMetricOptions(metricNameOptions, window.currentNumericFilter);

    let dataSourceOptions = window.currentDataSourceOptions;
    addDataSourceOptions(dataSourceOptions, window.currentDataSource);
    addOptionsForDatasetSelection(dataSourceOptions);

    let projectSourceOptions = window.currentProjectOptions;
    addProjectSourceOptions(projectSourceOptions, window.currentProjectSource);

    let labelTypeOptions = window.currentLabelTypeOptions;
    addLabelTypeOptions(labelTypeOptions, window.currentLabelTypes);

    renderLabelManipulationButtons();
    renderClassifierPanel();

    document.getElementById("with-times").checked = window.currentTimes === true;
    document.getElementById("without-times").checked = window.currentTimes === false;
    const withMetricsEl = document.getElementById("with-metrics");
    const withBEVEl = document.getElementById("with-bev");
    const metricsAvailable = window.currentWithMetricsAvailable !== false;
    const bevAvailable = window.currentWithBEVAvailable !== false;
    withMetricsEl.checked = window.currentWithMetrics === true;
    withBEVEl.checked = window.currentWithBEV === true;
    withMetricsEl.disabled = !metricsAvailable;
    withBEVEl.disabled = !bevAvailable;
    if (!metricsAvailable) {
        withMetricsEl.checked = false;
        window.currentWithMetrics = null;
    }
    if (!bevAvailable) {
        withBEVEl.checked = false;
        window.currentWithBEV = null;
    }
    document.getElementById("without-ann").checked = window.currentWithoutAnn === true;
    document.getElementById("left-hand-driving").checked = window.currentLeftHandDriving === true;
    document.getElementById("with-ego-data").checked = window.currentWithEgoData === true;
    document.getElementById("search-term").value = window.currentSearch || "";
    document.getElementById("speed-search-term").value = window.currentSpeedQuery;
    document.getElementById("search-country").value = window.currentCountryQuery;
    document.getElementById("search-clipid").value = window.currentClipIDQuery;
    document.getElementById("trajectory-pattern").value = window.currentTrajectoryPattern || "";
    document.getElementById("semantic-search-text").value = window.currentSemanticSearchText;
    document.getElementById("visual-search-text").value = window.currentVisualSearchText;
    document.getElementById("caption-embed-search-text").value = window.currentCaptionEmbedSearchText || "";
    const vppIdx = VPP_VALUES.indexOf(window.currentVideosPerPage || 6);
    document.getElementById("videos-per-page").value = vppIdx >= 0 ? vppIdx : 0;
    document.getElementById("videos-per-page-display").textContent = window.currentVideosPerPage || 6;

    const sil = String(window.currentSILAPIs || "");
    document.getElementById("with-drive").checked = sil.includes("Drive");
    document.getElementById("with-instant-nurec").checked = sil.includes("InstantNurec");
    document.getElementById("with-nurec").checked = sil.includes("Nurec");
    document.getElementById("with-sauron").checked = sil.includes("Autolabels");

    populateSILAPIs();

    showContextualFilters();

    // Build the export search buttons
    document.querySelectorAll(".export-search-link").forEach((link) => {
        link.href = link.href.split("?")[0] + "?" + window.location.hash.substr(1);
    });
}

function render() {
    // Update page info
    document.getElementById("page-number").innerText = window.currentPage;
    document.getElementById("total-pages").innerText = window.totalPages;
    document.getElementById("videos-total").innerText = window.totalVideos ?? 0;
    document.getElementById("annotations-count").innerText = window.annotationsCount;
    document.getElementById("manual-annotations-count").innerText = window.manualAnnotationsCount;
    document.getElementById("autolabel-annotations-count").innerText = window.autolabelAnnotationsCount;

    // Update the labeling recommendations
    updateShortcuts();

    // Render the main content
    renderVideos(window.currentVideos, window.currentOptions);

    // Render the filters
    renderFilters();

    // Update the pagination
    const prevBtn = document.getElementById("prevBtn");
    const nextBtn = document.getElementById("nextBtn");
    prevBtn.disabled = (window.currentPage == 0);
    nextBtn.disabled = (window.currentPage == window.totalPages - 1);

    const gotoInput = document.getElementById("goto-page-input");
    gotoInput.value = window.currentPage;
    gotoInput.max = window.totalPages - 1;
    document.getElementById("total-pages-pagination").innerText = window.totalPages;
}

function search() {
    resetRRFIfTooFewFilters();
    let filterContainer = document.getElementById("filters");
    let withTimes = document.getElementById("with-times").checked;
    let withoutTimes = document.getElementById("without-times").checked;
    let withoutAnn = document.getElementById("without-ann").checked;
    let leftHandDriving = document.getElementById("left-hand-driving").checked;
    let searchTerm = document.getElementById("search-term").value;
    let speedSearchTerm = document.getElementById("speed-search-term").value;
    let countryQuery = document.getElementById("search-country").value;
    let clipIDQuery = document.getElementById("search-clipid").value;
    let withEgoData = document.getElementById("with-ego-data").checked;
    let withMetrics = document.getElementById("with-metrics").checked;
    let withBEV = document.getElementById("with-bev").checked;
    let trajectoryPattern = document.getElementById("trajectory-pattern").value;
    let trajectoryShapeClipID = window.currentTrajectoryShapeClipID;
    let semanticSearchClipID = window.currentSemanticSearchClipID;
    let semanticSearchText = document.getElementById("semantic-search-text").value;
    let visualSearchText = document.getElementById("visual-search-text").value;
    let captionEmbedSearchText = document.getElementById("caption-embed-search-text").value;
    let searchTermInComments = document.getElementById("search-comment-term").value;
    let wmClassName = document.getElementById("wm-class-name").value;
    let wmMinCount = document.getElementById("wm-min-count").value;
    let wmMaxCount = document.getElementById("wm-max-count").value;
    let wmMaxDist = document.getElementById("wm-max-dist").value;
    let wmMinTime = document.getElementById("wm-min-time").value;
    let wmAngleRangeSelect = document.getElementById("wm-angle-range");
    let wmAngleRange = Array.from(wmAngleRangeSelect.selectedOptions).map(option => option.value);

    let filterOptions = document.getElementById("annotation-tag");
    let filterValue = Array.from(filterOptions.selectedOptions).map(option => option.value);
    filterValue = filterValue.join("||")

    let filterMode = window.currentFilterMode || "any";

    let metricOptions = document.getElementById("metric-names");
    let metricRanges = document.getElementById("metric-ranges");
    let metricValue = Array.from(metricOptions.selectedOptions).map(function (option) {
        const value = option.value;
        for (let i = 0; i < metricRanges.childNodes.length; i++) {
            const range = metricRanges.childNodes[i];
            if (range.firstChild.innerText == value) {
                const inputs = range.getElementsByTagName("input");
                const vmin = inputs[0].value || "-inf";
                const vmax = inputs[1].value || "inf";
                const order_input = range.getElementsByClassName("metric-sort-arrow");
                if (order_input[0].textContent == '▲') {
                    return `${value},${vmin},${vmax},asc`;
                }
                else {
                    return `${value},${vmin},${vmax},desc`;
                }
            }
        }
        return `${value},-inf,inf,desc`;
    });
    metricValue = metricValue.join("||");

    let dataSource = document.getElementById("data-source");
    let selectedDataSource = Array.from(dataSource.selectedOptions).map(option => option.value);
    let dataSourceMode = window.currentDataSourceMode || "any";
    selectedDataSource = selectedDataSource.join("||")

    let projectSource = document.getElementById("project-select");
    let selectedProjectSource = Array.from(projectSource.selectedOptions).map(option => option.value);
    selectedProjectSource = selectedProjectSource.join("||")

    let labelsToExcludeOptions = document.getElementById("labels-to-exclude-choices");
    let labelsToExclude = Array.from(labelsToExcludeOptions.selectedOptions).map(option => option.value);
    let labelsToExcludeMode = window.currentLabelsToExcludeMode || "any";
    labelsToExclude = labelsToExclude.join("||")

    let labelTypeOptions = document.getElementById("label-types");
    let labelTypes = Array.from(labelTypeOptions.selectedOptions).map(option => option.value);
    labelTypes = labelTypes.join("||")

    if (trajectoryShapeClipID === null) {
        let tsc = document.getElementById("trajectory-shape-clipid").value;
        if (tsc !== "") {
            trajectoryShapeClipID = document.getElementById("trajectory-shape-clipid").value;
        }
    }
    if (semanticSearchClipID === null) {
        let tsc = document.getElementById("semantic-search-clipid").value;
        if (tsc !== "") {
            semanticSearchClipID = document.getElementById("semantic-search-clipid").value;
        }
    }

    // If other filters were modified more recently than the clip ID field, the
    // clip ID search is no longer the user's intent — ignore it for this call.
    if (clipIDQuery && window._otherFilterTimestamp > window._clipIDTimestamp) {
        clipIDQuery = "";
    }

    // If the clip ID was typed more recently, clear all contextual filters so the
    // result is just that clip. populateUI will sync the DOM after the response.
    if (clipIDQuery && window._clipIDTimestamp > window._otherFilterTimestamp) {
        withTimes = false; withoutTimes = false; withoutAnn = false;
        leftHandDriving = false; withEgoData = false;
        searchTerm = ""; speedSearchTerm = ""; countryQuery = "";
        trajectoryPattern = ""; trajectoryShapeClipID = null;
        window.currentTrajectoryShapeClipID = null;
        semanticSearchClipID = null;
        window.currentSemanticSearchClipID = null;
        semanticSearchText = ""; visualSearchText = "";
        window.currentVisualSearchImageId = null;
        captionEmbedSearchText = ""; searchTermInComments = "";
        wmClassName = ""; wmAngleRange = [];
        filterValue = ""; labelsToExclude = ""; labelTypes = "";
        window.currentClassifierSearch = { run_id: null, expression: null };
        window.currentClipList = { hash: null, count: 0 };
    }

    // Reset the annotation select box to All
    if (withoutAnn) {
        filterValue = "";
        let select = filterContainer.getElementsByTagName("select")[0];
        if (select && select.selectedOptions.length > 0) {
            select.selectedOptions[0].disabled = false;
            select.selectedIndex = 0;
        } else if (select) {
            select.selectedIndex = 0;
        }
    }

    // Reset the pattern trajectory selection in case a speed search term is given
    if (speedSearchTerm) {
        document.getElementById("trajectory-pattern").selectedIndex = 0;
        trajectoryPattern = null;
    }

    // Reset the speed search if a trajectory pattern is selected
    if (trajectoryPattern) {
        document.getElementById("speed-search-term").value = "";
        speedSearchTerm = null;
    }

    let trajectoryShapeStartT = document.getElementById("trajectory-shape-start-t").value;
    let trajectoryShapeEndT = document.getElementById("trajectory-shape-end-t").value;
    if (trajectoryShapeClipID) {
        // In case the same start and end time is given assume that it was a mistake
        if (trajectoryShapeStartT === trajectoryShapeEndT) {
            trajectoryShapeStartT = null;
            trajectoryShapeEndT = null;
            resetTrajectoryTime();
        }
    }

    let apisFilter = populateSILAPIs();

    showPage(0, {
        filter: (filterValue != "") ? filterValue : null,
        numeric_filter: (metricValue != "") ? metricValue : null,
        times: (!withTimes && !withoutTimes) ? null : withTimes,
        without_ann: (withoutAnn != "") ? withoutAnn : null,
        left_hand_driving: leftHandDriving ? true : null,
        search: (searchTerm != "") ? searchTerm : null,
        search_speed: (speedSearchTerm != "") ? speedSearchTerm : null,
        search_country: (countryQuery != "") ? countryQuery : null,
        search_clipid: (clipIDQuery != "") ? clipIDQuery : null,
        with_ego_data: withEgoData ? true : null,
        with_metrics: withMetrics ? true : null,
        with_bev: withBEV ? true : null,
        trajectory_pattern: (trajectoryPattern != "") ? trajectoryPattern : null,
        trajectory_shape_clipid: (trajectoryShapeClipID != "") ? trajectoryShapeClipID : null,
        trajectory_shape_start_t: (trajectoryShapeClipID != "" && trajectoryShapeStartT != "0") ? trajectoryShapeStartT : null,
        trajectory_shape_end_t: (trajectoryShapeClipID != "" && trajectoryShapeEndT != "20") ? trajectoryShapeEndT : null,
        semantic_search_clipid: (semanticSearchClipID != "") ? semanticSearchClipID : null,
        semantic_search_text: (semanticSearchText != "") ? semanticSearchText : null,
        classifier_run_id: window.currentClassifierSearch.run_id,
        probability_expression: window.currentClassifierSearch.expression,
        clip_id_list_hash: window.currentClipList.hash,
        visual_search_text: (visualSearchText != "") ? visualSearchText : null,
        visual_search_image_id: window.currentVisualSearchImageId || null,
        caption_embed_search: (captionEmbedSearchText != "") ? captionEmbedSearchText : null,
        label_types: (labelTypes != "") ? labelTypes : null,
        search_comments: (searchTermInComments != "") ? searchTermInComments : null,
        wm_class_name: (wmClassName != "") ? wmClassName : null,
        wm_min_count: (wmClassName != "") ? wmMinCount : null,
        wm_max_count: (wmClassName != "") ? wmMaxCount : null,
        wm_max_dist: (wmClassName != "" && wmAngleRange.length > 0) ? wmMaxDist : null,
        wm_min_time: (wmClassName != "" && wmAngleRange.length > 0) ? wmMinTime : null,
        wm_angle_range: (wmClassName != "" && wmAngleRange.length > 0) ? wmAngleRange.join(",") : null,
        data_source: (selectedDataSource != "") ? selectedDataSource : null,
        data_source_mode: modePayloadValue(selectedDataSource, dataSourceMode),
        project_source: (selectedProjectSource != "") ? selectedProjectSource : null,
        labels_to_exclude: (labelsToExclude != "") ? labelsToExclude : null,
        labels_to_exclude_mode: modePayloadValue(labelsToExclude, labelsToExcludeMode),
        sil_apis: (apisFilter != "") ? apisFilter : null,
        ...clusterFilterPayload(),
        filter_mode: modePayloadValue(filterValue, filterMode),
        rank_mode: document.getElementById("rrf-toggle-checkbox")?.checked ? "rrf" : null,
        n: window.currentVideosPerPage !== 6 ? window.currentVideosPerPage : null,
    });
}

function updateShortcuts() {
    function shouldAdd(x) {
        let words = x.split(" ").map(x => x.trim()).filter(x => x !== "");
        if (words.length > 0 && words.length < 4) {
            return true;
        }
        return false;
    }

    let shortcutSet = new Set(
        document.getElementById("shortcuts").value
            .split(",")
            .map(x => x.trim())
            .filter(x => x !== '')
    );

    if (window.currentSearch !== null && shouldAdd(window.currentSearch)) {
        shortcutSet.add(window.currentSearch);
    }
    if (window.currentTrajectoryPattern !== null) {
        shortcutSet.add(window.currentTrajectoryPattern);
    }
    if (window.currentSemanticSearchText !== null && shouldAdd(window.currentSemanticSearchText)) {
        shortcutSet.add(window.currentSemanticSearchText);
    }

    if (window.currentClassifierSearch.run_id) {
        const run = (window.classifierStatuses?.runs || [])
            .find(r => r.run_id === window.currentClassifierSearch.run_id);
        if (run) {
            shortcutSet.add((run.positive_labels || []).slice().sort().join("&&"));
        }
    }

    if (window.currentVisualSearchText) {
        shortcutSet.add(window.currentVisualSearchText);
    }

    window.labelShortcuts = Array.from(shortcutSet);
}

function showPage(pageNum, filters, skipLoading = false) {
    // Reset selected annotation state when changing pages
    window.selectedAnnotation = null;

    if (window._rewriteOriginalQuery) {
        if (filters.search !== (window._rewriteOriginalQuery['caption'] || null)) clearRewriteTags('caption');
        if (filters.caption_embed_search !== (window._rewriteOriginalQuery['caption-embed'] || null)) clearRewriteTags('caption-embed');
        if (filters.semantic_search_text !== (window._rewriteOriginalQuery['semantic'] || null)) clearRewriteTags('semantic');
        if (filters.visual_search_text !== (window._rewriteOriginalQuery['visual'] || null)) clearRewriteTags('visual');
    }

    var captionExtraQueries = getSelectedRewrites('caption');
    var captionExtraQueriesParam = captionExtraQueries.length
        ? captionExtraQueries.join("||")
        : (filters.caption_extra_queries || null);

    var captionEmbedExtraQueries = getSelectedRewrites('caption-embed');
    var captionEmbedExtraQueriesParam = captionEmbedExtraQueries.length
        ? captionEmbedExtraQueries.join("||")
        : (filters.caption_embed_extra_queries || null);

    var semanticExtraQueries = getSelectedRewrites('semantic');
    var semanticExtraQueriesParam = semanticExtraQueries.length
        ? semanticExtraQueries.join("||")
        : (filters.semantic_extra_queries || null);

    var visualExtraQueries = getSelectedRewrites('visual');
    var visualExtraQueriesParam = visualExtraQueries.length
        ? visualExtraQueries.join("||")
        : (filters.visual_extra_queries || null);

    let path = buildEndpoint("/videos", {
        page: pageNum,
        ...filters,
        caption_extra_queries: captionExtraQueriesParam,
        caption_embed_extra_queries: captionEmbedExtraQueriesParam,
        semantic_extra_queries: semanticExtraQueriesParam,
        visual_extra_queries: visualExtraQueriesParam,
    });

    let hash = "#" + path.substr(8);
    if (window.location.hash != hash) {
        history.pushState(null, "", hash);
    }

    if (!skipLoading) {
        var hasRewrites = (captionExtraQueries.length && filters.search)
            || (captionEmbedExtraQueries.length && filters.caption_embed_search)
            || (semanticExtraQueries.length && filters.semantic_search_text)
            || (visualExtraQueries.length && filters.visual_search_text);
        showLoading(hasRewrites ? "Searching with rewrites..." : "Searching...");
    }


    if (window._searchAbortController) {
        window._searchAbortController.abort();
    }
    window._searchAbortController = new AbortController();

    fetch(path, { signal: window._searchAbortController.signal }).then(function (response) {
        return response.json();
    }).then(function (data) {
        window._searchAbortController = null;
        window.totalVideos = data.num_videos;
        window.currentPage = data.page;
        window.totalPages = data.total;
        window.currentVideosPerPage = data.n;
        window.annotationsCount = data.annotations_count;
        window.manualAnnotationsCount = data.manual_annotations_count;
        window.autolabelAnnotationsCount = data.autolabel_annotations_count;
        window.currentVideos = data.videos;
        window.currentOptions = data.options;
        window.currentFilter = data.filter;
        window.currentFilterMode = data.filter_mode || "any";
        window.currentDataSourceMode = data.data_source_mode || "any";
        window.currentLabelsToExcludeMode = data.labels_to_exclude_mode || "any";
        window.currentMetricNames = data.metric_names;
        window.currentNumericFilter = data.numeric_filter;
        window.currentTimes = data.times;
        window.currentLeftHandDriving = data.left_hand_driving;
        window.currentWithoutAnn = data.without_ann;
        window.currentSearch = data.search;
        window.searchJudgeClipIds = data.search_judge_clip_ids || [];
        window.vlmJudgeMaxK = data.vlm_judge_max_k != null ? data.vlm_judge_max_k : 0;
        var kInput = document.getElementById("vlm-judge-modal-top-k");
        if (kInput && window.vlmJudgeMaxK > 0) kInput.setAttribute("max", String(window.vlmJudgeMaxK));
        else if (kInput) kInput.removeAttribute("max");
        window.currentSpeedQuery = data.search_speed;
        window.currentCountryQuery = data.search_country;
        window.currentClipIDQuery = data.search_clipid;
        window.currentWithEgoData = data.with_ego_data;
        window.currentWithMetrics = data.with_metrics;
        window.currentWithBEV = data.with_bev;
        window.currentWithMetricsAvailable = data.with_metrics_available;
        window.currentWithBEVAvailable = data.with_bev_available;
        window.currentTrajectoryPattern = data.trajectory_pattern;
        window.currentTrajectoryShapeClipID = data.trajectory_shape_clipid;
        window.currentTrajectoryShapeStartT = data.trajectory_shape_start_t;
        window.currentTrajectoryShapeEndT = data.trajectory_shape_end_t;
        window.currentSemanticSearchClipID = data.semantic_search_clipid;
        window.currentSemanticSearchText = data.semantic_search_text;
        window.currentVisualSearchText = data.visual_search_text;
        window.currentVisualSearchImageId = data.visual_search_image_id || null;
        window.currentCaptionEmbedSearchText = data.caption_embed_search;
        window.currentClassifierSearch = {
            run_id: data.classifier_run_id || null,
            expression: data.probability_expression || null,
        };
        window.currentClusterSearch = {
            run_id: data.cluster_run_id || null,
            cluster_ids: clusterIdsFromData(data.cluster_ids),
        };
        // Preserve the count we already have if the hash didn't change
        // (avoids a redundant /clip_list fetch); otherwise hydrate.
        const newClipListHash = data.clip_id_list_hash || null;
        if (newClipListHash !== window.currentClipList.hash) {
            window.currentClipList = {hash: newClipListHash, count: 0};
            hydrateClipListFromUrl();
        }
        window.currentLabelTypes = data.label_types;
        window.currentLabelTypeOptions = data.label_type_options;
        window.currentSearchTermInComments = data.search_comments
        window.currentWMClassName = data.wm_class_name;
        window.currentWMMinCount = data.wm_min_count;
        window.currentWMMaxCount = data.wm_max_count;
        window.currentWMMaxDist = data.wm_max_dist;
        window.currentWMMinTime = data.wm_min_time;
        window.currentWMAngleRange = data.wm_angle_range;
        window.currentDataSourceOptions = data.data_source_options;
        window.currentDatasetMetadata = data.dataset_metadata;
        window.currentDataSource = data.data_source;
        window.currentProjectOptions = data.project_options;
        window.currentProjectSource = data.project_source;
        window.currentLabelsToExclude = data.labels_to_exclude;
        window.currentSILAPIs = data.sil_apis;
        window.currentExtraQueries = data.caption_extra_queries || [];
        window.currentCaptionEmbedExtraQueries = data.caption_embed_extra_queries || [];
        window.currentSemanticExtraQueries = data.semantic_extra_queries || [];
        window.currentVisualExtraQueries = data.visual_extra_queries || [];
        window._rewriteOriginalQuery = window._rewriteOriginalQuery || {};
        if (data.caption_extra_queries && data.caption_extra_queries.length) {
            window._rewriteOriginalQuery['caption'] = data.search;
            renderRewriteTags(data.caption_extra_queries, 'caption');
        }
        if (data.caption_embed_extra_queries && data.caption_embed_extra_queries.length) {
            window._rewriteOriginalQuery['caption-embed'] = data.caption_embed_search;
            renderRewriteTags(data.caption_embed_extra_queries, 'caption-embed');
        }
        if (data.semantic_extra_queries && data.semantic_extra_queries.length) {
            window._rewriteOriginalQuery['semantic'] = data.semantic_search_text;
            renderRewriteTags(data.semantic_extra_queries, 'semantic');
        }
        if (data.visual_extra_queries && data.visual_extra_queries.length) {
            window._rewriteOriginalQuery['visual'] = data.visual_search_text;
            renderRewriteTags(data.visual_extra_queries, 'visual');
        }

        // Restore clustering panel state if a cluster search is active.
        // loadClusteringResults is idempotent: a no-op if the panel is
        // already populated for this run.
        const cs = window.currentClusterSearch;
        if (cs.run_id) {
            const wantsZoom = !!window._restoreClusterZoom;
            window._restoreClusterZoom = false;
            loadClusteringResults(cs.run_id, cs.cluster_ids, wantsZoom);
        }

        // Show/hide rewrite buttons based on backend availability
        var rewriteSearchValues = {
            'caption': data.search,
            'caption-embed': data.caption_embed_search,
            'semantic': data.semantic_search_text,
            'visual': data.visual_search_text,
        };
        ['caption', 'caption-embed', 'semantic', 'visual'].forEach(function(t) {
            var btn = document.getElementById(t + '-rewrite-btn');
            if (!btn) return;
            btn.style.display = data.query_rewrite_available ? "" : "none";
            btn.disabled = !rewriteSearchValues[t];
        });

        // Show/hide VLM Judge block when judge is available
        window.vlmJudgeAvailable = !!data.vlm_judge_available;
        var vlmJudgeBlock = document.getElementById("vlm-judge-block");
        if (vlmJudgeBlock) {
            vlmJudgeBlock.style.display = data.vlm_judge_available ? "" : "none";
        }
        /* Top K row is shown only when user clicks VLM Judge, not on load */

        render();
        hideLoading();
    }).catch(function (error) {
        window._searchAbortController = null;
        if (error.name === 'AbortError') {
            return;
        }
        hideLoading();
        alert("Search request failed. Please try again.");
    });
}

function nextPage() {
    if (window.currentPage < window.totalPages - 1) {
        showPage(window.currentPage + 1, buildCurrentFilters(), true);
    }
    window.scrollTo(0, 0);
}

function prevPage() {
    if (window.currentPage > 0) {
        showPage(window.currentPage - 1, buildCurrentFilters(), true);
    }
    window.scrollTo(0, 0);
}

function goToPage(pageNum) {
    const page = Math.max(0, Math.min(parseInt(pageNum), window.totalPages - 1));
    if (!isNaN(page)) {
        showPage(page, buildCurrentFilters(), true);
        window.scrollTo(0, 0);
    }
}

window.onhashchange = function () {
    let filters = getQueryParams(window.location.hash.substr(1));
    let page = (filters.page == undefined) ? 0 : filters.page;
    let filter = (filters.filter === undefined) ? null : filters.filter;
    let numeric_filter = (filters.numeric_filter === undefined) ? null : filters.numeric_filter;
    let times = (filters.times === undefined) ? null : filters.times;
    let withoutAnn = (filters.without_ann === undefined) ? null : filters.without_ann;
    let leftHandDriving = (filters.left_hand_driving === undefined) ? null : filters.left_hand_driving;
    let searchTerm = (filters.search === undefined) ? null : filters.search;
    let speedQuery = (filters.search_speed === undefined) ? null : filters.search_speed;
    let countryQuery = (filters.search_country === undefined) ? null : filters.search_country;
    let clipIDQuery = (filters.search_clipid === undefined) ? null : filters.search_clipid;
    let withEgoData = (filters.with_ego_data === undefined) ? null : filters.with_ego_data;
    let withMetrics = (filters.with_metrics === undefined) ? null : filters.with_metrics;
    let withBEV = (filters.with_bev === undefined) ? null : filters.with_bev;
    withMetrics = (withMetrics === "true") ? true : null;
    withBEV = (withBEV === "true") ? true : null;
    let withTrajectoryPattern = (filters.trajectory_pattern === undefined) ? null : filters.trajectory_pattern;
    let trajectoryShapeClipID = (filters.trajectory_shape_clipid === undefined) ? null : filters.trajectory_shape_clipid;
    let trajectoryShapeStartT = (filters.trajectory_shape_start_t === undefined) ? null : filters.trajectory_shape_start_t;
    let trajectoryShapeEndT = (filters.trajectory_shape_end_t === undefined) ? null : filters.trajectory_shape_end_t;
    let semanticSearchClipID = (filters.semantic_search_clipid === undefined) ? null : filters.semantic_search_clipid;
    let semanticSearchText = (filters.semantic_search_text === undefined) ? null : filters.semantic_search_text;
    let visualSearchText = (filters.visual_search_text === undefined) ? null : filters.visual_search_text;
    let classifierRunId = (filters.classifier_run_id === undefined) ? null : filters.classifier_run_id;
    let probabilityExpression = (filters.probability_expression === undefined) ? null : filters.probability_expression;
    let clipIdListHash = (filters.clip_id_list_hash === undefined) ? null : filters.clip_id_list_hash;
    let labelTypes = (filters.label_types === undefined) ? null : filters.label_types;
    let searchTermInComments = (filters.search_comments === undefined) ? null : filters.search_comments;
    let wmClassName = (filters.wm_class_name === undefined) ? null : filters.wm_class_name;
    let wmMinCount = (filters.wm_min_count === undefined) ? null : filters.wm_min_count;
    let wmMaxCount = (filters.wm_max_count === undefined) ? null : filters.wm_max_count;
    let wmMaxDist = (filters.wm_max_dist === undefined) ? null : filters.wm_max_dist;
    let wmMinTime = (filters.wm_min_time === undefined) ? null : filters.wm_min_time;
    let wmAngleRange = (filters.wm_angle_range === undefined) ? null : filters.wm_angle_range;
    let dataSource = (filters.data_source === undefined) ? null : filters.data_source;
    let projectSource = (filters.project_source === undefined) ? null : filters.project_source;
    let labelsToExclude = (filters.labels_to_exclude === undefined) ? null : filters.labels_to_exclude;
    let silAPIs = (filters.sil_apis === undefined) ? null : filters.sil_apis;
    let clusterRunId = (filters.cluster_run_id === undefined) ? null : filters.cluster_run_id;
    let clusterIds = (filters.cluster_ids === undefined || filters.cluster_ids === "")
        ? null
        : String(filters.cluster_ids).split(",").filter(Boolean).join(",");
    const lo = filters.cluster_distance_min === undefined
        ? 0 : parseFloat(filters.cluster_distance_min);
    const hi = filters.cluster_distance_max === undefined
        ? 100 : parseFloat(filters.cluster_distance_max);
    const safeLo = isNaN(lo) ? 0 : Math.max(0, Math.min(100, lo));
    const safeHi = isNaN(hi) ? 100 : Math.max(0, Math.min(100, hi));
    const clusterZoom = filters.cluster_zoom === "1" || filters.cluster_zoom === 1;
    // One-shot flag consumed by the search-response handler that calls
    // loadClusteringResults — the response payload doesn't echo zoom
    // state, so we carry it across via this global instead.
    window._restoreClusterZoom = clusterZoom;
    setClusterDistanceSliderUI(safeLo, safeHi);
    let captionEmbedSearch = (filters.caption_embed_search === undefined) ? null : filters.caption_embed_search;
    let extraQueries = (filters.caption_extra_queries === undefined) ? null : filters.caption_extra_queries;
    let captionEmbedExtraQueries = (filters.caption_embed_extra_queries === undefined) ? null : filters.caption_embed_extra_queries;
    let semanticExtraQueries = (filters.semantic_extra_queries === undefined) ? null : filters.semantic_extra_queries;
    let visualExtraQueriesFromFilter = (filters.visual_extra_queries === undefined) ? null : filters.visual_extra_queries;
    let visualSearchImageId = (filters.visual_search_image_id === undefined) ? null : filters.visual_search_image_id;
    let filterMode = (filters.filter_mode === undefined) ? "any" : filters.filter_mode;
    let dataSourceMode = (filters.data_source_mode === undefined) ? "any" : filters.data_source_mode;
    let labelsToExcludeMode = (filters.labels_to_exclude_mode === undefined) ? "any" : filters.labels_to_exclude_mode;
    let rankMode = (filters.rank_mode === undefined) ? null : filters.rank_mode;
    window.currentRankMode = rankMode === "rrf" ? "rrf" : "priority";
    let n = (filters.n === undefined) ? 6 : parseInt(filters.n);
    window.currentVideosPerPage = n;

    showPage(page, {
        filter,
        numeric_filter,
        times,
        without_ann: withoutAnn,
        left_hand_driving: leftHandDriving,
        search: searchTerm,
        search_speed: speedQuery,
        search_country: countryQuery,
        search_clipid: clipIDQuery,
        with_ego_data: withEgoData,
        with_metrics: withMetrics,
        with_bev: withBEV,
        trajectory_pattern: withTrajectoryPattern,
        trajectory_shape_clipid: trajectoryShapeClipID,
        trajectory_shape_start_t: trajectoryShapeStartT,
        trajectory_shape_end_t: trajectoryShapeEndT,
        semantic_search_clipid: semanticSearchClipID,
        semantic_search_text: semanticSearchText,
        classifier_run_id: classifierRunId,
        probability_expression: probabilityExpression,
        clip_id_list_hash: clipIdListHash,
        visual_search_text: visualSearchText,
        visual_search_image_id: visualSearchImageId,
        label_types: labelTypes,
        search_comments: searchTermInComments,
        wm_class_name: wmClassName,
        wm_min_count: wmMinCount,
        wm_max_count: wmMaxCount,
        wm_max_dist: wmMaxDist,
        wm_min_time: wmMinTime,
        wm_angle_range: wmAngleRange,
        data_source: dataSource,
        data_source_mode: modePayloadValue(dataSource, dataSourceMode),
        project_source: projectSource,
        labels_to_exclude: labelsToExclude,
        labels_to_exclude_mode: modePayloadValue(labelsToExclude, labelsToExcludeMode),
        sil_apis: silAPIs,
        cluster_run_id: clusterRunId,
        cluster_ids: clusterIds,
        cluster_distance_min: safeLo > 0 ? safeLo : null,
        cluster_distance_max: safeHi < 100 ? safeHi : null,
        cluster_zoom: clusterZoom ? 1 : null,
        caption_embed_search: captionEmbedSearch,
        caption_extra_queries: extraQueries,
        caption_embed_extra_queries: captionEmbedExtraQueries,
        semantic_extra_queries: semanticExtraQueries,
        visual_extra_queries: visualExtraQueriesFromFilter,
        filter_mode: modePayloadValue(filter, filterMode),
        rank_mode: rankMode === "rrf" ? "rrf" : null,
        n: n !== 6 ? n : null,
    });
}


function goToFirstPage() {
    showPage(0, buildCurrentFilters());
    window.scrollTo(0, 0);
}

function showAutolabelWarning() {
    document.getElementById("autolabel-warning-box").style.display = "flex";

    const labelInput = document.getElementById("autolabel-label");
    const ids = window.currentClusterSearch.cluster_ids || [];
    if (!labelInput.value && ids.length === 1) {
        const run_id = window.currentClusterSearch.run_id;
        labelInput.value = `cluster_${run_id.slice(0, 6)}_${ids[0]}`;
        changeAutolabelButtons();
    }

    window.scrollTo(0, 0);
}

function hideAutolabelWarning() {
    document.getElementById("autolabel-warning-box").style.display = "none";
    document.getElementById("autolabel-label").value = "";
    changeAutolabelButtons();
}

function changeAutolabelButtons() {
    const unionButton = document.getElementById("autolabel-union-button");
    const replaceButton = document.getElementById("autolabel-replace-button");
    const clearButton = document.getElementById("clear-autolabel-button");
    const textLabel = document.getElementById("autolabel-label");
    if (textLabel.value == "") {
        unionButton.disabled = true;
        replaceButton.disabled = true;
        clearButton.disabled = true;
    } else {
        unionButton.disabled = false;
        replaceButton.disabled = false;
        clearButton.disabled = false;
    }
}

function autolabelSearch(action) {
    const label = document.getElementById("autolabel-label").value;
    const projectToWrite = document.getElementById("save-project-name").value.trim();
    if (!validateProjectName()) {
        if (!getProjectToWrite()) {
            logMissingProjectToWrite("updating a video annotation");
            return;
        }
    }

    let path = buildEndpoint("/videos", {
        page: window.currentPage,
        ...buildCurrentFilters(),
    });
    console.log(path);

    const pages = document.getElementById("pages-to-save").value;
    const nClips = document.getElementById("n-clips-to-save").value;

    const payload = `auto_label::${path}::${label}::${action}::${pages}::${nClips}::${projectToWrite}`;
    const req = new XMLHttpRequest();
    req.addEventListener("error", () => {
        console.log("Error communicating with server for classifier action!");
    });
    req.addEventListener("load", () => {
        if (req.status !== 200) {
            console.log("Failed to perform classifier action on server!");
            return;
        }
        search();
        checkClassifierState();
    });

    req.open("POST", "");
    req.setRequestHeader("Content-Type", "text/plain");
    req.send(payload);
    hideAutolabelWarning();
}

function modifyLabels(action, oldLabel, newLabel) {
    const projectToWrite = document.getElementById("save-project-name").value.trim();
    if (!validateProjectName()) {
        if (!getProjectToWrite()) {
            logMissingProjectToWrite("updating a video annotation");
            return;
        }
    }

    showLoading();
    const req = new XMLHttpRequest();
    req.addEventListener("error", () => {
        console.log("Error communicating with server for label manipulation!");
    });
    req.addEventListener("load", () => {
        if (req.status !== 200) {
            console.log("Failed to perform label operation on server!");
            return;
        }
        search();
    });
    req.open("POST", "");
    req.setRequestHeader("Content-Type", "text/plain");
    req.send(`${action}::${oldLabel}::${newLabel}::${projectToWrite}`);
}

function renameLabel() {
    const oldLabel = document.getElementById("rename-from-label").value;
    const newLabel = document.getElementById("rename-to-label").value.trim();
    if (!oldLabel || !newLabel) {
        return;
    }
    if (oldLabel === newLabel) {
        return;
    }
    modifyLabels("rename_label", oldLabel, newLabel);
    document.getElementById("rename-to-label").value = "";
}

function mergeLabels() {
    const selectElement = document.getElementById("merge-labels-from");
    const selectedValues = [];

    // Iterate through all options to find the selected ones
    for (let i = 0; i < selectElement.options.length; i++) {
        if (selectElement.options[i].selected) {
            selectedValues.push(selectElement.options[i].value);
        }
    }
    const newLabel = document.getElementById("merge-labels-to").value.trim();
    if (!newLabel || selectedValues.length === 0) {
        return
    }
    console.log(selectedValues.join(","), newLabel);
    modifyLabels("merge_label", selectedValues.join(","), newLabel);
    document.getElementById("merge-labels-to").value = "";
}

function deleteLabel() {
    const label = document.getElementById("delete-label").value;
    if (!label) {
        return;
    }
    modifyLabels("delete_label", label, "");
}

function applyMassLabel() {
    const projectToWrite = document.getElementById("save-project-name").value.trim();
    if (!validateProjectName()) {
        if (!getProjectToWrite()) {
            logMissingProjectToWrite("updating a video annotation");
            return;
        }
    }
    const label = document.getElementById("mass-label-value").value.trim();
    if (!label) {
        return;
    }

    const fileInput = document.getElementById("mass-label-clipid-file");
    if (!fileInput.files.length) {
        return
    }

    showLoading();

    const feedback = document.getElementById("mass-label-feedback");
    const file = fileInput.files[0];
    parseClipIDsFromFile(file)
        .then(clipIDs => {
            if (clipIDs.length === 0) {
                return;
            }
            feedback.textContent = `📄 Loaded ${clipIDs.length} clip ID(s). Applying label "${label}"...`;

            const req = new XMLHttpRequest();
            req.addEventListener("error", () => {
                feedback.textContent = "❌ Error communicating with the server";
            });
            req.addEventListener("load", () => {
                if (req.status !== 200) {
                    feedback.textContent = `❌ Server responded with ${req.status}`;
                    return;
                }
                search();
                hideLoading();
                // Show success directly under the button
                feedback.textContent = `✅ Applied label "${label}" to ${clipIDs.length} clip(s).`;
                feedback.style.display = "block";
                fileInput.value = "";
                document.getElementById("mass-label-value").value = "";
            });
            const payloadText = `mass_label::${label}::${clipIDs.join(",")}::${projectToWrite}`;
            req.open("POST", "");
            req.setRequestHeader("Content-Type", "text/plain");
            req.send(payloadText);
        })
        .catch(err => {
            feedback.textContent = `❌ ${err.message}`;
            hideLoading();
        });
}

function parseClipIDsFromFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();

        reader.onload = function (e) {
            const content = e.target.result;
            const ext = file.name.split(".").pop().toLowerCase();
            let clipIDs = [];

            try {
                if (ext === "json") {
                    const json = JSON.parse(content);
                    clipIDs = Array.isArray(json) ? json : json.clip_ids || [];
                } else if (ext === "csv") {
                    clipIDs = content.split(/\r?\n/).map(line => line.split(",")[0].trim()).filter(Boolean);
                } else if (ext === "txt") {
                    clipIDs = content.split(/\r?\n/).map(line => line.trim()).filter(Boolean);
                } else {
                    reject(new Error("Unsupported file type"));
                    return;
                }

                resolve(clipIDs);
            } catch (err) {
                reject(new Error("Error parsing file"));
            }
        };

        reader.onerror = () => reject(new Error("Failed to read file"));

        reader.readAsText(file);
    });
}

function parseAnnotationsFromFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onerror = () => reject(new Error("Failed to read file"));
        reader.onload = (e) => {
            try {
                const text = String(e.target.result || "");
                const ext = (file.name.split('.').pop() || '').toLowerCase();
                if (ext !== 'txt') {
                    reject(new Error("Unsupported file type (TXT)"));
                    return;
                }

                const lines = text.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
                if (lines.length === 0) {
                    resolve({ clipIds: [], keys: [], startTimes: [], endTimes: [], values: [] });
                    return;
                }

                const clipIds = [];
                const keys = [];
                const startTimes = [];
                const endTimes = [];
                const values = [];

                for (let i = 0; i < lines.length; i++) {
                    const parts = lines[i].split(',').map(s => s.trim());

                    if (parts.length === 4) {
                        // manual: clip_id, key, start_time, end_time
                        const [clipId, key, stRaw, etRaw] = parts;
                        if (!clipId || !key) {
                            continue;
                        }
                        const st = parseFloat(stRaw);
                        const et = parseFloat(etRaw);
                        clipIds.push(clipId);
                        keys.push(key);
                        // In case there is a negative value, Nan, undefined etc replace with -1
                        startTimes.push(Number.isFinite(st) && st >= 0 ? st : -1);
                        endTimes.push(Number.isFinite(et) && et >= 0 ? et : -1);
                        // manual labels have no numeric value
                        values.push(null);
                    } else if (parts.length === 3) {
                        // numeric: clip_id, key, value
                        const [clipId, key, vRaw] = parts;
                        if (!clipId || !key) {
                            continue;
                        }
                        const v = parseFloat(vRaw);
                        // Skip for invalid numeric value
                        if (!Number.isFinite(v)) {
                            continue;
                        }
                        clipIds.push(clipId);
                        keys.push(key);
                        startTimes.push(-1);
                        endTimes.push(-1);
                        values.push(v);
                    } else {
                        // ignore malformed lines
                        continue;
                    }
                }

                resolve({ clipIds, keys, startTimes, endTimes, values });
            } catch (err) {
                reject(new Error("Error parsing file"));
            }
        };
        reader.readAsText(file);
    });
}

function uploadAnnotations() {
    const projectToWrite = document.getElementById("save-project-name").value.trim();
    if (!validateProjectName()) {
        if (!getProjectToWrite()) {
            logMissingProjectToWrite("uploading annotations");
            return;
        }
    }

    const input = document.getElementById("upload-annotations-file");
    if (!input.files.length) {
        return;
    }

    showLoading();
    const file = input.files[0];
    console.log(file);

    const feedback = document.getElementById("upload-annotations-feedback");
    parseAnnotationsFromFile(file)
        .then(({ clipIds, keys, startTimes, endTimes, values }) => {
            if (clipIds.length === 0) {
                hideLoading();
                if (feedback) {
                    feedback.textContent = "❌ No valid annotations found in file";
                    feedback.style.display = 'block';
                }
                return;
            }

            const req = new XMLHttpRequest();
            req.addEventListener("error", () => {
                if (feedback) {
                    feedback.textContent = "❌ Error communicating with the server";
                    feedback.style.display = 'block';
                }
                hideLoading();
            });
            req.addEventListener("load", () => {
                if (req.status !== 200) {
                    if (feedback) {
                        feedback.textContent = `❌ Server responded with ${req.status}`;
                        feedback.style.display = 'block';
                    }
                    hideLoading();
                    return;
                }
                search();
                hideLoading();
                if (feedback) {
                    feedback.textContent = `✅ Uploaded ${clipIds.length} annotation row(s).`;
                    feedback.style.display = 'block';
                }
                input.value = "";
            });

            // Note: server-side handler currently needs aligning. Payload sends columns separately.
            const payload = `upload_annotations::${clipIds.join(',')}::${keys.join(',')}::${startTimes.join(',')}::${endTimes.join(',')}::${projectToWrite}::${values.map(v => (v == null ? '' : v)).join(',')}`;
            req.open("POST", "");
            req.setRequestHeader("Content-Type", "text/plain");
            req.send(payload);
        })
        .catch(err => {
            if (feedback) {
                feedback.textContent = `❌ ${err.message || String(err)}`;
                feedback.style.display = 'block';
            }
            hideLoading();
        });
}

function uploadCaptions() {
    const fileInput = document.getElementById("upload-captions-file");
    const file = fileInput.files[0];
    if (!file) {
        return;
    }

    const modelName = document.getElementById("model-name").value.trim();
    const datasetName = document.getElementById("dataset-name").value;

    const feedback = document.getElementById("upload-captions-feedback");

    const ext = (file.name.split('.').pop() || '').toLowerCase();
    if (ext !== "parquet") {
        feedback.textContent = "❌ Only parquet supported here";
        feedback.style.display = "block";
        return;
    }

    showLoading();
    const reader = new FileReader();

    reader.onerror = function () {
        // Ensure loading spinner is cleared on file read errors
        hideLoading();
        feedback.textContent = "❌ Failed to read parquet file";
        feedback.style.display = "block";
    };

    reader.onload = function (e) {
        try {
            const arrayBuffer = e.target.result;
            const bytes = new Uint8Array(arrayBuffer);

            // Convert bytes → base64 (text)
            let binary = "";
            const len = bytes.length;
            for (let i = 0; i < len; i++) {
                binary += String.fromCharCode(bytes[i]);
            }
            const base64Data = btoa(binary);

            const payload = `upload_captions::${modelName}::${datasetName}::${base64Data}`;

            const req = new XMLHttpRequest();

            req.addEventListener("error", () => {
                hideLoading();
                feedback.textContent = "❌ Error communicating with the server while uploading captions";
                feedback.style.display = "block";
            });

            req.addEventListener("load", () => {
                if (req.status !== 200) {
                    hideLoading();
                    feedback.textContent = `❌ Server responded with ${req.status} while uploading captions`;
                    feedback.style.display = "block";
                    return;
                }
                feedback.textContent = "✅ Uploaded captions from parquet file";
                feedback.style.display = "block";
                fileInput.value = "";
                hideLoading();
            });

            req.open("POST", "");
            req.setRequestHeader("Content-Type", "text/plain");
            req.send(payload);
        } catch (err) {
            hideLoading();
            feedback.textContent = "❌ Failed to prepare upload: " + String(err);
            feedback.style.display = "block";
        }
    };

    reader.readAsArrayBuffer(file);
}

function renderLabelManipulationButtons() {
    const oldLabel = document.getElementById("rename-from-label").value;
    let newLabel = document.getElementById("rename-to-label").value.trim();
    const renameLabelButton = document.getElementById("rename-label-button");
    renameLabelButton.disabled = true;
    if (oldLabel && newLabel && oldLabel !== newLabel) {
        renameLabelButton.disabled = false;
    }

    const selectElement = document.getElementById("merge-labels-from");
    const selectedValues = [];

    // Iterate through all options to find the selected ones
    for (let i = 0; i < selectElement.options.length; i++) {
        if (selectElement.options[i].selected) {
            selectedValues.push(selectElement.options[i].value);
        }
    }
    newLabel = document.getElementById("merge-labels-to").value.trim();
    const mergeLabelButton = document.getElementById("merge-label-button");
    mergeLabelButton.disabled = true;
    if (newLabel && selectedValues.length > 0) {
        mergeLabelButton.disabled = false;
    }

    const labelToDelete = document.getElementById("delete-label").value;
    const deleteLabelButton = document.getElementById("delete-label-button");
    deleteLabelButton.disabled = true;
    if (labelToDelete) {
        deleteLabelButton.disabled = false;
    }

    const labelValue = document.getElementById("mass-label-value").value;
    const fileInput = document.getElementById("mass-label-clipid-file");
    const massLabelButton = document.getElementById("mass-label-button");
    massLabelButton.disabled = true;
    if (labelValue && fileInput && fileInput.files && fileInput.files.length > 0) {
        massLabelButton.disabled = false;
    }

    const uploadAnnotationsButton = document.getElementById("upload-annotations-button");
    uploadAnnotationsButton.disabled = true;
    const annFileInput = document.getElementById("upload-annotations-file");
    if (annFileInput.files.length > 0) {
        uploadAnnotationsButton.disabled = false;
    }

    const uploadCaptionsButton = document.getElementById("upload-captions-button");
    uploadCaptionsButton.disabled = true;

    const selectedElem = document.getElementById("dataset-name");
    const selectedDatasets = [];

    // Iterate through all options to find the selected ones
    for (let i = 0; i < selectedElem.options.length; i++) {
        if (selectedElem.options[i].selected) {
            selectedDatasets.push(selectedElem.options[i].value);
        }
    }
    const modelName = document.getElementById("model-name").value.trim();
    const captFileInput = document.getElementById("upload-captions-file");
    if (modelName && captFileInput.files.length > 0 && selectedDatasets.length > 0) {
        uploadCaptionsButton.disabled = false;
    }
}

function addOptionsForLabelManipulation(options) {
    let labelToRename = document.getElementById("rename-from-label");
    labelToRename.innerHTML = "";
    defaultOpt = document.createElement("option");
    defaultOpt.text = "Select label";
    defaultOpt.value = "";
    labelToRename.add(defaultOpt);
    addOptionsToSelect(labelToRename, options, null);
    if ($(labelToRename).data("select2")) {
        $(labelToRename).select2("destroy");
    }
    $(labelToRename).select2({
        placeholder: "Select label to rename...",
        allowClear: true,
        width: "100%"
    });

    let labelToDelete = document.getElementById("delete-label");
    labelToDelete.innerHTML = "";
    defaultOpt = document.createElement("option");
    defaultOpt.text = "Select label";
    defaultOpt.value = "";
    labelToDelete.add(defaultOpt);
    addOptionsToSelect(labelToDelete, options, null);
    if ($(labelToDelete).data("select2")) {
        $(labelToDelete).select2("destroy");
    }
    $(labelToDelete).select2({
        placeholder: "Select label to delete...",
        allowClear: true,
        width: "100%"
    });

    let labelsToMerge = document.getElementById("merge-labels-from");
    labelsToMerge.innerHTML = "";
    defaultOpt = document.createElement("option");
    defaultOpt.text = "Select labels to merge...";
    defaultOpt.value = "";
    labelsToMerge.add(defaultOpt);
    addOptionsToSelect(labelsToMerge, options, null);
    if ($(labelsToMerge).data("select2")) {
        $(labelsToMerge).select2("destroy");
    }
    $(labelsToMerge).select2({
        placeholder: "Select labels to merge...",
        allowClear: true,
        multiple: true,
        width: "100%"
    });
}

function addOptionsForDatasetSelection(options) {
    let datasetName = document.getElementById("dataset-name");
    datasetName.innerHTML = "";
    defaultOpt = document.createElement("option");
    defaultOpt.text = "Select dataset...";
    defaultOpt.value = "";
    datasetName.add(defaultOpt);
    addOptionsToSelect(datasetName, options, null);
    if ($(datasetName).data("select2")) {
        $(datasetName).select2("destroy");
    }
    $(datasetName).select2({
        placeholder: "Select label to rename...",
        allowClear: true,
        width: "100%",
        templateResult: datasetTemplateResult,
        templateSelection: datasetTemplateResult,
    });
}


function showCommentsBox(clipId) {
    console.log(clipId);
    const commentsBox = document.getElementById("comments-box");
    commentsBox.style.display = "flex";
    // Clear any previous content from the comments box
    commentsBox.innerHTML = '';

    // Create the content div
    const commentsContent = document.createElement('div');
    commentsContent.className = 'comments-content';

    // Create heading
    const heading = document.createElement('h3');
    heading.textContent = 'Add Comments or Write your Caption';
    commentsContent.appendChild(heading);

    // Create the textarea
    const textareaElement = document.createElement('textarea');
    textareaElement.className = 'comments-text-area';
    textareaElement.id = 'comments-text-area';
    textareaElement.type = 'text';
    textareaElement.name = 'txtarea';
    textareaElement.maxLength = 1000000;
    textareaElement.rows = 10;
    textareaElement.cols = 50;
    commentsContent.appendChild(textareaElement);

    // Create the close button (X)
    const closeButton = document.createElement('span');
    closeButton.className = 'clear-annotation-button';
    closeButton.innerHTML = '&times;';
    closeButton.onclick = hideCommentsBox;
    commentsContent.appendChild(closeButton);

    // Create the button container div
    const buttonDiv = document.createElement('div');
    commentsContent.appendChild(buttonDiv);

    // Create the update button
    const updateButton = document.createElement('button');
    updateButton.className = 'update-comments-button';
    updateButton.id = 'update-comments-button';
    updateButton.textContent = 'Update';

    updateButton.onclick = function () {
        updateComments(clipId);
    };
    buttonDiv.appendChild(updateButton);

    commentsBox.appendChild(commentsContent);

    const videoData = window.currentVideos.find(v => v.annotations.clip_id === clipId);
    if (videoData && videoData.comments) {
        textareaElement.value = videoData.comments;
    } else {
        textareaElement.value = "";
    }
}

function hideCommentsBox() {
    document.getElementById("comments-box").style.display = "none";
    document.getElementById("comments-box").innerHTML = '';
}

function updateComments(video_id) {
    const comment = document.getElementById("comments-text-area").value;
    const req = new XMLHttpRequest();
    req.addEventListener("error", () => {
        console.log("Error communicating with server for label manipulation!");
    });
    req.addEventListener("load", () => {
        if (req.status !== 200) {
            console.log("Failed to perform label operation on server!");
            return;
        }
        search();
    });
    req.open("POST", "");
    req.setRequestHeader("Content-Type", "text/plain");
    req.send(`update_comment::${video_id}::${comment}`);
    hideCommentsBox();
}


function toggleDisplayMenu() {
    const button = document.querySelector('.gray-button[onclick="toggleDisplayMenu()"]');
    const menu = document.getElementById('display-group');
    button.classList.toggle('expanded');
    menu.classList.toggle('expanded');
}

function toggleLabelManipulationMenu() {
    const button = document.querySelector('.gray-button[onclick="toggleLabelManipulationMenu()"]');
    const menu = document.getElementById('label-manipulation-group');
    button.classList.toggle('expanded');
    menu.classList.toggle('expanded');

    // Only show feedback when there is content
    const feedback = document.getElementById("mass-label-feedback");
    const hasText = feedback && feedback.textContent.trim().length > 0;
    if (feedback) {
        feedback.style.display = (menu.classList.contains('expanded') && hasText) ? 'block' : 'none';
    }

    // Only show feedback when there is content
    const annFeedback = document.getElementById("upload-annotations-feedback");
    const hasTextFeedback = annFeedback && annFeedback.textContent.trim().length > 0;
    if (annFeedback) {
        annFeedback.style.display = (menu.classList.contains('expanded') && hasTextFeedback) ? 'block' : 'none';
    }
}

function toggleLabelVisibility(type) {
    window.visibleLabelTypes[type] = !window.visibleLabelTypes[type];
    const eye = document.getElementById(`eye-${type}`);

    // Update classes based on the new visibility state
    if (window.visibleLabelTypes[type]) {
        eye.classList.remove("hidden");
        eye.classList.add("visible");
    } else {
        eye.classList.remove("visible");
        eye.classList.add("hidden");
    }

    render();
}

function toggleVideoControlsMenu() {
    const button = document.querySelector('.gray-button[onclick="toggleVideoControlsMenu()"]');
    const menu = document.getElementById('video-controls-group');
    button.classList.toggle('expanded');
    menu.classList.toggle('expanded');
}


document.addEventListener('DOMContentLoaded', function () {

    window.currentVideosPerPage = 6;

    const gotoInput = document.getElementById("goto-page-input");
    gotoInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter") {
            goToPage(this.value);
            this.blur();
        }
    });
    gotoInput.addEventListener("blur", function () {
        goToPage(this.value);
    });

    const vppSlider = document.getElementById("videos-per-page");
    const vppDisplay = document.getElementById("videos-per-page-display");
    vppSlider.addEventListener("input", function () {
        vppDisplay.textContent = VPP_VALUES[parseInt(this.value)];
    });
    vppSlider.addEventListener("change", function () {
        const val = VPP_VALUES[parseInt(this.value)];
        window.currentVideosPerPage = val;
        search();
    });

    bindModeControl("filter-mode-row", "currentFilterMode");
    bindModeControl("data-source-mode-row", "currentDataSourceMode");
    bindModeControl("labels-to-exclude-mode-row", "currentLabelsToExcludeMode");

    initClassifierPanel({canTrain: true});

    document.getElementById("rename-to-label").addEventListener("input", renderLabelManipulationButtons);
    document.getElementById("merge-labels-to").addEventListener("input", renderLabelManipulationButtons);
    document.getElementById("mass-label-value").addEventListener("input", renderLabelManipulationButtons);

    document.getElementById("autolabel-label").addEventListener("input", changeAutolabelButtons);
    document.getElementById("upload-annotations-file").addEventListener("change", renderLabelManipulationButtons);

    document.getElementById("model-name").addEventListener("input", renderLabelManipulationButtons);
    document.getElementById("dataset-name").addEventListener("change", renderLabelManipulationButtons);
    document.getElementById("upload-captions-file").addEventListener("change", renderLabelManipulationButtons);

    changeAutolabelButtons();

    renderSectors();

    updateSelectedAngles();

    // Make angle selector functions globally accessible for reset
    window.wmAngleSelector = {
        renderSectors: renderSectors,
        updateSelectedAngles: updateSelectedAngles
    };

    // Set initial eye icon states
    for (const type in window.visibleLabelTypes) {
        const eye = document.getElementById(`eye-${type}`);
        if (eye) {
            if (window.visibleLabelTypes[type]) {
                eye.classList.add("visible");
            } else {
                eye.classList.add("hidden");
            }
        }
    }

    // Initialize trajectories eye icon based on global toggle
    const eyeTraj = document.getElementById('eye-trajectories');
    if (eyeTraj) {
        if (window.showAllTrajectories) {
            eyeTraj.classList.add('visible');
        } else {
            eyeTraj.classList.add('hidden');
        }
    }

    // Initialize metrics eye icon based on global toggle
    const eyeMetrics = document.getElementById('eye-metrics');
    if (eyeMetrics) {
        if (window.showAllMetrics) {
            eyeMetrics.classList.add('visible');
        } else {
            eyeMetrics.classList.add('hidden');
        }
    }

    // Initialize BEV eye icon based on global toggle
    const eyeBEV = document.getElementById('eye-bev');
    if (eyeBEV) {
        if (window.showAllBEV) {
            eyeBEV.classList.add('visible');
        } else {
            eyeBEV.classList.add('hidden');
        }
    }
});

function showQuickLabelsHelp() {
    toggleHelp("quick-labels-help-content", true);
}

function hideQuickLabelsHelp() {
    toggleHelp("quick-labels-help-content", false);
}

function showMassLabelHelp() {
    toggleHelp("mass-label-help-content", true);
}

function hideMassLabelHelp() {
    toggleHelp("mass-label-help-content", false);
}

function showUploadAnnotationsHelp() {
    toggleHelp("upload-annotations-help-content", true);
}

function hideUploadAnnotationsHelp() {
    toggleHelp("upload-annotations-help-content", false);
}

function showUploadCaptionsHelp() {
    toggleHelp("upload-captions-help-content", true);
}

function hideUploadCaptionsHelp() {
    toggleHelp("upload-captions-help-content", false);
}

// Clustering globals and functions live in clustering.js (shared with leaderboard).

// Event listener for Escape key
document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
        closeCaptionBoxes();
        hideCommentsBox();
        hideSpeedSearchHelp();
        hideCaptionSearchHelp();
        hideSemanticSearchHelp();
        hideVisualSearchHelp();
        hideVisualSearchImageHelp();
        hideCaptionEmbedSearchHelp();
        hideVideoToVideoSearchHelp();
        hideTrajectoryShapeSearchHelp();
        hideCommentSearchHelp();
        hideMassLabelHelp();
        hideUploadAnnotationsHelp();
        hideUploadCaptionsHelp();
        hideQuickLabelsHelp();
        hideClassifierMenuHelp();
        hideClusteringHelp();
        hideClosestClustersHelp();
        hideWMSearchHelp();
        hideVlmCheckHelp();
        hideVlmJudgeQueryBox();
        hideAutolabelWarning();
    } else if (event.key == "Enter" && event.target.closest(".sidebar")) {
        search();
    } else if ((event.key === "ArrowRight" || event.key === "ArrowLeft") &&
        !event.target.closest("input, textarea, select")) {
        if (event.key === "ArrowRight") nextPage();
        else prevPage();
    }
});

document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("filters").addEventListener("input", function (e) {
        if (e.target.id === "search-clipid") {
            window._clipIDTimestamp = Date.now();
        } else {
            window._otherFilterTimestamp = Date.now();
        }
    });
});
