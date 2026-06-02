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

function renderFilters() {
    let options = window.currentOptions;

    let projectSourceOptions = window.currentProjectOptions;
    addProjectSourceOptions(projectSourceOptions, window.currentProjectSource);
}

function renderMetrics() {
    let tabContainer = document.getElementById("tab-content-container");
    tabContainer.innerHTML = "";

    let metrics = window.currentMetrics;
    // Expecting metrics as: [{ modelName: { metrics: [names], values: [[...]], clips: [clip] } }, ...]
    if (metrics.length === 0) {
        const info = document.createElement("div");
        info.textContent = "No metrics available for this clip.";
        tabContainer.appendChild(info);
        return;
    }

    // Pivot per-model metrics into rows and collect union of metric names
    const hidden = new Set(["gt_category", "question", "pred_reasoning", "summary"]);
    const allMetricNames = [];
    const seenMetric = new Set();
    const rows = [];
    const summaries = []; // { model, text } for models with a summary field

    metrics.forEach((entry) => {
        const modelName = Object.keys(entry || {})[0];
        if (!modelName) return;
        const data = entry[modelName] || {};
        const mNames = Array.isArray(data.metrics) ? data.metrics : [];
        const values = Array.isArray(data.values) ? data.values : [];
        // values is NxM; for a single clip it should be 1xM. Fallback to 1D.
        const rowVals = Array.isArray(values[0]) ? values[0] : values;

        const row = { model: modelName };
        mNames.forEach((name, i) => {
            row[name] = rowVals?.[i];
            if (name === "summary" && rowVals?.[i] != null && rowVals[i] !== "") {
                summaries.push({ model: modelName, text: String(rowVals[i]) });
            }
            if (!hidden.has(name) && !seenMetric.has(name)) {
                allMetricNames.push(name);
                seenMetric.add(name);
            }
        });
        rows.push(row);
    });

    if (rows.length === 0 || allMetricNames.length === 0) {
        const info = document.createElement("div");
        info.textContent = "No metrics found to display.";
        tabContainer.appendChild(info);
        return;
    }

    // Format numeric cells similar to leaderboard (integers unformatted, else 6 decimals)
    const formattedRows = rows.map((r) => {
        const out = { model: r.model };
        allMetricNames.forEach((m) => {
            let v = r[m];
            if (typeof v === "number" && Number.isFinite(v)) {
                out[m] = Number.isInteger(v) ? Math.round(v) : Number(v).toFixed(6);
            } else if (v === undefined || v === null || v === "") {
                out[m] = "N/A";
            } else {
                out[m] = v;
            }
        });
        return out;
    });

    // Build table
    const tableId = "predictions-metrics-table";
    const tableEl = document.createElement("table");
    tableEl.id = tableId;
    tableEl.className = "display compact";
    tableEl.style.width = "100%";
    tabContainer.appendChild(tableEl);

    function escapeHtml(s) {
        if (s == null) return '';
        return String(s).replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;','\'':'&#39;'}[c]));
    }

    // Escape HTML and convert newlines to <br> for display in table cells
    function renderStringCell(s) {
        if (s == null) return '';
        return escapeHtml(String(s)).replace(/\n/g, '<br>');
    }

    // Define columns: Model + metrics
    const columns = [
        {
            title: "Model",
            data: "model",
            className: "model-col",
            render: function (data, type, row) {
                const text = data == null ? '' : String(data);
                return `<span class="model-text" title="${text}">${text}</span>`;
            }
        }
    ];
    allMetricNames.forEach((m) => {
        columns.push({
            title: m.replace(/_/g, ' '),
            data: m,
            render: function(data, type) {
                if (type !== 'display' || typeof data !== 'string') return data;
                return renderStringCell(data);
            }
        });
    });

    // Initialize DataTable
    const dt = $(`#${tableId}`).DataTable({
        data: formattedRows,
        columns: columns,
        paging: false,
        ordering: true,
        info: false,
        searching: false,
        scrollX: true,
        autoWidth: false,
    });

    // Keep header/body aligned when container resizes
    $(window).off(`resize.${tableId}`).on(`resize.${tableId}`, () => {
        dt.columns.adjust();
    });

    if (summaries.length > 0) {
        const summarySection = document.createElement("div");
        summarySection.style.marginTop = "16px";
        summaries.forEach(({ model, text }) => {
            const block = document.createElement("div");
            block.style.marginBottom = "12px";

            const label = document.createElement("div");
            label.style.fontWeight = "bold";
            label.style.marginBottom = "4px";
            label.textContent = model;

            const pre = document.createElement("pre");
            pre.style.margin = "0";
            pre.style.padding = "10px 12px";
            pre.style.background = "#f7f7f7";
            pre.style.border = "1px solid #ddd";
            pre.style.borderRadius = "6px";
            pre.style.whiteSpace = "pre-wrap";
            pre.style.wordBreak = "break-word";
            pre.style.fontSize = "13px";
            pre.textContent = text;

            block.appendChild(label);
            block.appendChild(pre);
            summarySection.appendChild(block);
        });
        tabContainer.appendChild(summarySection);
    }
}

function render() {
    document.getElementById("annotations-count").innerText = window.annotationsCount;
    document.getElementById("manual-annotations-count").innerText = window.manualAnnotationsCount;
    document.getElementById("autolabel-annotations-count").innerText = window.autolabelAnnotationsCount;

    let videos = window.currentVideos;
    let options = window.currentOptions;

    let currentPageDiv = document.getElementById("current-page");
    currentPageDiv.innerHTML = ""; // Clear existing tiles
    for (let i = 0; i < videos.length; i++) {
        let newTile = makeVideoTile(videos[i], options);
        currentPageDiv.appendChild(newTile);
    }

    // Render the metrics
    renderMetrics();

    // Render metrics time series chart
    if (window.currentFullMetrics && window.currentFullMetrics.timestamps) {
        renderMetricsTimeSeries();
    }

    // Ensure a dedicated grid container exists for prediction plots
    let predGrid = document.getElementById("predictions-grid");
    if (!predGrid) {
        predGrid = document.createElement("div");
        predGrid.id = "predictions-grid";
        predGrid.className = "predictions-grid";
        currentPageDiv.parentNode.insertBefore(predGrid, currentPageDiv.nextSibling);
    }
    predGrid.innerHTML = "";

    // Render the predictions
    let predictions = window.currentPredictions;
    for (let i = 0; i < predictions.length; i++) {

        const gt = predictions[i]["gt_positions"]; 
        const pred = predictions[i]["pred_positions"]; 
        const caps = predictions[i]["pred_captions"];

        if (gt.length > 0 && pred.length > 0) {
            const tileWithTrajectory = buildPredictionsTileWithTrajectory(
                predictions[i]
            );
            predGrid.appendChild(tileWithTrajectory);
            renderPredictionPlot(tileWithTrajectory, predictions[i]);
        } 
        if (caps.length > 0) {
            const tile = buildPredictionsTile(predictions[i]);
            predGrid.appendChild(tile);
            renderPredictedCaptions(tile, caps);
        }
    }

    // Render the filters
    renderFilters();
}

function showPage(
    clipId,
    projectSource,
) {
    let basePath = "/predictions?";
    let path = encodeQuery(
        {
            clip_id: clipId,
            project_source: projectSource
        },
        basePath
    );

    let hash = "#" + path.substr(basePath.length);
    if (window.location.hash != hash) {
        history.pushState(null, "", hash);
    }
    document.getElementById("loading-block").style.display = "flex";
    fetch(path).then(function (response) {
        return response.json();
    }).then(function (data) {

        window.annotationsCount = data.annotations_count;
        window.manualAnnotationsCount = data.manual_annotations_count;
        window.autolabelAnnotationsCount = data.autolabel_annotations_count;
        window.currentProjectOptions = data.project_options;
        window.currentProjectSource = data.project_source;
        window.currentPredictions = data.predictions;
        window.currentMetrics = data.metrics;
        window.currentOptions = data.options;
        window.currentVideos = data.videos;
        window.currentClipId = data.clip_id;
        window.vlmJudgeAvailable = data.vlm_judge_available || false;

        // Fetch full metrics time series data
        const fullMetricsPath = `/full_metrics?clip_id=${encodeURIComponent(clipId)}&model_name=ground_truth`;
        return fetch(fullMetricsPath);
    }).then(function (response) {
        return response.json();
    }).then(function (fullMetricsData) {
        window.currentFullMetrics = fullMetricsData;
        
        render();
        document.getElementById("loading-block").style.display = "none";
    }).catch(function(error) {
        console.error("Error loading data:", error);
        document.getElementById("loading-block").style.display = "none";
    });
}

function search() {
    let projectSource = document.getElementById("project-select");
    let selectedProjectSource = Array.from(projectSource.selectedOptions).map(option => option.value);
    selectedProjectSource = selectedProjectSource.join("||")

    showPage(
        window.currentClipId,
        (selectedProjectSource != "") ? selectedProjectSource : null,
    );
}

function buildPredictionsTile(predictions) {
    const modelName = predictions["model_name"];

    const pred_tile = document.getElementById("predictions-tile-template").cloneNode(true);
    pred_tile.id = "pred-tile-" + modelName;
    pred_tile.style.display = "";

    // Add the title name
    let modelnNameSpan = pred_tile.querySelector(".model-name-container span");
    const full = String(modelName ?? "");
    const MAX = 50;
    const display = full.length > MAX ? (full.slice(0, MAX - 1) + "…") : full;
    modelnNameSpan.textContent = display;
    modelnNameSpan.title = full; // show full name on hover
    return pred_tile;
}

function buildPredictionsTileWithTrajectory(predictions) {
    const pred_tile = buildPredictionsTile(predictions);
    const modelName = predictions["model_name"];

    const plotDiv = document.createElement("div");
    plotDiv.className = "trajectory-plot";
    plotDiv.dataset.clipId = modelName;
    plotDiv.style.display = "block"; // ensure it’s visible
    pred_tile.appendChild(plotDiv);
    return pred_tile;
}

function renderPredictedCaptions(tileEl, caption) {
    const raw = Array.isArray(caption) ? caption.join("\n") : (caption || "");
    const text = raw.trim();
    const capBox = document.createElement("div");
    capBox.className = "pred-captions";
    capBox.style.margin = "8px 12px"; // vertical + horizontal
    capBox.style.padding = "8px";
    capBox.style.background = "#f7f7f7";
    capBox.style.border = "1px solid #ddd";
    capBox.style.borderRadius = "6px";
    capBox.style.whiteSpace = "pre-wrap"; // keep line breaks if any
    capBox.style.overflowY = "auto"; // enable scrolling when needed

    capBox.textContent = text;
    tileEl.appendChild(capBox);

    const TILE_HEIGHT = 400; 

    function sizeCaptionBox() {
        // Fixed overall tile height
        tileEl.style.height = `${TILE_HEIGHT}px`;

        // Compute remaining height for captions after header/legend/plot and padding
        const header = tileEl.querySelector('.model-name-container');
        const legend = tileEl.querySelector('.legend-container');

        const headerH = header ? header.offsetHeight : 0;
        const legendH = legend ? legend.offsetHeight : 0;

        const styles = window.getComputedStyle(tileEl);
        const padTop = parseFloat(styles.paddingTop || '0');
        const padBot = parseFloat(styles.paddingBottom || '0');
        const verticalPadding = padTop + padBot;

        // capBox has symmetric vertical margins (8px top + 8px bottom)
        const capMargins = 16;

        const available = TILE_HEIGHT - (headerH + legendH + verticalPadding + capMargins);
        const capHeight = Math.max(60, available);
        capBox.style.height = `${capHeight}px`;
        capBox.style.maxHeight = `${capHeight}px`;
    }

    // Size once now (after DOM insertion)
    sizeCaptionBox();
    // And keep it responsive on resize
    window.addEventListener('resize', sizeCaptionBox);
}

function renderPredictionPlot(tileEl, predictions) {
    const modelName = predictions["model_name"];
    const plotDiv = tileEl.querySelector(".trajectory-plot");
    const margin = { top: 20, right: 20, bottom: 40, left: 50 };
    const w = plotDiv.clientWidth  - margin.left - margin.right;
    const h = plotDiv.clientHeight - margin.top  - margin.bottom;

    const width  = (w > 0 ? w : 600);
    const height = (h > 0 ? h : 300);

    d3.select(plotDiv).html("");

    const titleText = `Predictions Model ${modelName ?? ""}`.trim();
    drawTrajectoryPlot(
        predictions["gt_positions"],
        predictions["pred_positions"],
        plotDiv
    );
}

function normPoint(p) {
    if (!Array.isArray(p) || p.length < 3) {
        return null;
    }
    return { t: +p[0], x: +p[1], y: +p[2] };
}

function drawTrajectoryPlot(gtPositions, predPositions, plotDiv, titleText) {
    const gtData = (gtPositions || []).map(normPoint).filter(Boolean);
    const predData = (predPositions || []).map(
        (win) => (win || []).map(
          (traj) => (traj || []).map(normPoint).filter(Boolean)
        )
    );

    const margin = { top: 20, right: 20, bottom: 40, left: 50 };
    const width = plotDiv.clientWidth - margin.left - margin.right;
    const height = plotDiv.clientHeight - margin.top - margin.bottom;

    d3.select(plotDiv).html(""); // Clear previous SVG content

    const mainSvg = d3.select(plotDiv)
        .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom);

    const title = (titleText || "").trim();
    if (title) {
        mainSvg.append("text")
            .attr("x", (width + margin.left + margin.right) / 2)
            .attr("y", 16)
            .attr("text-anchor", "middle")
            .attr("class", "plot-title")
            .text(title);
    }

    const gElement = mainSvg.append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // Canonical ego frame: d.x = forward, d.y = right (so left = -y).
    // Plot forward on the vertical axis (up = forward) and lateral on the
    // horizontal axis (right = +y).
    let lateralExtent = d3.extent(gtData, d => d.y);
    let lateralMean = (lateralExtent[0] + lateralExtent[1]) / 2;
    let lateralSize = lateralExtent[1] - lateralExtent[0];

    let forwardExtent = d3.extent(gtData, d => d.x);
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

    const line = d3.line()
        .x(d => xScale(d.y))
        .y(d => yScale(d.x))
        .curve(d3.curveCatmullRom.alpha(0.5));

    const gMainPlot = gElement.append("g");

    const path = gMainPlot.append("path")
        .datum(gtData)
        .attr("fill", "none")
        .attr("stroke", "blue")
        .attr("stroke-width", 2)
        .attr("d", line);

    const pathNode = path.node();
    const pathTotalLength = pathNode ? pathNode.getTotalLength() : 0;

    gMainPlot.selectAll(".trajectory-point")
        .data(gtData)
        .enter()
        .append("circle")
        .attr("class", "trajectory-point")
        .attr("cx", d => xScale(d.y))
        .attr("cy", d => yScale(d.x))
        .attr("r", 3)
        .attr("fill", "red");

    const end = gtData[gtData.length - 1];
    if (end) {
        gMainPlot.append("rect")
            .attr("x", xScale(end.y) - 5)
            .attr("y", yScale(end.x) - 5)
            .attr("width", 10)
            .attr("height", 10)
            .attr("fill", "black");
    }

    const W = predData.length;
    const windowOpacity = d3.scaleLinear()
        .domain([0, Math.max(1, W - 1)])
        .range([0.25, 0.9]);

    const K = predData.reduce((m, win) => Math.max(m, win.length || 0), 0);
    const PALETTE_NO_BR = [
      "#f28e2b", // orange
      "#76b7b2", // teal
      "#59a14f", // green
      "#edc948", // yellow
      "#b07aa1", // purple
      "#ff9da7", // pink
      "#9c755f", // brown
      "#bab0ab"  // gray
    ];
    const HORIZON_SEC = 6.4;

    const colors = d3.scaleOrdinal(PALETTE_NO_BR).domain(d3.range(K));

    const predGroup = gElement.append("g").attr("class", "predictions");
    // active state per trajectory index
    const active = Array.from({ length: K }, () => true);

    predData.forEach((win, wIdx) => {
        const gWin = predGroup.append("g")
            .attr("class", `pred-window-${wIdx}`)
            .attr("opacity", 0);
        const op = windowOpacity(wIdx);

        win.forEach((traj, kIdx) => {
            const gTraj = gWin.append("g").attr("class", `traj-${kIdx}`);

            gTraj.append("path")
                 .datum(traj)
                 .attr("fill", "none")
                 .attr("stroke", colors(kIdx))
                 .attr("stroke-width", 1)
                 .attr("d", line);

            gTraj.selectAll("circle")
                 .data(traj)
                 .join("circle")
                 .attr("cx", d => xScale(d.y))
                 .attr("cy", d => yScale(d.x))
                 .attr("r", 1.5)
                 .attr("fill", colors(kIdx));
        });
    });

    // Legend: populate the surrounding .legend-container instead of SVG legend
    const legendContainer = plotDiv.parentElement?.querySelector?.('.legend-container');
    if (legendContainer) {
        legendContainer.innerHTML = '';
        const ul = document.createElement('div');
        ul.style.display = 'flex';
        ul.style.flexWrap = 'wrap';
        ul.style.gap = '8px';
        ul.style.justifyContent = 'center';

        d3.range(K).forEach((trajIdx) => {
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.style.display = 'inline-flex';
            item.style.alignItems = 'center';
            item.style.cursor = 'pointer';
            item.style.opacity = active[trajIdx] ? '1' : '0.6';

            const swatch = document.createElement('span');
            swatch.className = `legend-swatch swatch-${trajIdx}`;
            swatch.style.display = 'inline-block';
            swatch.style.width = '18px';
            swatch.style.height = '16px';
            swatch.style.borderRadius = '3px';
            swatch.style.background = colors(trajIdx);
            swatch.style.marginRight = '6px';

            const label = document.createElement('span');
            label.textContent = `Traj ${trajIdx}`;
            label.style.fontSize = '12px';

            item.appendChild(swatch);
            item.appendChild(label);

            item.addEventListener('click', () => {
                active[trajIdx] = !active[trajIdx];
                predGroup.selectAll(`.traj-${trajIdx}`).attr('opacity', active[trajIdx] ? 1 : 0);
                swatch.style.opacity = active[trajIdx] ? '1' : '0.3';
                item.style.opacity = active[trajIdx] ? '1' : '0.6';
            });

            ul.appendChild(item);
        });
        legendContainer.appendChild(ul);
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

    // Define the car icon
    const car = gElement.append("text")
        .attr("font-size", "24px")
        .attr("text-anchor", "middle")
        .attr("alignment-baseline", "middle")
        .text("🚗")
        .style("cursor", "grab");

    const videoElement = document.getElementById("video-tile-" + window.currentClipId).querySelector("video");

    function updatePredWindowVisibility(p) {
        // W is the time stamps at which we have a prediction
        const wIdx = Math.min(W - 1, Math.max(0, Math.floor((p ?? 0) * W)));
        predGroup.selectAll(`[class^="pred-window-"]`).attr("opacity", 0);
        predGroup.select(`.pred-window-${wIdx}`).attr("opacity", 1);
    }

    // Update car position based on video time
    function updateCarPosition(time) {
        const duration = videoElement.duration;
        if (isNaN(duration) || duration === 0 || pathTotalLength === 0) {
            car.style("display", "none");
            return;
        }
        car.style("display", "block");

        const percentage = time / duration;

        updatePredWindowVisibility(percentage);

        const pointIndex = Math.floor(percentage * (gtData.length - 1));
        const nextIndex = Math.min(pointIndex + 1, gtData.length - 1);
        const currentPoint = [xScale(gtData[pointIndex].y), yScale(gtData[pointIndex].x)];
        const nextPoint = [xScale(gtData[nextIndex].y), yScale(gtData[nextIndex].x)];
        const prevPerc = pointIndex / (gtData.length - 1);
        const nextPerc = nextIndex / (gtData.length - 1);
        const t = (percentage - prevPerc) / (nextPerc - prevPerc + 1e-6);
        const drawPointX = t * nextPoint[0] + (1-t) * currentPoint[0];
        const drawPointY = t * nextPoint[1] + (1-t) * currentPoint[1];
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
        .on("start", function(event) {
            this.__isDragging__ = true;
            videoElement.pause();
            d3.select(this).style("cursor", "grabbing");
        })
        .on("drag", function(event) {
            // Get the underlying SVG DOM element for coordinate transformation
            const svgDomElement = mainSvg.node();
            const svgPoint = svgDomElement.createSVGPoint();
            svgPoint.x = event.sourceEvent.clientX; // Use clientX/Y from the original event
            svgPoint.y = event.sourceEvent.clientY;

            // Apply the inverse of the SVG's screen CTM and then the inverse of the 'g' element's transform
            // to get coordinates relative to the 'g' element's user space.
            const transformedPoint = svgPoint.matrixTransform(svgDomElement.getScreenCTM().inverse())
                                             .matrixTransform(gElement.node().getCTM().inverse());

            // Find the closest point on the path to the dragged position
            let closestLength = 0;
            let minDistanceSq = Infinity; // Use squared distance for performance (no sqrt needed)

            // Iterate a sufficient number of points along the path for a decent approximation
            // A smaller step (e.g., 0.1 or 0.5) will increase precision but also computation.
            // For typical video trajectories, 1 might be okay, but you can refine if needed.
            const stepSize = 1; // You can adjust this for precision vs. performance

            for (let i = 0; i <= pathTotalLength; i += stepSize) {
                const p = pathNode.getPointAtLength(i);
                const dx = transformedPoint.x - p.x;
                const dy = transformedPoint.y - p.y;
                const distSq = dx * dx + dy * dy; // Euclidean distance squared

                if (distSq < minDistanceSq) {
                    minDistanceSq = distSq;
                    closestLength = i;
                }
            }

            // Optional: Refine the search around the closest point found
            // This can improve accuracy without iterating every single pixel of the path.
            // Search in a small window around the initially found closestLength
            const refinementRange = 5; // Search +/- 5 units around the best point
            const startRefine = Math.max(0, closestLength - refinementRange);
            const endRefine = Math.min(pathTotalLength, closestLength + refinementRange);

            for (let i = startRefine; i <= endRefine; i += 0.1) { // Smaller step for refinement
                const p = pathNode.getPointAtLength(i);
                const dx = transformedPoint.x - p.x;
                const dy = transformedPoint.y - p.y;
                const distSq = dx * dx + dy * dy;

                if (distSq < minDistanceSq) {
                    minDistanceSq = distSq;
                    closestLength = i;
                }
            }

            closestLength = Math.max(0, Math.min(pathTotalLength, closestLength));

            const newCarPosition = pathNode.getPointAtLength(closestLength);

            car.attr("transform", `translate(${newCarPosition.x}, ${newCarPosition.y}) scale(-1, 1)`);

            const newTime = (closestLength / pathTotalLength) * videoElement.duration;
            videoElement.currentTime = newTime;
        })
        .on("end", function() {
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

function renderMetricsTimeSeries() {
    const metricsData = window.currentFullMetrics;
    if (!metricsData || !metricsData.timestamps || !metricsData.metrics) {
        return;
    }

    const container = document.getElementById("metrics-timeseries-chart");
    if (!container) return;

    // Define which metrics to display
    const metricsToShow = ["nc", "ttc", "ttc_gt_traj", "comfort", "progress", "lk", "dac", "ddc", "pdm"];
    const timestamps = metricsData.timestamps;

    // Filter to only metrics that exist in the data and are in our list
    const availableMetrics = metricsToShow.filter(m => metricsData.metrics[m]);

    if (availableMetrics.length === 0) {
        return;
    }

    document.getElementById("metrics-timeseries-heading").style.display = "";
    document.getElementById("metrics-timeseries-container").style.display = "";

    // Clear previous content
    d3.select(container).html("");

    // Set up dimensions
    const margin = { top: 40, right: 150, bottom: 50, left: 60 };
    const width = Math.max(800, container.clientWidth) - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    // Create SVG
    const svg = d3.select(container)
        .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom);

    const g = svg.append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // Title
    svg.append("text")
        .attr("x", (width + margin.left + margin.right) / 2)
        .attr("y", 20)
        .attr("text-anchor", "middle")
        .attr("class", "plot-title")
        .style("font-size", "16px")
        .style("font-weight", "bold")
        .text("Metrics Over Time");

    // Scales
    const xScale = d3.scaleLinear()
        .domain([0, d3.max(timestamps)])
        .range([0, width]);

    // Find global min/max for y-axis across all metrics
    let globalMin = Infinity;
    let globalMax = -Infinity;
    availableMetrics.forEach(metric => {
        const values = metricsData.metrics[metric];
        // Filter out null and NaN values
        const validValues = values.filter(v => v != null && !isNaN(v));
        if (validValues.length > 0) {
            const min = d3.min(validValues);
            const max = d3.max(validValues);
            if (min < globalMin) globalMin = min;
            if (max > globalMax) globalMax = max;
        }
    });

    // Fallback if no valid values found
    if (!isFinite(globalMin) || !isFinite(globalMax)) {
        globalMin = 0;
        globalMax = 1;
    }

    const yScale = d3.scaleLinear()
        .domain([globalMin, globalMax])
        .range([height, 0]);

    // Color scale for different metrics
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10)
        .domain(availableMetrics);

    // Line generator with null handling
    const line = d3.line()
        .defined(d => d != null && !isNaN(d))
        .x((d, i) => xScale(timestamps[i]))
        .y(d => yScale(d));

    // Draw lines for each metric
    const linesGroup = g.append("g").attr("class", "metric-lines");
    availableMetrics.forEach(metric => {
        const values = metricsData.metrics[metric];
        // Filter out null values but keep index alignment
        linesGroup.append("path")
            .datum(values)
            .attr("class", `metric-line metric-${metric}`)
            .attr("fill", "none")
            .attr("stroke", colorScale(metric))
            .attr("stroke-width", 2)
            .attr("d", line);
    });

    // Add axes
    const xAxis = d3.axisBottom(xScale).ticks(10);
    g.append("g")
        .attr("class", "x-axis")
        .attr("transform", `translate(0,${height})`)
        .call(xAxis)
        .append("text")
        .attr("x", width / 2)
        .attr("y", 40)
        .attr("fill", "#000")
        .style("text-anchor", "middle")
        .text("Time (seconds)");

    const yAxis = d3.axisLeft(yScale).ticks(8);
    g.append("g")
        .attr("class", "y-axis")
        .call(yAxis)
        .append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", -height / 2)
        .attr("y", -45)
        .attr("fill", "#000")
        .style("text-anchor", "middle")
        .text("Metric Value");

    // Legend
    const legend = svg.append("g")
        .attr("class", "legend")
        .attr("transform", `translate(${width + margin.left + 10}, ${margin.top})`);

    availableMetrics.forEach((metric, i) => {
        const legendRow = legend.append("g")
            .attr("transform", `translate(0, ${i * 20})`);

        legendRow.append("rect")
            .attr("width", 15)
            .attr("height", 15)
            .attr("fill", colorScale(metric));

        legendRow.append("text")
            .attr("x", 20)
            .attr("y", 12)
            .style("font-size", "12px")
            .text(metric);
    });

    // Add time indicator line that follows video playback
    const timeIndicator = g.append("line")
        .attr("class", "time-indicator")
        .attr("y1", 0)
        .attr("y2", height)
        .attr("stroke", "red")
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "5,5")
        .style("pointer-events", "none")
        .style("display", "none");

    // Get video element
    const videoElement = document.getElementById("video-tile-" + window.currentClipId)?.querySelector("video");
    
    if (videoElement) {
        // Update time indicator based on video time
        function updateTimeIndicator() {
            if (videoElement.duration && videoElement.duration > 0) {
                const currentTime = videoElement.currentTime;
                timeIndicator
                    .style("display", "block")
                    .attr("x1", xScale(currentTime))
                    .attr("x2", xScale(currentTime));
            }
        }

        videoElement.addEventListener("timeupdate", updateTimeIndicator);
        videoElement.addEventListener("loadedmetadata", updateTimeIndicator);

        // Add click interaction to seek video
        const clickArea = svg.append("rect")
            .attr("class", "click-area")
            .attr("x", margin.left)
            .attr("y", margin.top)
            .attr("width", width)
            .attr("height", height)
            .attr("fill", "transparent")
            .style("cursor", "pointer")
            .on("click", function(event) {
                const [mouseX] = d3.pointer(event, this);
                const adjustedX = mouseX - margin.left;
                const clickedTime = xScale.invert(adjustedX);
                
                // Clamp to valid range
                const seekTime = Math.max(0, Math.min(videoElement.duration, clickedTime));
                videoElement.currentTime = seekTime;
                videoElement.pause();
                
                // Show feedback
                timeIndicator
                    .style("display", "block")
                    .attr("x1", xScale(seekTime))
                    .attr("x2", xScale(seekTime));
            });

        // Initialize time indicator
        if (videoElement.readyState >= 2) {
            updateTimeIndicator();
        }
    }
}

function makeVideoTile(video_data, options) {
    let clip_id = video_data["annotations"]["clip_id"];
    let clip_options = video_data["annotations"]["annotations"];
    let clip_country = video_data["country"];
    let clip_country_name = video_data["country_name"];
    let data_source = video_data["data_source"];

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

        return container;
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

    // Show data source (link to per-dataset stats) — match annotation page pills
    const dataSourceContainer = video_tile.querySelector(".data-source-container");
    if (dataSourceContainer && data_source) {
        const dataSources = Array.isArray(data_source)
            ? data_source
            : String(data_source).split(',').map(s => s.trim()).filter(Boolean);
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
    
    // Build tabs container
    let tabs = document.createElement("div");
    tabs.className = "caption-tabs";
    let tabButtons = document.createElement("div");
    tabButtons.className = "caption-tab-buttons";
    let tabContents = document.createElement("div");
    tabContents.className = "caption-tab-contents";

    // Video_data.captions is { model_name: [caption_text | {caption,start_time,end_time}, ...] }
    const vlmAvailable = window.vlmJudgeAvailable || false;
    if (video_data.captions && Object.keys(video_data.captions).length) {
        let first = true;

        Object.entries(video_data.captions).forEach(([model, texts]) => {
            const items = Array.isArray(texts) ? texts : (texts != null ? [texts] : []);

            // --- Tab button ---
            const btn = document.createElement("button");
            btn.textContent = model;
            btn.className = "caption-tab-btn" + (first ? " active" : "");
            tabButtons.appendChild(btn);

            // --- Tab content ---
            const content = document.createElement("div");
            content.className = "caption-tab-content";
            content.style.display = first ? "block" : "none";

            const modelScores = (video_data.vlm_caption_scores || {})[model] || [];
            const html = items.map((raw, idx) => {
                const isObj = raw && typeof raw === 'object';
                const text = isObj ? raw.caption : raw;
                const hasTimes = isObj && raw.start_time != null && raw.end_time != null;
                const timeHtml = hasTimes
                    ? `<div class=\"caption-time\"><span class=\"caption-time-pill\">${Math.round(raw.start_time)}–${Math.round(raw.end_time)} s</span></div>`
                    : '';
                const uid = isObj && raw.uid != null ? raw.uid : '';
                const cached = modelScores[idx];
                const cachedHtml = cached ? renderVlmCaptionScoresHtml(cached) : '';
                const evalRow = vlmAvailable
                    ? `<div class="vlm-judge-row">`
                        + `<button class="vlm-judge-btn" title="Score this caption with VLM Judge">Evaluate Caption</button>`
                        + `<div class="vlm-judge-result"${cachedHtml ? '' : ' style="display:none"'}>${cachedHtml}</div>`
                        + `</div>`
                    : '';
                return `<div class=\"caption-item\" data-idx=\"${idx}\" data-caption-uid=\"${uid}\">${timeHtml}${text ?? ""}${evalRow}</div>`;
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
    if (vlmAvailable) {
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
    }

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
            console.log(msg);
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
            console.log(msg);

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

        // Remove annotation button (hidden for numeric labels)
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

    return video_tile;
}

function getSearchLink() {
    let clip_id = window.currentClipId;
    let link = "/#page=0&search_clipid=" + clip_id;
    return `<a href="${link}" target="_blank" rel="noopener noreferrer">${clip_id}</a>`;
}

window.onhashchange = function () {
    const hash = window.location.hash.slice(1);
    const params = getQueryParams(hash);
    showPage(
        (params.clip_id !== undefined) ? params.clip_id : null,
        (params.project_source !== undefined) ? params.project_source : null,
    );
}
