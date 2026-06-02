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

// Classifier panel state.
window.classifierState = {
    runs: [],
    numAnnotations: {},
    numAutolabelledAnnotations: {},
    labelOptions: [],
    form: {
        embedType: "cosmos",
        positiveLabels: [],
        negativeLabels: [],
        useAutolabels: false,
        nNegative: 100,
        nPositive: null,
        probabilityExpression: "",
    },
    selectedRunId: null,
    pendingRunId: null,
    activeSearch: {run_id: null, expression: null},
    canTrain: true,
};

Object.defineProperty(window, "currentClassifierSearch", {
    get: () => window.classifierState.activeSearch,
    set: (v) => {
        // Only adopt a run as the panel's selected one when an active
        // classifier search arrives with a real run_id (Use Classifier,
        // URL restore). Clearing the classifier filter leaves the
        // currently selected run alone so the user can keep clicking
        // around it (View Positives → clear chip → View Negatives).
        const next = v || {run_id: null, expression: null};
        window.classifierState.activeSearch = next;
        window.classifierState.form.probabilityExpression =
            next.expression || "";
        if (next.run_id) {
            window.classifierState.selectedRunId = next.run_id;
        }
        renderClassifierPanel();
    },
    configurable: true,
});

Object.defineProperty(window, "classifierStatuses", {
    get: () => ({
        runs: window.classifierState.runs,
        pending: window.classifierState.runs
            .filter(r => r.status === "pending")
            .map(r => r.run_id),
        number_of_annotations: window.classifierState.numAnnotations,
        number_of_autolabelled_annotations:
            window.classifierState.numAutolabelledAnnotations,
    }),
    set: (v) => {
        if (!v) return;
        window.classifierState.runs = v.runs || [];
        window.classifierState.numAnnotations =
            v.number_of_annotations || {};
        window.classifierState.numAutolabelledAnnotations =
            v.number_of_autolabelled_annotations || {};
        renderClassifierPanel();
    },
    configurable: true,
});

// Current label filter mode ("any" = OR, "all" = AND)
window.currentFilterMode = "any";

// Current data-source filter mode ("any" = OR, "all" = AND)
window.currentDataSourceMode = "any";

// Current labels-to-exclude filter mode ("any" = OR, "all" = AND)
window.currentLabelsToExcludeMode = "any";

// Current cluster search (set when a cluster centroid is clicked on the t-SNE plot)
window.currentClusterSearch = {run_id: null, cluster_ids: []};
window.currentClusterDistanceRange = { lo: 0, hi: 100 };
window._clusterSelectMode = "range";

// Current clip-list search: ``hash`` is the content-addressed key
// returned by /upload_clip_list; ``count`` is its size (used only for
// the chip label). The hash flows through the URL via the
// ``clip_id_list_hash`` query param; ``count`` is hydrated from
// /clip_list when restoring from URL state.
window.currentClipList = {hash: null, count: 0};

function setClusterSelectMode(mode) {
    window._clusterSelectMode = mode;
    const rangeTab = document.getElementById("cluster-select-tab-range");
    const boxTab = document.getElementById("cluster-select-tab-box");
    const rangePane = document.getElementById("cluster-select-range-pane");
    const boxPane = document.getElementById("cluster-select-box-pane");
    if (rangeTab) rangeTab.classList.toggle("active", mode === "range");
    if (boxTab) boxTab.classList.toggle("active", mode === "box");
    if (rangePane) rangePane.style.display = mode === "range" ? "" : "none";
    if (boxPane) boxPane.style.display = mode === "box" ? "" : "none";
    const overlay = document.getElementById("cluster-box-select-overlay");
    if (overlay) overlay.style.display = "none";
    const canvas = document.getElementById("umap-canvas");
    if (canvas && window._clusteringRun?.zoomedCid != null) {
        canvas.style.cursor = mode === "box" ? "crosshair" : "";
    }
}

window.classifierStatusCheckStop = false;


var REWRITE_SEARCH_TYPES = {
    'caption': {
        inputId: 'search-term',
        btnId: 'caption-rewrite-btn',
        infoId: 'caption-rewrite-info',
        tagsId: 'caption-rewrite-tags',
    },
    'caption-embed': {
        inputId: 'caption-embed-search-text',
        btnId: 'caption-embed-rewrite-btn',
        infoId: 'caption-embed-rewrite-info',
        tagsId: 'caption-embed-rewrite-tags',
    },
    'semantic': {
        inputId: 'semantic-search-text',
        btnId: 'semantic-rewrite-btn',
        infoId: 'semantic-rewrite-info',
        tagsId: 'semantic-rewrite-tags',
    },
    'visual': {
        inputId: 'visual-search-text',
        btnId: 'visual-rewrite-btn',
        infoId: 'visual-rewrite-info',
        tagsId: 'visual-rewrite-tags',
    },
};

function showLoading(message) {
    const el = document.getElementById("loading-block");
    const msg = document.getElementById("loading-message");
    const elapsed = document.getElementById("loading-elapsed");
    if (msg) {
        msg.textContent = message || "Loading...";
    }
    if (elapsed) {
        elapsed.textContent = "";
    }
    if (el) {
        el.style.display = "flex";
    }
    const start = Date.now();
    clearInterval(window._loadingTimer);
    window._loadingTimer = setInterval(function() {
        const secs = Math.floor((Date.now() - start) / 1000);
        if (elapsed && secs >= 1) {
            elapsed.textContent = secs + "s elapsed";
        }
    }, 500);
}

function hideLoading() {
    clearInterval(window._loadingTimer);
    const el = document.getElementById("loading-block");
    if (el) {
        el.style.display = "none";
    }
}

// Helper to zip 2 arrays
function zip2(a, b) {
    return a.map((v, i) => [v, b[i]]);
}

// Helper to format time (e.g., 00:00)
function formatTime(seconds) {
    if (isNaN(seconds) || seconds < 0) return "00:00";
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes.toString().padStart(2, "0")}:${remainingSeconds.toString().padStart(2, "0")}`;
}

function getQueryParams(queryString) {
    const dictionary = {};
    const params = queryString.split('&');
    for (const param of params) {
        const [key, value] = param.split('=');
        if (key != "") {
            dictionary[key] = decodeURIComponent(value || '');
        }
    }
    return dictionary;
}

function encodeQuery(components, query = "") {
    return Object.entries(components)
        .reduce((query, [key, value]) => {
            if (value === null || value === undefined) {
                return query;
            }
            if (Array.isArray(value)) {
                value = value.join("||");
            }
            if (query != "") {
                query += "&";
            }
            return query + encodeURIComponent(key) + "=" + encodeURIComponent(value);
        }, query);
}

// Input history manager using localStorage
const InputHistory = {
    setup: function(inputId, storageKey, maxEntries = 10) {
        const input = document.getElementById(inputId);
        if (!input) return;

        // Create and attach datalist for suggestions
        const datalist = document.createElement('datalist');
        datalist.id = inputId + '-history';
        input.parentNode.insertBefore(datalist, input.nextSibling);
        input.setAttribute('list', datalist.id);

        // Load history and auto-fill with most recent value
        const loadHistory = () => {
            try {
                const history = JSON.parse(localStorage.getItem(storageKey) || '[]');

                // Auto-fill input with most recent value if field is empty
                if (history.length > 0 && !input.value) {
                    input.value = history[0];
                }

                // Populate datalist with all history for suggestions
                datalist.innerHTML = history.map(v => `<option value="${v}">`).join('');
            } catch (e) {}
        };

        // Save to history
        const saveValue = () => {
            const value = input.value.trim();
            if (!value) return;

            try {
                let history = JSON.parse(localStorage.getItem(storageKey) || '[]');
                history = [value, ...history.filter(v => v !== value)].slice(0, maxEntries);
                localStorage.setItem(storageKey, JSON.stringify(history));
            } catch (e) {}
        };

        loadHistory();
        input.addEventListener('blur', saveValue);
    }
};

// Remove specific query keys from a URL/path string
function stripQueryKeysFromPath(path, keysToRemove) {
    const str = String(path || "");
    const qIndex = str.indexOf("?");
    if (qIndex < 0) return str;
    const base = str.slice(0, qIndex);
    const qs = str.slice(qIndex + 1);
    const params = getQueryParams(qs);
    for (const k of keysToRemove) {
        delete params[k];
    }
    const newQuery = encodeQuery(params, "");
    return base + (newQuery ? ("?" + newQuery) : "");
}

function buildEndpoint(parsedPath, params) {
    const components = {...params};
    // Boolean-backed fields: convert true -> "true", everything else -> null
    // so encodeQuery drops them. Without this a JS false would serialize
    // as the string "false", which the backend would read as truthy.
    components.without_ann = components.without_ann ? "true" : null;
    components.left_hand_driving = (components.left_hand_driving === true) ? "true" : null;
    components.with_ego_data = components.with_ego_data ? "true" : null;
    components.with_metrics = (components.with_metrics === true) ? "true" : null;
    components.with_bev = (components.with_bev === true) ? "true" : null;

    return encodeQuery(components, parsedPath + "?");
}

// Cluster-filter URL/payload fragment shared by every search builder
// (main.js, annotation.js, leaderboard.js). Range / zoom only apply to
// single-cluster selections; in multi-cluster mode the slider is hidden
// anyway and the gates here keep stale params out of the URL.
function clusterFilterPayload() {
    const ids = window.currentClusterSearch.cluster_ids || [];
    const single = ids.length === 1;
    const range = window.currentClusterDistanceRange || { lo: 0, hi: 100 };
    return {
        cluster_run_id: window.currentClusterSearch.run_id,
        cluster_ids: ids.length ? ids.join(",") : null,
        cluster_distance_min: single && range.lo > 0 ? range.lo : null,
        cluster_distance_max: single && range.hi < 100 ? range.hi : null,
        cluster_zoom: single && window._clusteringRun?.zoomedCid != null ? 1 : null,
    };
}

// Normalize whatever form the server / URL returns for cluster_ids
// (array, comma-separated string, or null/empty) into a string[].
function clusterIdsFromData(value) {
    if (!value) return [];
    if (Array.isArray(value)) return value.map(String);
    return String(value).split(",").filter(Boolean);
}

function buildCurrentFilters() {
    return {
        filter: window.currentFilter,
        numeric_filter: window.currentNumericFilter,
        times: window.currentTimes,
        without_ann: window.currentWithoutAnn,
        left_hand_driving: window.currentLeftHandDriving,
        search: window.currentSearch,
        caption_extra_queries: (window.currentExtraQueries && window.currentExtraQueries.length)
            ? window.currentExtraQueries.join("||")
            : null,
        caption_embed_extra_queries: (window.currentCaptionEmbedExtraQueries && window.currentCaptionEmbedExtraQueries.length)
            ? window.currentCaptionEmbedExtraQueries.join("||")
            : null,
        semantic_extra_queries: (window.currentSemanticExtraQueries && window.currentSemanticExtraQueries.length)
            ? window.currentSemanticExtraQueries.join("||")
            : null,
        visual_extra_queries: (window.currentVisualExtraQueries && window.currentVisualExtraQueries.length)
            ? window.currentVisualExtraQueries.join("||")
            : null,
        search_speed: window.currentSpeedQuery,
        search_country: window.currentCountryQuery,
        search_clipid: window.currentClipIDQuery,
        with_ego_data: window.currentWithEgoData,
        with_metrics: window.currentWithMetrics,
        with_bev: window.currentWithBEV,
        trajectory_pattern: window.currentTrajectoryPattern,
        trajectory_shape_clipid: window.currentTrajectoryShapeClipID,
        trajectory_shape_start_t: window.currentTrajectoryShapeStartT,
        trajectory_shape_end_t: window.currentTrajectoryShapeEndT,
        semantic_search_clipid: window.currentSemanticSearchClipID,
        semantic_search_text: window.currentSemanticSearchText,
        visual_search_text: window.currentVisualSearchText,
        visual_search_image_id: window.currentVisualSearchImageId,
        caption_embed_search: window.currentCaptionEmbedSearchText,
        classifier_run_id: window.currentClassifierSearch.run_id,
        probability_expression: window.currentClassifierSearch.expression,
        clip_id_list_hash: window.currentClipList.hash,
        label_types: window.currentLabelTypes,
        search_comments: window.currentSearchTermInComments,
        wm_class_name: window.currentWMClassName,
        wm_min_count: window.currentWMMinCount,
        wm_max_count: window.currentWMMaxCount,
        wm_max_dist: window.currentWMMaxDist,
        wm_min_time: window.currentWMMinTime,
        wm_angle_range: window.currentWMAngleRange,
        data_source: window.currentDataSource,
        project_source: window.currentProjectSource,
        labels_to_exclude: window.currentLabelsToExclude,
        sil_apis: window.currentSILAPIs,
        ...clusterFilterPayload(),
        filter_mode: modePayloadValue(window.currentFilter, window.currentFilterMode),
        data_source_mode: modePayloadValue(window.currentDataSource, window.currentDataSourceMode),
        labels_to_exclude_mode: modePayloadValue(window.currentLabelsToExclude, window.currentLabelsToExcludeMode),
        rank_mode: window.currentRankMode === "rrf" ? "rrf" : null,
        n: window.currentVideosPerPage !== 6 ? window.currentVideosPerPage : null,
    };
}

function updateRRFToggleVisibility() {
    const container = document.getElementById("rrf-toggle-container");
    if (!container) return;
    const chips = [
        "semantic-search-text-display",
        "semantic-search-video-display",
        "visual-search-text-display",
        "visual-search-image-display",
        "classifier-search",
        "caption-embed-search-display",
        "trajectory-shape-search",
        "cluster-search",
    ];
    let activeScored = chips.filter(id => {
        const el = document.getElementById(id);
        return el && el.style.display !== "none";
    }).length;
    if ((window.currentNumericFilter || []).length > 0) activeScored += 1;
    container.style.display = activeScored >= 2 ? "flex" : "none";

    const box = document.getElementById("rrf-toggle-checkbox");
    if (box) box.checked = window.currentRankMode === "rrf";
}

function resetRRFIfTooFewFilters() {
    let scored = 0;
    if (window.currentSemanticSearchText)        scored++;
    if (window.currentSemanticSearchClipID)      scored++;
    if (window.currentVisualSearchText)          scored++;
    if (window.currentVisualSearchImageId)       scored++;
    if (window.currentClassifierSearch?.run_id)  scored++;
    if (window.currentCaptionEmbedSearchText)    scored++;
    if (window.currentTrajectoryShapeClipID)     scored++;
    if (window.currentClusterSearch?.cluster_ids?.length) scored++;
    if ((window.currentNumericFilter || []).length > 0) scored++;
    if (scored < 2) {
        const box = document.getElementById("rrf-toggle-checkbox");
        if (box) box.checked = false;
        window.currentRankMode = "priority";
    }
}

// Trajectory-style dual slider: lightweight per-thumb update for live
// dragging (just labels + gradient) so it stays smooth, and a single
// commit step on release that triggers the expensive zoom rerender +
// search. Kept separate from the trajectory helpers so the two sliders
// don't share state by accident.
function updateClusterDistanceMinValue(value) {
    const loInput = document.getElementById("cluster-distance-min");
    const hiInput = document.getElementById("cluster-distance-max");
    if (!loInput || !hiInput) return;
    let lo = Number(value);
    let hi = Number(hiInput.value);
    if (lo > hi) {
        hi = lo;
        hiInput.value = String(hi);
        document.getElementById("cluster-distance-max-value").textContent = String(hi);
    }
    document.getElementById("cluster-distance-min-value").textContent = String(lo);
    fillClusterDistanceSlider();
    setClusterDistanceToggleAccessible();
}

function updateClusterDistanceMaxValue(value) {
    const loInput = document.getElementById("cluster-distance-min");
    const hiInput = document.getElementById("cluster-distance-max");
    if (!loInput || !hiInput) return;
    let hi = Number(value);
    let lo = Number(loInput.value);
    if (hi < lo) {
        lo = hi;
        loInput.value = String(lo);
        document.getElementById("cluster-distance-min-value").textContent = String(lo);
    }
    document.getElementById("cluster-distance-max-value").textContent = String(hi);
    fillClusterDistanceSlider();
    setClusterDistanceToggleAccessible();
}

// Snap the cluster-distance slider UI (state + DOM + gradient) to a
// given range. Used by reset-to-defaults paths and URL-restore paths.
function setClusterDistanceSliderUI(lo, hi) {
    window.currentClusterDistanceRange = { lo, hi };
    const loInput = document.getElementById("cluster-distance-min");
    const hiInput = document.getElementById("cluster-distance-max");
    const loLabel = document.getElementById("cluster-distance-min-value");
    const hiLabel = document.getElementById("cluster-distance-max-value");
    if (loInput) loInput.value = String(lo);
    if (hiInput) hiInput.value = String(hi);
    if (loLabel) loLabel.textContent = String(lo);
    if (hiLabel) hiLabel.textContent = String(hi);
    fillClusterDistanceSlider();
    setClusterDistanceToggleAccessible();
    if (typeof renderBreadcrumb === "function") renderBreadcrumb();
}

function fillClusterDistanceSlider() {
    const loInput = document.getElementById("cluster-distance-min");
    const hiInput = document.getElementById("cluster-distance-max");
    if (!loInput || !hiInput) return;
    const min = Number(hiInput.min || 0);
    const max = Number(hiInput.max || 100);
    const span = max - min || 1;
    const fromPct = (Number(loInput.value) - min) / span * 100;
    const toPct = (Number(hiInput.value) - min) / span * 100;
    hiInput.style.background = `linear-gradient(
        to right,
        #C6C6C6 0%,
        #C6C6C6 ${fromPct}%,
        #25daa5 ${fromPct}%,
        #25daa5 ${toPct}%,
        #C6C6C6 ${toPct}%,
        #C6C6C6 100%)`;
}

function setClusterDistanceToggleAccessible() {
    const hiInput = document.getElementById("cluster-distance-max");
    if (!hiInput) return;
    hiInput.style.zIndex = Number(hiInput.value) <= 0 ? 2 : 0;
}

// Called on `onchange` (drag release) — does the expensive work:
// snapshot the range to global state, rerender the zoom plot, search.
function commitClusterDistanceRange() {
    const loInput = document.getElementById("cluster-distance-min");
    const hiInput = document.getElementById("cluster-distance-max");
    if (!loInput || !hiInput) return;
    const lo = Number(loInput.value);
    const hi = Number(hiInput.value);
    window.currentClusterDistanceRange = { lo, hi };
    if (typeof renderUMAPPlot === "function" && window._clusteringRun?.zoomedCid != null) {
        renderUMAPPlot();
    }
    if (typeof renderBreadcrumb === "function") renderBreadcrumb();
    search();
}

function toggleClusterDistanceOutliers() {
    const r = window.currentClusterDistanceRange || { lo: 0, hi: 100 };
    const lo = 100 - r.hi;
    const hi = 100 - r.lo;
    const loInput = document.getElementById("cluster-distance-min");
    const hiInput = document.getElementById("cluster-distance-max");
    if (loInput) loInput.value = String(lo);
    if (hiInput) hiInput.value = String(hi);
    document.getElementById("cluster-distance-min-value").textContent = String(lo);
    document.getElementById("cluster-distance-max-value").textContent = String(hi);
    fillClusterDistanceSlider();
    setClusterDistanceToggleAccessible();
    commitClusterDistanceRange();
}

function buildCurrentEndpoint(parsedPath, modelName = null) {
    return buildEndpoint(parsedPath, {
        page: 0,
        model_name: modelName,
        ...buildCurrentFilters(),
    });
}


function populateSILAPIs() {
    const sil = String(window.currentSILAPIs || "");
    const active = new Set(sil.split("||").filter(Boolean));
    let drive = document.getElementById("with-drive");
    let nurec = document.getElementById("with-nurec");
    let instant = document.getElementById("with-instant-nurec");
    let autolabels = document.getElementById("with-sauron");

    // Check what is active
    if (drive) {
        drive.checked = active.has("Drive");
    }
    if (nurec) {
        nurec.checked = active.has("Nurec");
    }
    if (instant) {
        instant.checked = active.has("InstantNurec");
    }
    if (autolabels) {
        autolabels.checked = active.has("Autolabels");
    }
    const selected = [];
    if (drive.checked) {
        selected.push("Drive");
    }
    if (nurec.checked) {
        selected.push("Nurec");
    }
    if (instant.checked) {
        selected.push("InstantNurec");
    }
    if (autolabels.checked) {
        selected.push("Autolabels");
    }
    let silAPIs = selected.length ? selected.join("||") : null;
    return silAPIs;
}

window.onload = function() {
    window.onhashchange();

    // Initialize input history for persistent autocomplete
    InputHistory.setup('save-project-name', 'saveToHistory');
    //InputHistory.setup('shortcuts', 'quickLabelsHistory');

    // Auto-run search when "with-ego-data" changes
    const egoSlider = document.getElementById("with-ego-data");
    if (egoSlider) {
        egoSlider.addEventListener("input", function () {
            window.currentWithEgoData = egoSlider.checked ? true : null;
            search();
        });
    }

    // Auto-run search when "with-times" or "without-times" changes
    const withTimes = document.getElementById("with-times");
    const withoutTimes = document.getElementById("without-times");

    if (withTimes) {
        withTimes.addEventListener("input", function () {
            normalizeTimesFilter(withTimes); // keep exclusivity
            window.currentTimes = withTimes.checked ? "true" : (withoutTimes.checked ? "false" : null);
            search();
        });
    }

    if (withoutTimes) {
        withoutTimes.addEventListener("input", function () {
            normalizeTimesFilter(withoutTimes);
            window.currentTimes = withTimes.checked ? "true" : (withoutTimes.checked ? "false" : null);
            search();
        });
    }

    const withMetrics = document.getElementById("with-metrics");
    if (withMetrics) {
        withMetrics.addEventListener("input", function () {
            window.currentWithMetrics = withMetrics.checked ? "true" : null;
            search();
        });
    }

    const withBEV = document.getElementById("with-bev");
    if (withBEV) {
        withBEV.addEventListener("input", function () {
            window.currentWithBEV = withBEV.checked ? "true" : null;
            search();
        });
    }

    // Auto-run search when "left-hand-driving" changes
    const lhdCheckbox = document.getElementById("left-hand-driving");
    if (lhdCheckbox) {
        lhdCheckbox.addEventListener("input", function () {
            window.currentLeftHandDriving = lhdCheckbox.checked ? true : null;
            search();
        });
    }

    for (const [id, label] of [
        ['with-drive', 'Drive'],
        ['with-nurec', 'Nurec'],
        ['with-sauron', 'Autolabels'],
        ['with-instant-nurec', 'InstantNurec'],
    ]) {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener("input", function () {
                window.currentSILAPIs = el.checked ? label : null;
                search();
            });
        }
    }

    document.addEventListener("click", function (event) {
        if (event.target.tagName === "BUTTON") {
            const msg = document.getElementById("label-success-message");
            if (msg) {
                msg.style.display = "none";
                msg.textContent = "";
            }
        }
    });

    // Wire logout buttons if present
    const logoutBtn = document.getElementById('logout-btn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', async () => {
            try {
                await fetch('/', { method: 'POST', body: 'logout::' });
            } catch (e) {
                // ignore network errors for logout
            }
            window.location.replace('/login');
        });
    }
}

function newUniqueId(length = 10) {
    var s = "";
    while (s.length < length) {
        s += Math.random().toString(36).substr(2);
    }
    return s.substr(0, length);
}

function updateVideoAnnotation(video_id, caption, action, annotationId = null, startTime = -1, endTime = -1) {
    const projectToWrite = document.getElementById("save-project-name").value.trim();
    if (!validateProjectName()) {
        if (!getProjectToWrite()) {
            logMissingProjectToWrite("updating a video annotation");
            return;
        }
    }

    showLoading("Saving...");

    annotationId = annotationId === null ? newUniqueId() : annotationId;
    const req = new XMLHttpRequest();
    req.addEventListener("error", () => {
        console.log("Error communicating with server for annotation update!");
    });
    req.addEventListener("load", () => {
        if (req.status !== 200) {
            console.log("Failed to update annotation on server!");
        }
        // Update local data model based on action
        for (let i = 0; i < window.currentVideos.length; i++) {
            if (window.currentVideos[i].annotations.clip_id === video_id) {
                let annotations = window.currentVideos[i].annotations.annotations;
                if (action === "add") {
                    const newAnnotation = {
                        "uid": annotationId,
                        "key": caption,
                        "start_time": startTime,
                        "end_time": endTime,
                        "label_type": "manual",
                        "project": projectToWrite
                    };
                    annotations.push(newAnnotation);

                    // Also add to global options if it's a new ad-hoc caption
                    if (!window.currentOptions.includes(caption)) {
                        window.currentOptions.push(caption);
                        window.currentOptions.sort();
                    }

                    // Update the annotation count
                    window.annotationsCount++;
                    window.manualAnnotationsCount++;
                } else if (action === "remove") {
                    to_remove = annotations.filter(ann => ann.uid === annotationId)[0];
                    annotations = annotations.filter(ann => ann.uid !== annotationId);
                    window.currentVideos[i].annotations.annotations = annotations;
                    // Update the annotation count
                    window.annotationsCount--;
                    if (to_remove.label_type === "manual") {
                        window.manualAnnotationsCount--;
                    } else if (to_remove.label_type == "autolabel") {
                        window.autolabelAnnotationsCount--;
                    } 
                } else if (action === "update_times") {
                    for (let j = 0; j < annotations.length; j++) {
                        if (annotations[j].uid === annotationId) {
                            if (startTime !== -1 && startTime !== null) annotations[j].start_time = startTime;
                            if (endTime !== -1 && endTime !== null) annotations[j].end_time = endTime;
                            // When updating times, the backend converts label_type to 'manual'. Reflect this locally
                            if (annotations[j].label_type !== "manual") {
                                const prevType = annotations[j].label_type;
                                annotations[j].label_type = "manual";
                                // Adjust counts to keep UI stats consistent until next fetch
                                if (typeof window.manualAnnotationsCount === 'number') window.manualAnnotationsCount++;
                                if (prevType === "autolabel" && typeof window.autolabelAnnotationsCount === 'number') {
                                    window.autolabelAnnotationsCount = Math.max(0, window.autolabelAnnotationsCount - 1);
                                }
                            }
                            break;
                        }
                    }
                }
                break;
            }
        }

        checkClassifierState();
        render();
        hideLoading();
    });
    req.open("POST", "");
    req.setRequestHeader("Content-Type", "text/plain");
    req.send(`${action}::${video_id}::${annotationId}::${caption}::${startTime}::${endTime}::${projectToWrite}`);
}

function submitCaption(el) {
    let video_id = el.parentElement.parentElement.id.substr(11);
    let caption = el.parentElement.getElementsByTagName("select")[0].selectedOptions[0].value;
    if (caption === "") {
        return;
    }
    updateVideoAnnotation(video_id, caption, "add", null, -1, -1);
}

function resetTimeButtons(container) {
    container.querySelectorAll(".start-log-btn").forEach((btn) => {
        btn.disabled = true;
        btn.textContent = "Start Annotation";
    });
    container.querySelectorAll(".end-log-btn").forEach((btn) => {
        btn.disabled = true;
    });
}

function verifyAnnotation() {
    const el = this;
    const videoTile = el.parentElement.parentElement.parentElement;
    const video_id = videoTile.id.substr(11);
    const annotationBtn = el.parentElement.firstChild; // the main annotation button
    const annotation = annotationBtn.value;
    const annotationId = annotationBtn.id; // original uid
    const labelType = annotationBtn.dataset.labelType;

    // Only proceed for autolabel
    if (labelType !== "autolabel") {
        return;
    }

    // Require a target project for manual annotations
    if (!validateProjectName()) {
        if (!getProjectToWrite()) {
            logMissingProjectToWrite("verifying an annotation");
            return;
        }
    }

    showLoading("Saving...");
    const projectToWrite = document.getElementById("save-project-name").value.trim();
    console.log(projectToWrite);
    const req = new XMLHttpRequest();
    req.addEventListener("error", () => {
        console.log("Error communicating with server for verify action!");
    });
    req.addEventListener("load", () => {
        if (req.status !== 200) {
            console.log("Failed to verify annotation on server!");
        }
        // Update local data model so render() reflects the change without navigation
        for (let i = 0; i < window.currentVideos.length; i++) {
            if (window.currentVideos[i].annotations.clip_id === video_id) {
                let annotations = window.currentVideos[i].annotations.annotations;
                for (let j = 0; j < annotations.length; j++) {
                    if (annotations[j].uid === annotationId) {
                        const prevType = annotations[j].label_type;
                        annotations[j].label_type = "manual";

                        // Mirror previous client-side counts adjustments
                        if (prevType !== "manual") {
                            window.manualAnnotationsCount++;
                            if (prevType === "autolabel") {
                                window.autolabelAnnotationsCount--;
                            }
                        }
                        break;
                    }
                }
                break;
            }
        }

        checkClassifierState();
        render();
        hideLoading();
    });
    req.open("POST", "");
    req.setRequestHeader("Content-Type", "text/plain");
    // Keep payload shape consistent with other actions
    req.send(`verify::${video_id}::${annotationId}::${annotation}::-1::-1::${projectToWrite}`);
}

function removeAnnotation() {
    let el = this;
    let videoTile = el.parentElement.parentElement.parentElement;
    let video_id = videoTile.id.substr(11);
    let annotationBtn = el.parentElement.firstChild;
    let annotation = annotationBtn.value;
    let annotationId = annotationBtn.id;

    if (window.selectedAnnotation &&
        window.selectedAnnotation.videoId === video_id &&
        window.selectedAnnotation.uid === annotationId) { 
        window.selectedAnnotation = null;
        resetTimeButtons(videoTile);
    }

    updateVideoAnnotation(video_id, annotation, "remove", annotationId);
}

function selectAnnotation() {
    let el = this;
    let wrapper = el.parentElement;
    let videoTile = wrapper.parentElement.parentElement;
    let video_id = videoTile.id.substr(11);
    let annotation = el.value;
    let annotationId = el.id;

    // Do not allow time updates for numeric labels
    if (el.dataset.labelType === 'numeric') {
        return;
    }

    // Clear all previous selections across all videos
    document.querySelectorAll(".annotation-button-wrapper.selected").forEach((selected) => {
        selected.classList.remove("selected");
    });
    resetTimeButtons(document);

    // Set current as selected
    wrapper.classList.add("selected");
    videoTile.querySelector(".start-log-btn").disabled = false;

    window.selectedAnnotation = {
        videoId: video_id,
        annotationKey: annotation,
        uid: annotationId,
        startTime: -1,
        endTime: -1
    };

}

function makeTimeSpan(start, end) {
    let timeDisplay = document.createElement("span");
    timeDisplay.className = "annotation-time-display";
    if (start === -1) {
        return timeDisplay;
    }
    var label = formatTime(start) + " - ";
    if (end !== -1) {
        label += formatTime(end);
    }
    timeDisplay.innerText = " (" + label + ")";

    return timeDisplay;
}


function checkClassifierState() {
    return fetch("/classifiers_status")
        .then(response => response.json())
        .then(data => {
            if (window.classifierStatusCheckStop) {
                return;
            }
            const state = window.classifierState;
            state.runs = data.runs || [];
            state.numAnnotations = data.number_of_annotations || {};
            state.numAutolabelledAnnotations =
                data.number_of_autolabelled_annotations || {};

            // Auto-select a newly-completed pending run.
            if (state.pendingRunId) {
                const done = state.runs.find(
                    r => r.run_id === state.pendingRunId
                        && r.status === "done"
                );
                if (done) {
                    state.selectedRunId = state.pendingRunId;
                    state.pendingRunId = null;
                    state.lastRenderedSelectedRunId = undefined;
                }
            }

            renderClassifierPanel();
            if ((data.pending || []).length > 0) {
                setTimeout(checkClassifierState, 1000);
            }
        });
}

function useClassifier() {
    const expressionInput = document.getElementById("probability-expression");
    const MIN_SCORE = 0.3;
    const expr = expressionInput.value.trim();
    if (!expr) {
        expressionInput.focus();
        expressionInput.setCustomValidity("Please enter a probability expression, e.g. p > 0.95");
        expressionInput.reportValidity();
        return;
    }
    const ltMatch = expr.match(/^p\s*<=?\s*([\d.]+)$/);
    const rangeMatch = expr.match(/^[\d.]+\s*<=?\s*p\s*<=?\s*([\d.]+)$/);
    const minProbThr = ltMatch ? parseFloat(ltMatch[1])
                     : rangeMatch ? parseFloat(rangeMatch[1])
                     : null;
    if (minProbThr !== null && minProbThr <= MIN_SCORE) {
        expressionInput.focus();
        expressionInput.setCustomValidity(`No scores <= ${MIN_SCORE} are stored. Use a value above ${MIN_SCORE}.`);
        expressionInput.reportValidity();
        return;
    }
    expressionInput.setCustomValidity("");

    const runId = window.classifierState.selectedRunId;
    if (!runId) return;
    window.classifierState.form.probabilityExpression = expr;
    window.classifierState.activeSearch = {run_id: runId, expression: expr};
    renderClassifierPanel();
    search();
}

function trainClassifier() {
    if (!window.classifierState.canTrain) return;
    window.classifierStatusCheckStop = true;
    const state = window.classifierState;
    const f = state.form;
    const label = (f.positiveLabels || []).slice().sort().join("&&");
    const negativeLabelsArg = (f.negativeLabels || []).join(",");
    const payload =
        `train_classifier::${label}::${f.nNegative}::${negativeLabelsArg}` +
        `::${f.useAutolabels}::${f.nPositive ?? ""}::${f.embedType}`;

    // Optimistic UI: synthesize a pending run so renderClassifierPanel's
    // matchingPending check flips the button to "Training (Please wait)..."
    // immediately. The real server status replaces state.runs wholesale
    // on the next /classifiers_status response.
    const optimisticRunId = `pending:${Date.now()}`;
    state.runs.push({
        run_id: optimisticRunId,
        status: "pending",
        embed_type: f.embedType,
        positive_labels: (f.positiveLabels || []).slice(),
        negative_labels: (f.negativeLabels || []).slice(),
        started_at: Date.now() / 1000,
    });
    renderClassifierPanel();

    fetch("", {
        method: "POST",
        headers: {"Content-Type": "text/plain"},
        body: payload,
    })
        .then(r => r.json())
        .then(data => {
            // Remember the new run_id so checkClassifierState auto-
            // selects it once its status flips to "done".
            if (data && data.run_id) {
                state.pendingRunId = data.run_id;
            }
            window.classifierStatusCheckStop = false;
            checkClassifierState();
        })
        .catch(() => {
            console.log("Failed to perform classifier action on server!");
            state.runs = state.runs.filter(r => r.run_id !== optimisticRunId);
            renderClassifierPanel();
            window.classifierStatusCheckStop = false;
        });
}

function exportClassifier() {
    const runId = window.classifierState.selectedRunId;
    if (!runId) return;
    window.location.href = `/classifier/export/${encodeURIComponent(runId)}`;
}

function viewClassifierTrainingClips(kind) {
    const state = window.classifierState;
    if (!state.selectedRunId) {
        return;
    }
    if (kind !== "positive" && kind !== "negative") return;
    const run = state.runs.find(r => r.run_id === state.selectedRunId);
    const hash = run?.[`${kind}_clip_list_hash`];
    const count = run?.[`n_${kind}_clips`] || 0;
    if (!hash) return;
    window.currentClipList = {hash, count};
    search();
}


function resetClassifierForm() {
    const f = window.classifierState.form;
    f.embedType = "cosmos";
    f.positiveLabels = [];
    f.negativeLabels = [];
    f.useAutolabels = false;
    f.nNegative = 100;
    f.nPositive = null;
    f.probabilityExpression = "";
}

function deleteClassifierRun() {
    const runId = window.classifierState.selectedRunId;
    if (!runId) return;
    const wasActive =
        window.classifierState.activeSearch.run_id === runId;
    fetch("", {
        method: "POST",
        headers: {"Content-Type": "text/plain"},
        body: `delete_classifier_run::${runId}`,
    })
        .then(r => r.json())
        .then(() => {
            const state = window.classifierState;
            resetClassifierForm();
            state.selectedRunId = null;
            state.lastRenderedSelectedRunId = null;
            if (wasActive) {
                // The deleted run was the active classifier search:
                // clearing activeSearch + re-running search() drops
                // classifier_run_id from the URL hash and refreshes
                // results without the deleted classifier.
                resetClassifierSearch(false);
            } else {
                renderClassifierPanel();
            }
            checkClassifierState();
        })
        .catch(e => console.error("Failed to delete classifier run", e));
}


// silent = true: skip triggering a new search (used when called from clearSearch())
function resetTrajectorySearch(silent = false) {
    window.currentTrajectoryShapeClipID = null;
    window.currentTrajectoryShapeStartT = null;
    window.currentTrajectoryShapeEndT = null;

    document.getElementById("trajectory-shape-search").style.display = "none";
    document.getElementById("trajectory-shape-clipid").value = "";
    document.getElementById("trajectory-shape-clipid").disabled = false;
    resetTrajectoryTime();
    if (!silent) {
        search();
    }
}

function resetTrajectoryTime() {
    document.getElementById("trajectory-shape-start-t").value = "0";
    document.getElementById("trajectory-shape-end-t").value = "20";
    updateTrajectoryShapeStartTimeValue("0");
    updateTrajectoryShapeEndTimeValue("20");
}

function fillTrajectoryShapeSlider() {
    const startInput = document.getElementById("trajectory-shape-start-t");
    const endInput = document.getElementById("trajectory-shape-end-t");
    if (!startInput || !endInput) return;
    const min = Number(endInput.min || 0);
    const max = Number(endInput.max || 100);
    const start = Number(startInput.value);
    const end = Number(endInput.value);
    const rangeDistance = max - min;
    const fromPosition = start - min;
    const toPosition = end - min;
    endInput.style.background = `linear-gradient(
      to right,
      #C6C6C6 0%,
      #C6C6C6 ${(fromPosition)/(rangeDistance)*100}%,
      #25daa5 ${((fromPosition)/(rangeDistance))*100}%,
      #25daa5 ${(toPosition)/(rangeDistance)*100}%, 
      #C6C6C6 ${(toPosition)/(rangeDistance)*100}%, 
      #C6C6C6 100%)`;
}

function setTrajectoryShapeToggleAccessible() {
    const endInput = document.getElementById("trajectory-shape-end-t");
    if (!endInput) return;
    if (Number(endInput.value) <= 0) {
      endInput.style.zIndex = 2;
    } else {
      endInput.style.zIndex = 0;
    }
}

function updateTrajectoryShapeStartTimeValue(value) {
    const startInput = document.getElementById("trajectory-shape-start-t");
    const endInput = document.getElementById("trajectory-shape-end-t");
    let start = Number(value);
    let end = Number(endInput.value);

    if (start > end) {
        end = start;
        endInput.value = String(end);
        document.getElementById("trajectory-shape-end-t-value").textContent = end + ' s';
    }

    document.getElementById("trajectory-shape-start-t-value").textContent = start + ' s';
    fillTrajectoryShapeSlider();
    setTrajectoryShapeToggleAccessible();
}

function updateTrajectoryShapeEndTimeValue(value) {
    const startInput = document.getElementById("trajectory-shape-start-t");
    const endInput = document.getElementById("trajectory-shape-end-t");
    let end = Number(value);
    let start = Number(startInput.value);

    if (end < start) {
        start = end;
        startInput.value = String(start);
        document.getElementById("trajectory-shape-start-t-value").textContent = start + ' s';
    }

    document.getElementById("trajectory-shape-end-t-value").textContent = end + ' s';
    fillTrajectoryShapeSlider();
    setTrajectoryShapeToggleAccessible();
}


// silent = true: skip triggering a new search (used when called from clearSearch())
function resetWMSearch(silent = false) {
    window.currentWMClassName = null;
    window.currentWMMinCount = null;
    window.currentWMMaxCount = null;
    window.currentWMMaxDist = null;
    window.currentWMMinTime = null;
    window.currentWMAngleRange = null;

    document.getElementById("wm-search").style.display = "none";
    document.getElementById("wm-class-name").value = "";
    document.getElementById("wm-angle-range").selectedIndex = 0;
    document.getElementById("wm-min-count").value = "1";
    document.getElementById("wm-max-count").value = "500";
    document.getElementById("wm-max-dist").value = "10";
    document.getElementById("wm-min-time").value = "0";
    updateWMDistanceValue("10");
    updateWMTimeValue("0");
    resetWMAngleSelector();
    toggleWMSearchButton();
    if (!silent) {
        search();
    }
}

// silent = true: skip triggering a new search (used when called from clearSearch())
function resetSemanticSearchVideo(silent = false) {
    window.currentSemanticSearchClipID = null;
    document.getElementById("semantic-search-video-display").style.display = "none";
    document.getElementById("semantic-search-clipid").value = "";
    document.getElementById("semantic-search-clipid").disabled = false;
    if (!silent) {
        search();
    }
}

// silent = true: skip triggering a new search (used when called from clearSearch())
function resetSemanticSearchText(silent = false) {
    window.currentSemanticSearchText = null;
    document.getElementById("semantic-search-text-display").style.display = "none";
    document.getElementById("semantic-search-text").value = "";
    if (!silent) {
        search();
    }
}

// silent = true: skip triggering a new search (used when called from clearSearch())
function resetClassifierSearch(silent = false) {
    window.classifierState.activeSearch = {run_id: null, expression: null};
    window.classifierState.selectedRunId = null;
    window.classifierState.form.probabilityExpression = "";
    window.classifierState.lastRenderedSelectedRunId = undefined;
    document.getElementById("classifier-search").style.display = "none";
    renderClassifierPanel();
    if (!silent) {
        search();
    }
}

// silent = true: skip triggering a new search (used when called from clearSearch())
// Deselects the cluster and exits zoom but keeps the run's overview UMAP
// visible. To exit the run entirely, use the "← Back to Runs" button.
function resetClusterSearch(silent = false) {
    window.currentClusterSearch.cluster_ids = [];
    document.getElementById("cluster-search").style.display = "none";
    const run = window._clusteringRun;
    if (run) {
        run.selectedCids = [];
        setZoom(null);
        const infoPanel = document.getElementById("cluster-info-panel");
        if (infoPanel) infoPanel.style.display = "none";
        if (typeof renderUMAPPlot === "function") renderUMAPPlot();
    }
    setClusterDistanceSliderUI(0, 100);
    if (typeof renderBreadcrumb === "function") renderBreadcrumb();
    if (!silent) {
        search();
    }
}

// silent = true: skip triggering a new search (used when called from clearSearch())
function uploadImageAndSearch() {
    const file = document.getElementById("visual-search-image").files[0];
    if (!file) return;
    showLoading("Uploading image...");
    file.arrayBuffer().then(function(buf) {
        return fetch("/upload_image", {
            method: "POST",
            headers: { "Content-Length": buf.byteLength },
            body: buf,
        });
    }).then(function(r) { return r.json(); })
    .then(function(data) {
        window._otherFilterTimestamp = Date.now();
        window.currentVisualSearchImageId = data.upload_id;
        search();
    }).catch(function() { hideLoading(); });
}

function resetVisualSearchText(silent = false) {
    window.currentVisualSearchText = null;
    window.currentVisualExtraQueries = [];
    document.getElementById("visual-search-text").value = "";
    clearRewriteTags('visual');
    if (!silent) {
        search();
    }
}

function resetVisualSearchImage(silent = false) {
    window.currentVisualSearchImageId = null;
    document.getElementById("visual-search-image").value = "";
    if (!silent) {
        search();
    }
}

function resetVisualSearch(silent = false) {
    window.currentVisualSearchText = null;
    window.currentVisualSearchImageId = null;
    window.currentVisualExtraQueries = [];
    document.getElementById("visual-search-text").value = "";
    document.getElementById("visual-search-image").value = "";
    clearRewriteTags('visual');
    if (!silent) {
        search();
    }
}

function resetCaptionEmbedSearch(silent = false) {
    window.currentCaptionEmbedSearchText = null;
    document.getElementById("caption-embed-search-display").style.display = "none";
    document.getElementById("caption-embed-search-text").value = "";
    if (!silent) {
        search();
    }
}

// silent = true: skip triggering a new search (used when called from clearSearch())
function resetCaptionSearch(silent = false) {
    window.currentSearch = null;
    document.getElementById("caption-search").style.display = "none";
    document.getElementById("search-term").value = "";
    if (typeof clearRewriteTags === "function") clearRewriteTags();
    if (!silent) {
        search();
    }
}

function normalizeTimesFilter(el) {
    let withTimes = document.getElementById("with-times");
    let withoutTimes = document.getElementById("without-times");

    if (el == withTimes) {
        if (withoutTimes.checked && el.checked) {
            withoutTimes.checked = false;
        }
    } else if (el == withoutTimes) {
        if (withTimes.checked && el.checked) {
            withTimes.checked = false;
        }
    }
}

function showSpeedSearchHelp() {
    toggleHelp("speed-search-help-content", true);
}

function hideSpeedSearchHelp() {
    toggleHelp("speed-search-help-content", false);
}



function syncRewriteBtn(searchType) {
    var ids = REWRITE_SEARCH_TYPES[searchType];
    var btn = document.getElementById(ids.btnId);
    var input = document.getElementById(ids.inputId);
    btn.disabled = !input.value.trim();
    if (!input.value.trim()) clearRewriteTags(searchType);
}

function hideQueryRewriterHelp() {
    toggleHelp("query-rewriter-help-content", false);
}

function showVlmCheckHelp() {
    toggleHelp("vlm-check-help-content", true);
}
function hideVlmCheckHelp() {
    toggleHelp("vlm-check-help-content", false);
}

function renderVlmCaptionScoresHtml(data) {
    const s = data.scores;
    return `<div class="vlm-judge-scores">`
        + `<span class="vlm-judge-score" title="Scene">🏙️ <span class="vlm-judge-attr">Scene</span> ${s.scene}</span>`
        + `<span class="vlm-judge-score" title="Action">🚗 <span class="vlm-judge-attr">Action</span> ${s.action}</span>`
        + `<span class="vlm-judge-score" title="Road Entities">🚶 <span class="vlm-judge-attr">Road Entities</span> ${s.road_entities}</span>`
        + `<span class="vlm-judge-score" title="Temporal">⏱️ <span class="vlm-judge-attr">Temporal</span> ${s.temporal}</span>`
        + `<span class="vlm-judge-score vlm-judge-overall" title="Overall">★ <span class="vlm-judge-attr">Overall</span> ${s.overall}/10</span>`
        + `</div>`
        + (data.reasoning ? `<div class="vlm-judge-reasoning">${data.reasoning}</div>` : '');
}

function rewriteQuery(searchType) {
    var ids = REWRITE_SEARCH_TYPES[searchType];
    var searchTerm = document.getElementById(ids.inputId).value.trim();
    if (!searchTerm) return;

    showLoading("Rewriting query...");

    fetch("/rewrite?query=" + encodeURIComponent(searchTerm))
        .then(function(r) { return r.json(); })
        .then(function(data) {
            hideLoading();
            window._rewriteOriginalQuery = window._rewriteOriginalQuery || {};
            window._rewriteOriginalQuery[searchType] = searchTerm;
            var rewrites = (data.rewrites || []).filter(function(q) {
                return q.trim().toLowerCase() !== searchTerm.toLowerCase();
            });
            renderRewriteTags(rewrites, searchType);
        })
        .catch(function(err) {
            hideLoading();
            console.error("Rewrite failed:", err);
        });
}

function renderRewriteTags(rewrites, searchType) {
    var ids = REWRITE_SEARCH_TYPES[searchType];
    var container = document.getElementById(ids.infoId);
    var tagsContainer = document.getElementById(ids.tagsId);
    if (!container || !tagsContainer) return;

    if (!rewrites || rewrites.length === 0) {
        container.style.display = "none";
        return;
    }

    tagsContainer.innerHTML = "";
    rewrites.forEach(function(query) {
        var tag = document.createElement("label");
        tag.className = "query-rewrite-tag";

        var cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = true;
        cb.value = query;
        cb.addEventListener("change", function() {
            tag.classList.toggle("unchecked", !cb.checked);
        });

        var text = document.createTextNode(query);
        tag.appendChild(cb);
        tag.appendChild(text);
        tagsContainer.appendChild(tag);
    });

    container.style.display = "block";
}

function clearRewriteTags(searchType) {
    var types = searchType ? [searchType] : Object.keys(REWRITE_SEARCH_TYPES);
    types.forEach(function(t) {
        var ids = REWRITE_SEARCH_TYPES[t];
        var container = document.getElementById(ids.infoId);
        var tagsContainer = document.getElementById(ids.tagsId);
        if (tagsContainer) tagsContainer.innerHTML = "";
        if (container) container.style.display = "none";
    });
    window._rewriteOriginalQuery = window._rewriteOriginalQuery || {};
    if (searchType) {
        window._rewriteOriginalQuery[searchType] = null;
    } else {
        window._rewriteOriginalQuery = {};
        window.currentExtraQueries = [];
        window.currentCaptionEmbedExtraQueries = [];
        window.currentSemanticExtraQueries = [];
        window.currentVisualExtraQueries = [];
    }
}

function clearRewritesAndSearch(searchType) {
    clearRewriteTags(searchType);
    if (searchType === 'caption') window.currentExtraQueries = [];
    else if (searchType === 'caption-embed') window.currentCaptionEmbedExtraQueries = [];
    else if (searchType === 'semantic') window.currentSemanticExtraQueries = [];
    else if (searchType === 'visual') window.currentVisualExtraQueries = [];
    search();
}

function getSelectedRewrites(searchType) {
    var ids = REWRITE_SEARCH_TYPES[searchType];
    var tags = document.querySelectorAll("#" + ids.tagsId + " input[type='checkbox']:checked");
    var selected = [];
    tags.forEach(function(cb) { selected.push(cb.value); });
    return selected;
}

function showCaptionSearchHelp() {
    toggleHelp("caption-search-help-content", true);
}

function hideCaptionSearchHelp() {
    toggleHelp("caption-search-help-content", false);
}

function showSemanticSearchHelp() {
    toggleHelp("semantic-search-help-content", true);
}

function hideSemanticSearchHelp() {
    toggleHelp("semantic-search-help-content", false);
}

function showVisualSearchImageHelp() {
    toggleHelp("visual-search-image-help-content", true);
}

function hideVisualSearchImageHelp() {
    toggleHelp("visual-search-image-help-content", false);
}

function showVisualSearchHelp() {
    toggleHelp("visual-search-help-content", true);
}

function hideVisualSearchHelp() {
    toggleHelp("visual-search-help-content", false);
}

function showCaptionEmbedSearchHelp() {
    toggleHelp("caption-embed-search-help-content", true);
}

function hideCaptionEmbedSearchHelp() {
    toggleHelp("caption-embed-search-help-content", false);
}

function showVideoToVideoSearchHelp() {
    toggleHelp("video-to-video-search-help-content", true);
}

function hideVideoToVideoSearchHelp() {
    toggleHelp("video-to-video-search-help-content", false);
}

function showTrajectoryShapeSearchHelp() {
    toggleHelp("trajectory-shape-search-help-content", true);
}

function hideTrajectoryShapeSearchHelp() {
    toggleHelp("trajectory-shape-search-help-content", false);
}

function showCommentSearchHelp() {
    toggleHelp("comment-search-help-content", true);
}

function hideCommentSearchHelp() {
    toggleHelp("comment-search-help-content", false);
}

function showClassifierMenuHelp() {
    toggleHelp("classifier-menu-help-content", true);
}

function hideClassifierMenuHelp() {
    toggleHelp("classifier-menu-help-content", false);
}

function toggleClassifierMenu(el) {
    const adv = document.getElementById("classifier-container");
    let activate = adv.style.display === "none";
    adv.style.display = (activate) ? "block" : "none";
    el.classList.toggle("selected", activate);

    window.scrollTo(0, 0);
}

function toggleClusteringMenu(btn) {
    const container = document.getElementById("clustering-container");
    const isHidden = container.style.display === "none";
    container.style.display = isHidden ? "block" : "none";
    btn.classList.toggle("selected", isHidden);
    if (isHidden) {
        checkClusteringState();
        window.scrollTo(0, 0);
    }
}

function showClipListModal() {
    const modal = document.getElementById("clip-list-modal");
    if (!modal) return;
    // Reset transient state so reopens look fresh.
    const feedback = document.getElementById("clip-list-feedback");
    if (feedback) {
        feedback.textContent = "";
        feedback.classList.remove("success");
    }
    const input = document.getElementById("clip-list-file");
    if (input) input.value = "";
    const btn = document.getElementById("clip-list-upload-button");
    if (btn) btn.disabled = true;
    modal.style.display = "flex";
}

function hideClipListModal() {
    const modal = document.getElementById("clip-list-modal");
    if (modal) modal.style.display = "none";
}

function showClipListHelp() {
    toggleHelp("clip-list-help-content", true);
}

function hideClipListHelp() {
    toggleHelp("clip-list-help-content", false);
}

function onClipListFileChange() {
    const input = document.getElementById("clip-list-file");
    const btn = document.getElementById("clip-list-upload-button");
    if (btn) btn.disabled = !(input && input.files && input.files.length > 0);
}

function uploadClipListAndSearch() {
    const input = document.getElementById("clip-list-file");
    const file = input.files[0];
    const feedback = document.getElementById("clip-list-feedback");
    const btn = document.getElementById("clip-list-upload-button");
    if (!file) return;
    btn.disabled = true;
    feedback.classList.add("success");
    feedback.textContent = `Uploading ${file.name}...`;
    file.arrayBuffer()
        .then(buf => fetch("/upload_clip_list", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: buf,
        }))
        .then(r => r.json().then(data => ({status: r.status, data})))
        .then(({status, data}) => {
            if (status >= 400 || data.error) {
                feedback.classList.remove("success");
                feedback.textContent = data.error || `Upload failed (${status})`;
                btn.disabled = false;
                return;
            }
            window.currentClipList = {hash: data.hash, count: data.count};
            input.value = "";
            hideClipListModal();
            search();
        })
        .catch(e => {
            feedback.classList.remove("success");
            feedback.textContent = `Upload failed: ${e}`;
            btn.disabled = false;
        });
}

// silent = true: skip triggering a new search (used when called from clearSearch())
function resetClipListSearch(silent = false) {
    window.currentClipList = {hash: null, count: 0};
    document.getElementById("clip-list-search").style.display = "none";
    const feedback = document.getElementById("clip-list-feedback");
    if (feedback) feedback.textContent = "";
    if (!silent) {
        search();
    }
}

// If the URL restored a clip_id_list_hash but the count is missing
// (page load, not a fresh upload), fetch it once so the chip can render.
function hydrateClipListFromUrl() {
    const cl = window.currentClipList;
    if (!cl.hash || cl.count > 0) return;
    fetch(`/clip_list?hash=${encodeURIComponent(cl.hash)}`)
        .then(r => r.ok ? r.json() : null)
        .then(data => {
            if (data && data.hash === window.currentClipList.hash) {
                window.currentClipList.count = data.count;
                showContextualFilters();
            }
        })
        .catch(() => {});
}


function runClustering() {
    const nClustersInput  = document.getElementById("cluster-n-clusters");
    const maxPtsInput     = document.getElementById("cluster-max-points-per-centroid");
    if (!nClustersInput.reportValidity() || !maxPtsInput.reportValidity()) return;

    const n_clusters              = nClustersInput.value;
    const spherical               = document.getElementById("cluster-spherical").checked;
    const max_points_per_centroid = maxPtsInput.value;
    const embed_type              = document.getElementById("cluster-embed-type").value;

    const path = buildEndpoint(
        "/videos",
        {page: window.currentPage, ...buildCurrentFilters(),}
    );

    const btn = document.getElementById("run-clustering-button");
    btn.disabled = true;
    const payload = `run_clustering::${path}::${n_clusters}::${spherical}::${max_points_per_centroid}::${embed_type}`;
    fetch("", {
        method: "POST",
        headers: {"Content-Type": "text/plain"},
        body: payload,
    })
        .then(r => r.json())
        .then(data => {
            btn.disabled = false;
            checkClusteringState();
        })
        .catch(() => { btn.disabled = false; });
}

function checkClusteringState() {
    return fetch("/clustering_status")
        .then(r => r.json())
        .then(data => {
            if (window.clusteringStatusCheckStop) {
                return;
            }

            window.clusteringStatuses = data;

            if (typeof renderClusteringRuns === "function") {
                renderClusteringRuns(data.runs || []);
            }
            const hasPending = (data.runs || []).some(r => r.status === "pending");
            setTimeout(checkClusteringState, hasPending ? 2000 : 10000);
        });
}

function addOptionsToSelect(el, options, selected) {
    for (var i=0; i < options.length; i++) {
        let opt = document.createElement("option");
        opt.text = options[i];
        if (options[i] == selected) {
            opt.selected = true;
        }
        el.add(opt);
    }
}

function syncModeControl(rowId, selectSelector, modeVar) {
    const row = document.getElementById(rowId);
    if (!row) return;
    const count = $(selectSelector).val()?.length || 0;
    row.style.display = count >= 2 ? "flex" : "none";
    const mode = window[modeVar] || "any";
    row.querySelectorAll(".fmb").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.value === mode);
    });
}

function bindModeControl(rowId, modeVar) {
    document.getElementById(rowId).addEventListener("click", function (e) {
        const btn = e.target.closest(".fmb");
        if (!btn) return;
        window[modeVar] = btn.dataset.value;
        search();
    });
}

function modePayloadValue(items, mode) {
    const len = items?.length ?? 0;
    return (len > 0 && mode && mode !== "any") ? mode : null;
}

function toggleHelp(id, show) {
    const box = document.getElementById(id);
    if (box) box.style.display = show ? "block" : "none";
}

function addAnnotationOptions(options, currentFilter) {
    let select = document.getElementById("annotation-tag");
    select.innerHTML = "";
    //let defaultOpt = document.createElement("option");
    //defaultOpt.text = "All";
    //defaultOpt.value = "";
    //select.add(defaultOpt);
    addOptionsToSelect(select, options, null);

    if ($(select).data("select2")) {
        $(select).select2("destroy");
    }
    $(select).select2({
        placeholder: "Select labels...",
        allowClear: true,
        multiple: true,
        width: "100%"
    });
    $(select).off("change.filterMode").on("change.filterMode", () => syncModeControl("filter-mode-row", "#annotation-tag", "currentFilterMode"));

    if (currentFilter !== null) {
        $(select).val(currentFilter).trigger("change");
    }
}

function addMetricOptions(options, currentFilter) {
    let select = document.getElementById("metric-names");
    let ranges = document.getElementById("metric-ranges");
    select.innerHTML = "";
    ranges.innerHTML = "";
    addOptionsToSelect(select, options, null);

    const $select = $(select);

    if ($select.data("select2")) {
        $select.select2("destroy");
    }
    $select.select2({
        placeholder: "Select labels...",
        allowClear: true,
        multiple: true,
        width: "100%"
    });

    let find_range = function (value) {
        for (let i = 0; i < ranges.childNodes.length; i++) {
            const el = ranges.childNodes[i];
            if (el.firstChild.innerText == value) {
                return el;
            }
        }
        return undefined;
    };

    let add_range = function (value, vmin, vmax, order) {
        if (find_range(value)) {
            return;
        }

        let container = document.createElement("div");
        let text = document.createElement("span");
        text.innerText = value;
        container.appendChild(text);

        // Text input for min value
        let input_min = document.createElement("input");
        input_min.type = "number";
        input_min.placeholder = "min";
        input_min.value = vmin || "-inf";
        container.appendChild(input_min);

        // Text input for max value
        let input_max = document.createElement("input");
        input_max.type = "number";
        input_max.placeholder = "max";
        input_max.value = vmax || "inf";
        container.appendChild(input_max);

        // Ordering
        const arrow = document.createElement("span");
        arrow.className = "metric-sort-arrow";
        if (order === "asc") {
            arrow.textContent = '▲';
            arrow.title = 'Ascending order';
        } else {
            arrow.textContent = '▼';
            arrow.title = 'Descending order';
        }
        arrow.addEventListener('click', () => {
            if (arrow.textContent === '▼') {
                arrow.textContent = '▲';
                arrow.title = 'Ascending order';
                container.classList.add('asc');
                container.classList.remove('desc');
            } else {
                arrow.textContent = '▼';
                arrow.title = 'Descending order';
                container.classList.add('desc');
                container.classList.remove('asc');
            }
        });
        container.classList.add('desc');
        container.appendChild(arrow);

        ranges.appendChild(container);
    };

    if (currentFilter !== null) {
        $select.val(currentFilter.map(option => option[0])).trigger("change");
        currentFilter.forEach(function (option) {
            add_range(option[0], option[1], option[2], option[3]);
        });
    }

    $select.off("select2:select");
    $select.off("select2:unselect");

    $select.on("select2:select", function (e) {
        console.log(e.params.data);
        const value = e.params.data.text;
        add_range(value, null, null);
    });

    $select.on("select2:unselect", function (e) {
        console.log(e.params.data);
        const value = e.params.data.text;
        const el = find_range(value);
        if (el) {
            el.remove();
        }
    });
}

function addClassifierSearchOptions(options) {
    window.classifierState.labelOptions = (options || []).slice();
    renderClassifierPanel();
}

let renderingClassifierPanel = false;

function renderClassifierPanel() {
    if (renderingClassifierPanel) return;
    renderingClassifierPanel = true;
    try {
        doRenderClassifierPanel();
    } finally {
        renderingClassifierPanel = false;
    }
}

function doRenderClassifierPanel() {
    const state = window.classifierState;

    const labelSel = document.getElementById("classifier-label");
    const negSel = document.getElementById("negative-labels");
    const embedSel = document.getElementById("classifier-embed-type");
    const runsSel = document.getElementById("classifier-runs-select");
    const trainBtn = document.getElementById("train-classifier-button");
    const useBtn = document.getElementById("use-classifier-button");
    const exportBtn = document.getElementById("export-classifier-button");
    const viewPosBtn = document.getElementById("view-positive-clips-button");
    const viewNegBtn = document.getElementById("view-negative-clips-button");
    const delBtn = document.getElementById("delete-classifier-run-button");
    const exprInput = document.getElementById("probability-expression");
    const autolabelCb = document.getElementById("train-with-autolabel");
    const nNegInput = document.getElementById("negative-samples");
    const nPosInput = document.getElementById("positive-samples-input");
    const nPosSamples = document.getElementById("positive-samples");
    if (!labelSel || !runsSel) return;

    // When selectedRunId changes, copy run metadata into state.form so
    // the form fields reflect the selected run. Subsequent renders
    // without a selection change don't re-copy — user edits persist.
    // If the run isn't yet in state.runs (page-load race: search
    // response sets selectedRunId before the /classifiers_status
    // poll returns) we leave lastRenderedSelectedRunId stale so the
    // next render syncs once the run becomes available.
    if (state.selectedRunId !== state.lastRenderedSelectedRunId) {
        if (state.selectedRunId) {
            const run = state.runs.find(
                r => r.run_id === state.selectedRunId
                    && r.status === "done"
            );
            if (run) {
                state.form.embedType = run.embed_type || "cosmos";
                state.form.positiveLabels =
                    (run.positive_labels || []).slice();
                state.form.negativeLabels =
                    (run.negative_labels || []).slice();
                state.form.useAutolabels = !!run.use_autolabels;
                if (run.n_negative_clips != null) {
                    state.form.nNegative = run.n_negative_clips;
                }
                if (run.n_positive_clips != null) {
                    state.form.nPositive = run.n_positive_clips;
                }
                state.lastRenderedSelectedRunId = state.selectedRunId;
            }
        } else {
            state.lastRenderedSelectedRunId = state.selectedRunId;
        }
    }

    const selectedRun = state.selectedRunId
        ? state.runs.find(
            r => r.run_id === state.selectedRunId
                && r.status === "done"
        )
        : null;

    const canTrain = state.canTrain;
    const positiveLabels = (state.form.positiveLabels || []).slice().sort();
    const negativeLabels = (state.form.negativeLabels || []).slice().sort();

    // Embed type.
    if (embedSel.value !== state.form.embedType) {
        embedSel.value = state.form.embedType;
    }

    // Positive labels select2 — rebuild options only when they change.
    const optsKey = JSON.stringify(state.labelOptions);
    if (state.lastLabelOptionsKey !== optsKey
        || !$(labelSel).data("select2")) {
        state.lastLabelOptionsKey = optsKey;
        labelSel.innerHTML = "";
        addOptionsToSelect(labelSel, state.labelOptions, null);
        if ($(labelSel).data("select2")) {
            $(labelSel).select2("destroy");
        }
        $(labelSel).select2({
            placeholder: "Select positive labels...",
            allowClear: true,
            multiple: true,
            width: "100%",
        });
        $(labelSel).off("change.classifierPanel")
            .on("change.classifierPanel", onPositiveLabelsChange);
    }
    const havePos = ($(labelSel).val() || []).slice().sort();
    if (JSON.stringify(havePos) !== JSON.stringify(positiveLabels)) {
        $(labelSel).val(positiveLabels).trigger("change.select2");
    }

    // Negative labels: available = labelOptions − positiveLabels.
    const availableNeg = (state.labelOptions || [])
        .filter(x => !positiveLabels.includes(x));
    const wantNeg = negativeLabels.filter(v => availableNeg.includes(v));
    const negOptsKey = JSON.stringify(availableNeg);
    if (state.lastNegOptionsKey !== negOptsKey
        || !$(negSel).data("select2")) {
        state.lastNegOptionsKey = negOptsKey;
        negSel.innerHTML = "";
        const placeholderOpt = document.createElement("option");
        placeholderOpt.text = "Select negative labels...";
        placeholderOpt.value = "";
        negSel.add(placeholderOpt);
        addOptionsToSelect(negSel, availableNeg, null);
        if ($(negSel).data("select2")) {
            $(negSel).select2("destroy");
        }
        $(negSel).select2({
            placeholder: "Select negative labels...",
            allowClear: true,
            multiple: true,
            width: "100%",
        });
        $(negSel).off("change.classifierPanel")
            .on("change.classifierPanel", onNegativeLabelsChange);
    }
    const haveNeg = ($(negSel).val() || []).slice().sort();
    if (JSON.stringify(haveNeg) !== JSON.stringify(wantNeg.slice().sort())) {
        $(negSel).val(wantNeg).trigger("change.select2");
    }

    // Autolabel checkbox.
    if (autolabelCb.checked !== state.form.useAutolabels) {
        autolabelCb.checked = state.form.useAutolabels;
    }

    // Negative samples.
    if (Number(nNegInput.value) !== state.form.nNegative) {
        nNegInput.value = state.form.nNegative ?? "";
    }

    // Probability expression.
    if (exprInput.value !== (state.form.probabilityExpression || "")) {
        exprInput.value = state.form.probabilityExpression || "";
    }

    // Positive samples — derived from annotation counts.
    if (positiveLabels.length > 0) {
        let nPos = 0;
        let nPosAuto = 0;
        for (const lbl of positiveLabels) {
            nPos += state.numAnnotations[lbl] || 0;
            nPosAuto += state.numAutolabelledAnnotations[lbl] || 0;
        }
        const maxPos = state.form.useAutolabels ? nPos : nPos - nPosAuto;
        const maxSpan = document.getElementById("positive-samples-max");
        if (maxSpan) maxSpan.textContent = `/ ${maxPos}`;
        nPosInput.max = maxPos;
        // Auto-fill only when the determining inputs change, so user
        // edits to the value aren't snapped back on every render.
        const fp = JSON.stringify({
            l: positiveLabels,
            a: state.form.useAutolabels,
            r: selectedRun?.run_id || null,
        });
        if (state.posSamplesFp !== fp) {
            state.posSamplesFp = fp;
            state.form.nPositive =
                selectedRun?.n_positive_clips || maxPos;
        }
        if (Number(nPosInput.value) !== state.form.nPositive) {
            nPosInput.value = state.form.nPositive;
        }
        nPosSamples.style.display = "flex";
    } else {
        nPosSamples.style.display = "none";
        state.posSamplesFp = null;
    }

    // Runs list — newest first.
    const allRuns = (state.runs || []).slice()
        .sort((a, b) => (b.started_at || 0) - (a.started_at || 0));
    runsSel.innerHTML = "";
    if (allRuns.length === 0) {
        const opt = document.createElement("option");
        opt.disabled = true;
        opt.value = "";
        opt.text = "No trained runs yet";
        runsSel.add(opt);
    } else {
        allRuns.forEach(run => {
            const opt = document.createElement("option");
            opt.value = run.run_id;
            const icon = run.status === "done" ? "✓"
                : run.status === "pending" ? "⏳"
                : "✗";
            const pos = (run.positive_labels || []).join(",") || "—";
            const neg = (run.negative_labels || []).join(",") || "—";
            const trained = run.trained_by ? ` · ${run.trained_by}` : "";
            // The server pre-stamps pending runs with n_*_clips=0
            // (the actual counts are computed by the training
            // subprocess), so show "training..." for pending runs
            // instead of the misleading "0+/0−".
            const counts = run.status === "done"
                ? `${run.n_positive_clips}+/${run.n_negative_clips}−`
                : run.status === "pending"
                    ? "training..."
                    : "failed";
            opt.textContent =
                `${icon} ${run.run_id} · pos=[${pos}] neg=[${neg}] `
                + `· ${run.embed_type || "cosmos"} · ${counts}${trained}`;
            if (run.search_params) opt.title = run.search_params;
            opt.disabled = run.status !== "done";
            if (run.run_id === state.selectedRunId) opt.selected = true;
            runsSel.add(opt);
        });
    }

    // Train is blocked only when an *equivalent* run is already
    // pending — a different (embed, pos, neg) combo can train
    // concurrently.
    const matchingPending = allRuns.some(r => {
        if (r.status !== "pending") return false;
        if ((r.embed_type || "cosmos") !== state.form.embedType) return false;
        const rp = (r.positive_labels || []).slice().sort();
        const rn = (r.negative_labels || []).slice().sort();
        return JSON.stringify(rp) === JSON.stringify(positiveLabels)
            && JSON.stringify(rn) === JSON.stringify(negativeLabels);
    });
    trainBtn.textContent = matchingPending
        ? "Training (Please wait)..."
        : "Train Classifier";

    const hasLabels = positiveLabels.length > 0;
    const setDisabled = (el, want) => {
        if (el && el.disabled !== want) el.disabled = want;
    };
    setDisabled(trainBtn, !canTrain || !hasLabels || matchingPending);
    setDisabled(nNegInput, !canTrain || !hasLabels);
    setDisabled(nPosInput, !canTrain || !hasLabels);
    setDisabled(autolabelCb, !canTrain || !hasLabels);
    setDisabled(embedSel, !canTrain);
    if (labelSel.disabled !== !canTrain) {
        $(labelSel).prop("disabled", !canTrain);
    }
    const negShouldDisable = !canTrain || !hasLabels;
    if (negSel.disabled !== negShouldDisable) {
        $(negSel).prop("disabled", negShouldDisable);
    }

    setDisabled(useBtn, !selectedRun);
    setDisabled(exportBtn, !selectedRun);
    setDisabled(exprInput, !selectedRun);
    setDisabled(delBtn, !state.selectedRunId);

    if (viewPosBtn) {
        const n = selectedRun?.n_positive_clips || 0;
        viewPosBtn.textContent = n
            ? `View Positives (${n.toLocaleString()})`
            : "View Positives";
        setDisabled(viewPosBtn, !selectedRun?.positive_clip_list_hash);
    }
    if (viewNegBtn) {
        const n = selectedRun?.n_negative_clips || 0;
        viewNegBtn.textContent = n
            ? `View Negatives (${n.toLocaleString()})`
            : "View Negatives";
        setDisabled(viewNegBtn, !selectedRun?.negative_clip_list_hash);
    }

    // The autolabel-label input lives outside the classifier panel
    // (annotation page only) but mirrors the positive labels.
    const autolabelLabel = document.getElementById("autolabel-label");
    if (autolabelLabel) {
        autolabelLabel.value = positiveLabels.join("&&");
    }
}

function onPositiveLabelsChange() {
    if (renderingClassifierPanel) return;
    const v = $(document.getElementById("classifier-label")).val() || [];
    const state = window.classifierState;
    // Clearing all positives means the rest of the training config no
    // longer makes sense, so fall back to a fresh form instead of
    // leaving stale grayed-out negatives / autolabel / sample counts.
    if (v.length === 0) {
        resetClassifierForm();
    } else {
        state.form.positiveLabels = v.slice();
    }
    state.selectedRunId = null;
    renderClassifierPanel();
}

function onNegativeLabelsChange() {
    if (renderingClassifierPanel) return;
    const v = $(document.getElementById("negative-labels")).val() || [];
    window.classifierState.form.negativeLabels = v.slice();
    window.classifierState.selectedRunId = null;
    renderClassifierPanel();
}

function onEmbedTypeChange() {
    if (renderingClassifierPanel) return;
    window.classifierState.form.embedType =
        document.getElementById("classifier-embed-type").value;
    window.classifierState.selectedRunId = null;
    renderClassifierPanel();
}

function onAutolabelToggle() {
    if (renderingClassifierPanel) return;
    window.classifierState.form.useAutolabels =
        document.getElementById("train-with-autolabel").checked;
    window.classifierState.selectedRunId = null;
    renderClassifierPanel();
}

function onNegativeSamplesInput() {
    const v = parseInt(
        document.getElementById("negative-samples").value, 10);
    const state = window.classifierState;
    if (Number.isFinite(v)) {
        state.form.nNegative = v;
    }
    // Any edit to a training-config param forks from the run.
    if (state.selectedRunId !== null) {
        state.selectedRunId = null;
        renderClassifierPanel();
    }
}

function onPositiveSamplesInput() {
    const el = document.getElementById("positive-samples-input");
    const max = parseInt(el.max, 10);
    if (el.value && Number.isFinite(max) && parseInt(el.value, 10) > max) {
        el.value = max;
    }
    const v = parseInt(el.value, 10);
    const state = window.classifierState;
    if (Number.isFinite(v)) {
        state.form.nPositive = v;
    }
    if (state.selectedRunId !== null) {
        state.selectedRunId = null;
        renderClassifierPanel();
    }
}

function onProbabilityExpressionInput() {
    window.classifierState.form.probabilityExpression =
        document.getElementById("probability-expression").value;
}

function onClassifierRunsSelectChange() {
    if (renderingClassifierPanel) return;
    const v = document.getElementById("classifier-runs-select").value || null;
    window.classifierState.selectedRunId = v;
    // Force re-sync even if the user re-picked the same run after
    // editing the form.
    window.classifierState.lastRenderedSelectedRunId = undefined;
    renderClassifierPanel();
}

// One-time event binding for the classifier panel. Each page calls
// this once on DOMContentLoaded with its own canTrain value.
function initClassifierPanel(opts) {
    opts = opts || {};
    window.classifierState.canTrain = opts.canTrain !== false;

    document.getElementById("classifier-embed-type")
        .addEventListener("change", onEmbedTypeChange);
    document.getElementById("train-with-autolabel")
        .addEventListener("click", onAutolabelToggle);
    document.getElementById("negative-samples")
        .addEventListener("input", onNegativeSamplesInput);
    document.getElementById("positive-samples-input")
        .addEventListener("input", onPositiveSamplesInput);
    document.getElementById("probability-expression")
        .addEventListener("input", onProbabilityExpressionInput);
    document.getElementById("classifier-runs-select")
        .addEventListener("change", onClassifierRunsSelectChange);

    checkClassifierState();
}

function addProjectSourceOptions(options, currentProjectSource) {
    let select = document.getElementById("project-select");
    select.innerHTML = "";
    let defaultOpt = document.createElement("option");
    defaultOpt.text = "All";
    defaultOpt.value = "";
    select.add(defaultOpt);
    addOptionsToSelect(select, options, currentProjectSource);

    if ($(select).data("select2")) {
        $(select).select2("destroy");
    }
    $(select).select2({
        placeholder: "Load annotations from projects...",
        allowClear: true,
    });
    // Set the value after Select2 initialization if a filter is already applied
    if (currentProjectSource) {
        $(select).val(currentProjectSource).trigger("change");
    }
}

const NVIDIA_ICON_SVG = '<img src="https://cdn.simpleicons.org/nvidia/76b900" width="12" height="12" style="vertical-align:middle;margin-right:5px;flex-shrink:0"/>';
const LOCK_ICON = '<span style="margin-right:5px;font-size:11px">🔒</span>';
const GLOBE_ICON = '<span style="margin-right:5px;font-size:11px">🌐</span>';

function datasetTemplateResult(state) {
    if (!state.id) return state.text;
    const meta = (window.currentDatasetMetadata || {})[state.text];
    const lic = meta && meta.license;
    let icon = "";
    if (lic === "licensed")      icon = LOCK_ICON;
    else if (lic === "internal") icon = NVIDIA_ICON_SVG;
    else if (lic === "public")   icon = GLOBE_ICON;
    return $('<span style="display:inline-flex;align-items:center">').html(icon + DOMPurify.sanitize(state.text));
}

function addGroupedDataSourceOptions(select, options, selected) {
    const metadata = window.currentDatasetMetadata || {};
    const groups = {};
    for (const name of options) {
        const cat = (metadata[name] && metadata[name].category) || "Other";
        (groups[cat] = groups[cat] || []).push(name);
    }
    for (const cat of Object.keys(groups).sort()) {
        const optgroup = document.createElement("optgroup");
        optgroup.label = cat;
        for (const name of groups[cat].sort()) {
            const opt = document.createElement("option");
            opt.text = name;
            opt.value = name;
            if (name === selected) opt.selected = true;
            optgroup.appendChild(opt);
        }
        select.appendChild(optgroup);
    }
}

function addDataSourceOptions(options, currentDataSource) {
    let select = document.getElementById("data-source");
    select.innerHTML = "";
    let defaultOpt = document.createElement("option");
    defaultOpt.text = "All";
    defaultOpt.value = "";
    select.add(defaultOpt);
    addGroupedDataSourceOptions(select, options, currentDataSource);

    if ($(select).data("select2")) {
        $(select).select2("destroy");
    }
    $(select).select2({
        placeholder: "Select dataset...",
        allowClear: true,
        templateResult: datasetTemplateResult,
        templateSelection: datasetTemplateResult,
    });
    $(select).off("change.dataSourceMode").on("change.dataSourceMode", () => syncModeControl("data-source-mode-row", "#data-source", "currentDataSourceMode"));
    // Set the value after Select2 initialization if a filter is already applied
    if (currentDataSource) {
        $(select).val(currentDataSource).trigger("change");
    }
    syncModeControl("data-source-mode-row", "#data-source", "currentDataSourceMode");
}

function addLabelTypeOptions(options, currentLabelTypes) {
    let select = document.getElementById("label-types");
    select.innerHTML = "";
    let defaultOpt = document.createElement("option");
    defaultOpt.text = "All";
    defaultOpt.value = "";
    select.add(defaultOpt);
    // Add label type options with user-friendly labels while preserving values
    const displayName = (v) => {
        switch (v) {
            case "manual": return "Manual";
            case "autolabel": return "Autolabels";
            default: return v;
        }
    };
    for (let i = 0; i < options.length; i++) {
        const v = options[i];
        const opt = document.createElement("option");
        opt.value = v;
        opt.text = displayName(v);
        select.add(opt);
    }

    if ($(select).data("select2")) {
        $(select).select2("destroy");
    }
    $(select).select2({
        placeholder: "Select annotation types...",
        allowClear: true,
    });
    // Set the value after Select2 initialization if a filter is already applied
    if (window.currentLabelTypes) {
        $(select).val(currentLabelTypes).trigger("change");
    }

    if (currentLabelTypes !== null) {
        $(select).val(currentLabelTypes).trigger("change");
    }
}


function showContextualFilters() {
    const chips = [
        {
            active: window.currentTrajectoryShapeClipID,
            id: "trajectory-shape-search",
            label: () => {
                let text = `Trajectory Shape Search: ${window.currentTrajectoryShapeClipID}`;
                if (window.currentTrajectoryShapeStartT) text += `: ${window.currentTrajectoryShapeStartT}s - `;
                if (window.currentTrajectoryShapeEndT) text += `${window.currentTrajectoryShapeEndT}s`;
                return text;
            }
        },
        {
            active: window.currentSemanticSearchClipID,
            id: "semantic-search-video-display",
            label: () => `Video-to-Video Search: ${window.currentSemanticSearchClipID}`
        },
        {
            active: window.currentSemanticSearchText,
            id: "semantic-search-text-display",
            label: () => {
                var label = `Text-to-Video Search: ${window.currentSemanticSearchText}`;
                if (window.currentSemanticExtraQueries && window.currentSemanticExtraQueries.length) {
                    label += ` + ${window.currentSemanticExtraQueries.join(", ")}`;
                }
                return label;
            }
        },
        {
            active: window.currentClassifierSearch.run_id,
            id: "classifier-search",
            label: () => {
                const cs = window.currentClassifierSearch;
                const run = (window.classifierStatuses?.runs || [])
                    .find(r => r.run_id === cs.run_id);
                const embedKey = run?.embed_type || "cosmos";
                const embed = (
                    embedKey === "caption" ? "Caption"
                    : embedKey === "visual" ? "Visual"
                    : "Text-to-Video"
                );
                const labelText = run
                    ? (run.positive_labels || []).slice().sort().join("&&")
                    : cs.run_id;
                return `Classifier Search [${embed}]: ${labelText} — ${cs.expression}`;
            }
        },
        {
            active: window.currentClusterSearch.cluster_ids?.length,
            id: "cluster-search",
            label: () => {
                const ids = window.currentClusterSearch.cluster_ids || [];
                return ids.length === 1
                    ? `Cluster Search: Cluster ${ids[0]}`
                    : `Cluster Search: ${ids.length} clusters`;
            }
        },
        {
            active: window.currentClipList.hash,
            id: "clip-list-search",
            label: () => {
                const cl = window.currentClipList;
                // The hash is opaque to the user — only the count is
                // informative in the chip. The hash lives in the URL
                // for sharing / bookmarking.
                return cl.count
                    ? `Clip List Search: ${cl.count.toLocaleString()} clips`
                    : `Clip List Search`;
            }
        },
        {
            active: window.currentVisualSearchText,
            id: "visual-search-text-display",
            label: () => {
                var label = `Text-to-Image Search: ${window.currentVisualSearchText}`;
                if (window.currentVisualExtraQueries && window.currentVisualExtraQueries.length) {
                    label += ` + ${window.currentVisualExtraQueries.join(", ")}`;
                }
                return label;
            }
        },
        {
            active: window.currentVisualSearchImageId,
            id: "visual-search-image-display",
            label: () => `Image-to-Video Search`
        },
        {
            active: window.currentCaptionEmbedSearchText,
            id: "caption-embed-search-display",
            label: () => {
                var label = `Caption Semantic Search: ${window.currentCaptionEmbedSearchText}`;
                if (window.currentCaptionEmbedExtraQueries && window.currentCaptionEmbedExtraQueries.length) {
                    label += ` + ${window.currentCaptionEmbedExtraQueries.join(", ")}`;
                }
                return label;
            }
        },
        {
            active: window.currentSearch,
            id: "caption-search",
            label: () => {
                var captionLabel = `Caption Substring Search: ${window.currentSearch}`;
                if (window.currentExtraQueries && window.currentExtraQueries.length) {
                    captionLabel += ` + ${window.currentExtraQueries.join(", ")}`;
                }
                return captionLabel;
            }
        },
        {
            active: window.currentWMClassName,
            id: "wm-search",
            label: () => {
                const sel = document.getElementById("wm-class-name");
                const opt = Array.from(sel.options).find(o => o.value === window.currentWMClassName);
                const classText = (opt && opt.text) ? opt.text : window.currentWMClassName;
                const parts = [classText];
                if (window.currentWMMinCount || window.currentWMMaxCount) {
                    const min = window.currentWMMinCount ?? '';
                    const max = window.currentWMMaxCount ?? '';
                    if (min || max) parts.push(`Count: ${min}${min && max ? '–' : ''}${max}`);
                }
                if (window.currentWMMaxDist != null) parts.push(`Dist ≤ ${window.currentWMMaxDist}m`);
                if (window.currentWMMinTime != null) parts.push(`Time ≥ ${window.currentWMMinTime}s`);
                if (window.currentWMAngleRange) {
                    const ang = Array.isArray(window.currentWMAngleRange)
                        ? window.currentWMAngleRange.join(", ")
                        : String(window.currentWMAngleRange).split("||").filter(Boolean).join(", ");
                    if (ang) parts.push(`Angles: ${ang}`);
                }
                return `Perception-based Search: ${parts.join(' | ')}`;
            }
        },
    ];

    const anyActive = chips.some(c => c.active);
    document.getElementById("contextual-filters-container").style.display = anyActive ? "" : "none";

    const updateChip = function (el, label) {
        el.parentElement.style.display = "flex";
        el.style.display = "flex";
        el.querySelector("label").textContent = label;
    };

    for (const { active, id, label } of chips) {
        const el = document.getElementById(id);
        if (active) updateChip(el, label());
        else el.style.display = "none";
    }

    // positive-samples is not a chip but must also be hidden when classifier is inactive
    if (!window.currentClassifierSearch.run_id) {
        document.getElementById("positive-samples").style.display = "none";
    }

    // Sync trajectory form inputs when active, clear them when not
    if (window.currentTrajectoryShapeClipID) {
        document.getElementById("trajectory-shape-clipid").value = window.currentTrajectoryShapeClipID;
        if (window.currentTrajectoryShapeStartT) {
            document.getElementById("trajectory-shape-start-t").value = window.currentTrajectoryShapeStartT;
            updateTrajectoryShapeStartTimeValue(window.currentTrajectoryShapeStartT);
        }
        if (window.currentTrajectoryShapeEndT) {
            document.getElementById("trajectory-shape-end-t").value = window.currentTrajectoryShapeEndT;
            updateTrajectoryShapeEndTimeValue(window.currentTrajectoryShapeEndT);
        }
    } else {
        document.getElementById("trajectory-shape-clipid").value = "";
    }

    // Sync semantic video input when active, clear it when not
    if (window.currentSemanticSearchClipID) {
        document.getElementById("semantic-search-clipid").value = window.currentSemanticSearchClipID;
    } else {
        document.getElementById("semantic-search-clipid").value = "";
        document.getElementById("semantic-search-clipid").disabled = false;
    }

    // Clear WM inputs when inactive so they don't bleed into new tab searches
    if (!window.currentWMClassName) {
        document.getElementById("wm-class-name").value = "";
        document.getElementById("wm-angle-range").selectedIndex = 0;
        document.getElementById("wm-min-count").value = "1";
        document.getElementById("wm-max-count").value = "500";
        document.getElementById("wm-max-dist").value = "10";
        document.getElementById("wm-min-time").value = "0";
        updateWMDistanceValue("10");
        updateWMTimeValue("0");
        resetWMAngleSelector();
    }

    updateRRFToggleVisibility();
}


function clearSearch() {
    resetTrajectorySearch(true);
    document.getElementById("trajectory-pattern").selectedIndex = 0;
    window.currentTrajectoryPattern = null;

    resetSemanticSearchVideo(true);
    resetSemanticSearchText(true);
    resetVisualSearch(true);
    resetCaptionEmbedSearch(true);
    resetCaptionSearch(true);
    resetWMSearch(true);

    window.currentDataSource = null;
    window.currentLabelTypes = null;
    window.currentSILAPIs = null;

    // Close World Model search panel
    const wmContainer = document.getElementById("wm-search-container");
    const wmBtn = document.querySelector(".wm-button-search");
    if (wmContainer) {
        wmContainer.style.display = "none";
    }
    if (wmBtn) {
        wmBtn.classList.remove("selected");
    }

    // Clear classifier search state and close tools panel
    resetClassifierSearch(true);
    const classifierToolsBtn = document.getElementById("classifier-tools-btn");
    const classifierContainer = document.getElementById("classifier-container");
    if (classifierContainer) {
        classifierContainer.style.display = "none";
    }
    if (classifierToolsBtn) {
        classifierToolsBtn.classList.remove("selected");
    }

    // Clear cluster search state and close tools panel
    resetClusterSearch(true);
    const clusteringToolsBtn = document.getElementById("clustering-tools-btn");
    const clusteringContainer = document.getElementById("clustering-container");
    if (clusteringContainer) {
        clusteringContainer.style.display = "none";
    }
    if (clusteringToolsBtn) {
        clusteringToolsBtn.classList.remove("selected");
    }

    const shortcuts = document.getElementById("shortcuts");
    if (shortcuts) {
        shortcuts.value = "";
        window.labelShortcuts = [];
    }

    window.currentRankMode = "priority";
    const rrfBox = document.getElementById("rrf-toggle-checkbox");
    if (rrfBox) rrfBox.checked = false;

    showPage(0, {});
    window.scrollTo(0, 0);
}


function addLabelsToExclude(options, currentLabelsToExclude) {
    let labelsToExclude = document.getElementById("labels-to-exclude-choices");
    labelsToExclude.innerHTML = "";
    defaultOpt = document.createElement("option");
    defaultOpt.text = "Select labels to exclude...";
    defaultOpt.value = "";
    labelsToExclude.add(defaultOpt);
    addOptionsToSelect(labelsToExclude, options, null);
    if ($(labelsToExclude).data("select2")) {
        $(labelsToExclude).select2("destroy");
    }
    $(labelsToExclude).select2({
        placeholder: "Select labels to exclude...",
        allowClear: true,
        multiple: true,
        width: "100%"
    });
    $(labelsToExclude).off("change.labelsToExcludeMode").on("change.labelsToExcludeMode", () => syncModeControl("labels-to-exclude-mode-row", "#labels-to-exclude-choices", "currentLabelsToExcludeMode"));

    // Set the value after Select2 initialization if a filter is already applied
    if (currentLabelsToExclude !== null) {
        $(labelsToExclude).val(currentLabelsToExclude).trigger("change");
    }
    syncModeControl("labels-to-exclude-mode-row", "#labels-to-exclude-choices", "currentLabelsToExcludeMode");
}

function toggleAnnotationFilterMenu() {
    const button = document.querySelector('.gray-button[onclick="toggleAnnotationFilterMenu()"]');
    const menu = document.getElementById('clip-label-group');
    
    button.classList.toggle('expanded');
    menu.classList.toggle('expanded');
}

function toggleSILAPIsMenu() {
    const button = document.querySelector('.gray-button[onclick="toggleSILAPIsMenu()"]');
    const menu = document.getElementById('sil-api-group');
    
    button.classList.toggle('expanded');
    menu.classList.toggle('expanded');
}


function toggleCaptionSearchMenu() {
    const button = document.querySelector('.gray-button[onclick="toggleCaptionSearchMenu()"]');
    const menu = document.getElementById('caption-search-filters-group');
    if (!button || !menu) return;
    button.classList.toggle('expanded');
    menu.classList.toggle('expanded');
}

function toggleVisualSearchMenu() {
    const button = document.querySelector('.gray-button[onclick="toggleVisualSearchMenu()"]');
    const menu = document.getElementById('visual-search-filters-group');
    if (!button || !menu) return;
    button.classList.toggle('expanded');
    menu.classList.toggle('expanded');
}

function toggleTrajectorySearchMenu() {
    const button = document.querySelector('.gray-button[onclick="toggleTrajectorySearchMenu()"]');
    const menu = document.getElementById('trajectory-search-filters');
    if (!button || !menu) return;
    button.classList.toggle('expanded');
    menu.classList.toggle('expanded');
}

function validateProjectName() {
    const projectInput = document.getElementById("save-project-name");
    const name = projectInput.value.trim();

    if (!name) {
        projectInput.focus();
        return false;
    }
    return true;
}

function showProjectToWriteHelp() {
    let overlay = document.getElementById("project-help-overlay");
    if (!overlay) {
      overlay = document.createElement("div");
      overlay.id = "project-help-overlay";
      overlay.onclick = hideProjectToWriteHelp;
      document.body.appendChild(overlay);
    }
    overlay.style.display = "block";

    // Show modal (ensure it's at the top-level stacking context)
    const box = document.getElementById("project-to-write-help");
    if (box && box.parentElement !== document.body) {
      document.body.appendChild(box); // escape any parent with lower z-index/overflow
    }
    box.style.display = "block"; // use block, not flex, to avoid tiny sizing

    // Nudge focus to the input
    const input = document.getElementById("save-project-name");
    if (input) {
      input.classList.add("input-pulse");
      setTimeout(() => input.classList.remove("input-pulse"), 1200);
      input.focus();
    }
}

function hideProjectToWriteHelp() {
    const box = document.getElementById("project-to-write-help");
    if (box) box.style.display = "none";
    const overlay = document.getElementById("project-help-overlay");
    if (overlay) overlay.style.display = "none";
}

function getProjectToWrite() {
    const el = document.getElementById("save-project-name");
    return (el?.value || "").trim();
}

function logMissingProjectToWrite(context) {
    showProjectToWriteHelp();
    const input = document.getElementById("save-project-name");
    if (input) input.focus();
}

function closeCaptionBoxes() {
    document.querySelectorAll('.caption-box').forEach(box => {
        box.style.display = 'none';
    });
}



//// Common leaderboard functions

function getTabId() {
    if (window._currentTabId === undefined) {
        window._currentTabId = 0;
    }
    window._currentTabId += 1;
    return window._currentTabId;
}

function makeDefaultTab() {
    // TODO: Figure out a better way to set the default leaderboard and path
    id = getTabId();
    return {
        "id": `search-tab-${id}`,
        "path": "/metrics?",
        "label": "Policy - All clips",
        "leaderboard": "Policy",
        "selectedModels": null,
        "metrics": null,
        "search": null,
    };
}

function tabToURLString(tab) {
    let s = "";
    s += "label=";
    s += encodeURIComponent(tab.label);
    s += "&path=";
    s += encodeURIComponent(tab.path);
    s += "&leaderboard=";
    s += encodeURIComponent(tab.leaderboard);
    if (tab.selectedModels !== null) {
        s += "&selectedModels=";
        s += tab.selectedModels.map(encodeURIComponent).join(",");
    }
    return s;
}

function tabFromURLString(tabString) {
    let tab = makeDefaultTab();
    tabString.split("&").forEach(function (x) {
        let [k, v] = x.split("=");
        tab[k] = decodeURIComponent(v);
    });

    if (tab.selectedModels !== null) {
        tab.selectedModels = tab.selectedModels.split(",").map(decodeURIComponent);
    }

    if (tab.path === null) {
        // TODO: Find out what to do with the error
    }

    return tab;
}

function tabKey(tab) {
    if (tab.key === undefined || tab.key === null) {
        let k = "";
        k += tab.path;
        k += "&leaderboard=" + encodeURIComponent(tab.leaderboard);
        if (tab.selectedModels !== null) {
            k += "&selectedModels=";
            k += tab.selectedModels.map(encodeURIComponent).join(",");
        }
        tab.key = k;
    }
    return tab.key;
}

function navigateToLeaderboard() {
    const tab = makeDefaultTab();
    tab.path = stripQueryKeysFromPath(buildCurrentEndpoint("/metrics"), ["page", "n"]);
    tab.label = "From search";

    let url = "/leaderboard";
    url += "#tabs=" + encodeURIComponent(tabToURLString(tab));
    url += "&active=" + encodeURIComponent(tabKey(tab));

    window.open(url, "_blank", "noopener,noreferrer");

    return false;
}

function tabLabel(tab) {
    let [_1, _2, id] = tab.id.split("-");

    let searchName = `Search ${id}`;

    let queryParams = getQueryParams(tab.path.split("?")[1]);
    let reduction = queryParams["reduction"] || null;
    delete queryParams["reduction"];
    let projectSource = queryParams["project_source"] || null;
    delete queryParams["project_source"];

    if (Object.keys(queryParams).length == 0) {
        searchName = "All clips";
    }

    else if (Object.keys(queryParams).length == 1) {
        if (queryParams.filter !== undefined) {
            let label = queryParams.filter.split("||");
            if (label.length == 1) {
                searchName = `Label: ${label[0]}`;
            }
        }

        if (queryParams.search !== undefined) {
            searchName = `Caption: ${queryParams.search}`;
        }

        if (queryParams.semantic_search_text !== undefined) {
            searchName = `Text-to-Video Search: ${queryParams.semantic_search_text}`;
        }

        if (queryParams.semantic_search_clipid !== undefined) {
            searchName = `Video-to-Video Search: ${queryParams.semantic_search_clipid}`;
        }
    }

    return `${tab.leaderboard} - ${searchName}`;
}

function openBugReportModal() {
    document.getElementById("bug-report-title").value = "";
    document.getElementById("bug-report-description").value = "";
    document.getElementById("bug-report-feedback").textContent = "";
    document.getElementById("bug-report-modal").style.display = "block";
    document.getElementById("bug-report-overlay").style.display = "block";
}

function closeBugReportModal() {
    document.getElementById("bug-report-modal").style.display = "none";
    document.getElementById("bug-report-overlay").style.display = "none";
}

function submitBugReport() {
    const title = document.getElementById("bug-report-title").value.trim();
    const description = document.getElementById("bug-report-description").value.trim();
    const feedback = document.getElementById("bug-report-feedback");
    if (!title) {
        feedback.textContent = "Please enter a title.";
        return;
    }
    const payload = ["report_bug", title, description, navigator.userAgent, window.location.href].join("::");
    const req = new XMLHttpRequest();
    req.addEventListener("load", () => {
        if (req.status !== 200) {
            feedback.textContent = `Server error (${req.status})`;
            return;
        }
        const data = JSON.parse(req.responseText);
        if (data.ok) {
            feedback.textContent = "Bug report submitted. Thank you!";
            setTimeout(closeBugReportModal, 1500);
        } else {
            feedback.textContent = `Submission failed: ${data.error || "unknown error"}`;
        }
    });
    req.addEventListener("error", () => {
        feedback.textContent = "Error communicating with the server";
    });
    req.open("POST", "");
    req.setRequestHeader("Content-Type", "text/plain");
    req.send(payload);
}

// ── Shared Modals (Agent Setup + Contact) ──────────────────────
// Injected once via JS so the HTML lives in a single place across all pages.

function toggleModal(id, show) {
    document.getElementById(id + "-overlay").style.display = show ? "block" : "none";
    document.getElementById(id + "-modal").style.display = show ? "block" : "none";
}
function openAgentSetupModal() { toggleModal("agent-setup", true); }
function closeAgentSetupModal() { toggleModal("agent-setup", false); }
function openContactModal() { toggleModal("contact", true); }
function closeContactModal() { toggleModal("contact", false); }

function copyAgentSetupPrompt() {
    var text = document.getElementById("agent-setup-prompt").textContent;
    navigator.clipboard.writeText(text).then(function() {
        var btn = document.querySelector(".agent-copy-btn");
        btn.textContent = "✅ Copied!";
        setTimeout(function() { btn.textContent = "📋 Copy"; }, 1500);
    });
}

function injectSharedModals() {
    if (document.getElementById("agent-setup-modal")) return;
    var html = ''
        + '<div class="modal-overlay" id="agent-setup-overlay" onclick="closeAgentSetupModal()" style="display:none;"></div>'
        + '<div class="help-box agent-setup-modal" id="agent-setup-modal" style="display:none; max-width:560px;">'
        +   '<span class="clear-annotation-button" onclick="closeAgentSetupModal()">&times;</span>'
        +   '<h3>🤖 SIL Wheel Agent</h3>'
        +   '<p>Interact with the SIL Wheel directly from your IDE using an AI agent. '
        +   'Describe what you\'re looking for in plain English and it handles search composition, '
        +   'clip&nbsp;ID export, and everything in between.</p>'
        +   '<p style="color:var(--muted); font-size:0.9em;">'
        +   'Example: <em>"Find me 100 clips of construction zones in rain for MADS-1M"</em> '
        +   '&ndash; the agent combines caption search, classifiers, and semantic search, and exports a clip&nbsp;ID list '
        +   'ready for your training pipeline.</p>'
        +   '<h4>Setup</h4>'
        +   '<p style="font-size:0.9em;">No git clone needed. Works from any project folder. Open your project in '
        +   'Cursor, Claude Code, or another agent-enabled IDE and give it this prompt:</p>'
        +   '<div class="agent-setup-prompt-block">'
        +     '<button class="agent-copy-btn" onclick="copyAgentSetupPrompt()" title="Copy to clipboard">📋 Copy</button>'
        +     '<pre id="agent-setup-prompt">Set up the SIL Wheel Agent by following the skill file at '
        +     'https://research.nvidia.com/labs/sil/projects/sil-wheel-agent/skill.md - curl (not fetch) it and follow the install instructions. '
        +     'My username is "[USERNAME]" and my password is "[PASSWORD]". Find a clip of a stroller passing in front of the car.</pre>'
        +   '</div>'
        +   '<p style="font-size:0.85em; color:var(--muted);">Replace <code>[USERNAME]</code> and <code>[PASSWORD]</code> with your SIL Wheel credentials.</p>'
        + '</div>'
        + '<div class="modal-overlay" id="contact-overlay" onclick="closeContactModal()" style="display:none;"></div>'
        + '<div class="help-box" id="contact-modal" style="display:none; max-width:480px;">'
        +   '<span class="clear-annotation-button" onclick="closeContactModal()">&times;</span>'
        +   '<h3>📬 Contact</h3>'
        +   '<p>The SIL Wheel is developed and maintained by the '
        +   '<a href="https://research.nvidia.com/labs/toronto-ai/" target="_blank" rel="noopener">NVIDIA Spatial Intelligence Lab</a>.</p>'
        +   '<table class="contact-table">'
        +     '<tr><td><strong>General questions</strong></td>'
        +     '<td><a href="https://nvidia.enterprise.slack.com/archives/C09HP1PPTRR" target="_blank" rel="noopener">#sil-wheel</a> on Slack</td></tr>'
        +     '<tr><td><strong>Development</strong></td>'
        +     '<td><a href="https://nvidia.enterprise.slack.com/archives/C0AAH7EK22C" target="_blank" rel="noopener">#sil-wheel-dev</a> on Slack</td></tr>'
        +     '<tr><td><strong>Urgent issues</strong></td>'
        +     '<td>Despoina Paschalidou via Slack DM or <a href="mailto:dpaschalidou@nvidia.com">dpaschalidou@nvidia.com</a></td></tr>'
        +   '</table>'
        + '</div>';
    document.body.insertAdjacentHTML("beforeend", html);
}

document.addEventListener("DOMContentLoaded", injectSharedModals);

document.addEventListener("DOMContentLoaded", function() {
    const box = document.getElementById("rrf-toggle-checkbox");
    if (!box) return;
    box.addEventListener("change", function(e) {
        window.currentRankMode = e.target.checked ? "rrf" : "priority";
        if (typeof search === "function") search();
    });
});

document.addEventListener("keydown", function(e) {
    if (e.key === "Escape") hideProjectToWriteHelp();
});

// Draws a small fixed compass in the top-right corner of a trajectory plot's
// inner group, so the viewer can read forward/left/right off the canvas
// without inspecting axis labels. Forward is always up because the plots
// render the canonical ego frame with forward on the vertical axis.
window.drawTrajectoryCompass = function(parentG, width, height) {
    const size = 48;
    const inset = 8;
    const half = size / 2;
    const cx = width - inset - half;
    const cy = inset + half;

    const compass = parentG.append("g")
        .attr("class", "trajectory-compass")
        .attr("transform", `translate(${cx}, ${cy})`)
        .attr("pointer-events", "none");

    compass.append("rect")
        .attr("x", -half).attr("y", -half)
        .attr("width", size).attr("height", size)
        .attr("rx", 4).attr("ry", 4)
        .attr("fill", "white").attr("fill-opacity", 0.85)
        .attr("stroke", "#888").attr("stroke-width", 0.5);

    const armLen = 14;
    const stroke = "#333";

    const addArrow = (x1, y1, x2, y2, head) => {
        compass.append("line")
            .attr("x1", x1).attr("y1", y1).attr("x2", x2).attr("y2", y2)
            .attr("stroke", stroke).attr("stroke-width", 1.2);
        compass.append("polygon")
            .attr("points", head)
            .attr("fill", stroke);
    };

    // Forward (up)
    addArrow(0, 2, 0, -armLen, `0,${-armLen - 4} -3,${-armLen + 2} 3,${-armLen + 2}`);
    // Left
    addArrow(-2, 0, -armLen, 0, `${-armLen - 4},0 ${-armLen + 2},-3 ${-armLen + 2},3`);
    // Right
    addArrow(2, 0, armLen, 0, `${armLen + 4},0 ${armLen - 2},-3 ${armLen - 2},3`);

    const label = (x, y, anchor, text) => {
        compass.append("text")
            .attr("x", x).attr("y", y)
            .attr("text-anchor", anchor)
            .attr("font-size", "9px")
            .attr("fill", "#333")
            .text(text);
    };
    label(0, -armLen - 6, "middle", "F");
    label(-armLen - 6, 3, "end", "L");
    label(armLen + 6, 3, "start", "R");
};
