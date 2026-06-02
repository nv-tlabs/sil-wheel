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

// Keep track of metric-search tabs; start empty and add default only if none exist
// See makeDefaultTab() in main.js for the schema.
window.currentTabs = {};
window.activeTab = null;

// Keeps the models that are allowed for each leaderboard option
window.modelsByLeaderboard = null; // { leaderboard: [models...] }

function getSelectedModels(leaderboard) {
    const select = document.getElementById("models-select");
    let models = Array.from(select.selectedOptions).map(o => o.value);
    let allModels = window.modelsByLeaderboard[leaderboard];
    models = models.filter(m => allModels.includes(m));
    if (models.length == 0 || models.length == allModels.length) {
        return null;
    }
    return models;
}

function addNewTab() {
    const tab = makeDefaultTab();
    if (tabKey(tab) in window.currentTabs) {
        activateTab(window.currentTabs[tabKey(tab)]);
        return;
    }

    window.currentTabs[tabKey(tab)] = tab;
    window.activeTab = tabKey(tab);
    showLoading("Loading...");
    fetchTabData(tab).then(() => {
        render();
        hideLoading();
    });
}

function setGlobalSearchFlags(search) {
    window.currentOptions = search.options;
    window.currentFilter = search.filter;
    window.currentTimes = search.times;
    window.currentLeftHandDriving = search.leftHandDriving;
    window.currentWithoutAnn = search.withoutAnn;
    window.currentSearch = search.search;
    window.currentExtraQueries = search.extraQueries || [];
    window.currentCaptionEmbedExtraQueries = search.captionEmbedExtraQueries || [];
    window.currentSemanticExtraQueries = search.semanticExtraQueries || [];
    window.currentVisualExtraQueries = search.visualExtraQueries || [];
    window.currentSpeedQuery = search.speedQuery;
    window.currentCountryQuery = search.countryQuery;
    window.currentClipIDQuery = search.clipIDQuery;
    window.currentWithEgoData = search.withEgoData;
    window.currentTrajectoryPattern = search.trajectoryPattern;
    window.currentTrajectoryShapeClipID = search.trajectoryShapeClipID;
    window.currentTrajectoryShapeStartT = search.trajectoryShapeStartT;
    window.currentTrajectoryShapeEndT = search.trajectoryShapeEndT;
    window.currentSemanticSearchClipID = search.semanticSearchClipID;
    window.currentSemanticSearchText = search.semanticSearchText;
    window.currentVisualSearchText = search.visualSearchText;
    window.currentVisualSearchImageId = search.visualSearchImageId || null;
    window.currentCaptionEmbedSearchText = search.captionEmbedSearchText;
    window.currentLabelTypes = search.labelTypes;
    window.currentSearchTermInComments = search.searchTermInComments;
    window.currentWMClassName = search.wmClassName;
    window.currentWMMinCount = search.wmMinCount;
    window.currentWMMaxCount = search.wmMaxCount;
    window.currentWMMaxDist = search.wmMaxDist;
    window.currentWMMinTime = search.wmMinTime;
    window.currentWMAngleRange = search.wmAngleRange;
    window.currentDataSourceOptions = search.dataSourceOptions;
    window.currentDatasetMetadata = search.datasetMetadata;
    window.currentDataSource = search.dataSource;
    window.currentProjectSource = search.projectSource;
    window.currentLabelsToExclude = search.labelsToExclude;
    window.currentSILAPIs = search.silAPIs;
    window.currentFilterMode = search.filterMode || "any";
    window.currentDataSourceMode = search.dataSourceMode || "any";
    window.currentLabelsToExcludeMode = search.labelsToExcludeMode || "any";

    window.currentClassifierSearch = {
        run_id: search.classifierSearch.run_id || null,
        expression: search.classifierSearch.expression || null,
    };
    window.currentClusterSearch = {
        run_id: search.clusterSearch.runId,
        cluster_ids: search.clusterSearch.clusterIds || [],
    };
    const newClipListHash = search.clipIdListHash || null;
    if (newClipListHash !== window.currentClipList.hash) {
        window.currentClipList = {hash: newClipListHash, count: 0};
        if (typeof hydrateClipListFromUrl === "function") {
            hydrateClipListFromUrl();
        }
    }
    setClusterDistanceSliderUI(
        search.clusterDistanceMin ?? 0,
        search.clusterDistanceMax ?? 100,
    );
}

function search() {
    resetRRFIfTooFewFilters();
    let filterContainer = document.getElementById("filters");
    let withTimes = document.getElementById("with-times").checked;
    let withoutTimes = document.getElementById("without-times").checked;
    let withoutAnn = document.getElementById("without-ann").checked;
    let leftHandDriving = document.getElementById("left-hand-driving").checked;
    let searchTerm = document.getElementById("search-term").value;
    if (window._rewriteOriginalQuery) {
        if (searchTerm !== (window._rewriteOriginalQuery['caption'] || null)) clearRewriteTags('caption');
        if (document.getElementById("caption-embed-search-text").value !== (window._rewriteOriginalQuery['caption-embed'] || null)) clearRewriteTags('caption-embed');
        if (document.getElementById("semantic-search-text").value !== (window._rewriteOriginalQuery['semantic'] || null)) clearRewriteTags('semantic');
    }
    var extraQueries = getSelectedRewrites('caption');
    let speedSearchTerm = document.getElementById("speed-search-term").value;
    let countryQuery = document.getElementById("search-country").value;
    let clipIDQuery = document.getElementById("search-clipid").value;
    let withEgoData = document.getElementById("with-ego-data").checked;
    let trajectoryPattern = document.getElementById("trajectory-pattern").value;
    let trajectoryShapeClipID = window.currentTrajectoryShapeClipID;
    let semanticSearchClipID = window.currentSemanticSearchClipID;
    let semanticSearchText = document.getElementById("semantic-search-text").value;
    var captionEmbedExtraQueries = getSelectedRewrites('caption-embed');
    var semanticExtraQueries = getSelectedRewrites('semantic');
    var visualExtraQueries = getSelectedRewrites('visual');
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
    let reduction = document.getElementById("reduction-method").value;
    reduction = (reduction != "") ? reduction : null;

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

    let withSameClips = document.getElementById("with-same-clips").checked;

    let filterOptions = document.getElementById("annotation-tag");
    let filterValue = Array.from(filterOptions.selectedOptions).map(option => option.value);
    filterValue = filterValue.join("||")

    let filterMode = window.currentFilterMode || "any";

    let dataSource = document.getElementById("data-source");
    let selectedDataSource = Array.from(dataSource.selectedOptions).map(option => option.value);
    let dataSourceMode = window.currentDataSourceMode || "any";
    selectedDataSource = selectedDataSource.join("||")

    let labelsToExcludeOptions = document.getElementById("labels-to-exclude-choices");
    let labelsToExclude = Array.from(labelsToExcludeOptions.selectedOptions).map(option => option.value);
    let labelsToExcludeMode = window.currentLabelsToExcludeMode || "any";
    labelsToExclude = labelsToExclude.join("||")

    let projectSource = document.getElementById("project-select");
    let selectedProjectSource = Array.from(projectSource.selectedOptions).map(option => option.value);
    selectedProjectSource = selectedProjectSource.join("||")

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

    showPage(0, {
        filter: (filterValue != "") ? filterValue : null,
        filter_mode: modePayloadValue(filterValue, filterMode),
        numeric_filter: (metricValue != "") ? metricValue : null,
        times: (!withTimes && !withoutTimes) ? null : withTimes,
        without_ann: (withoutAnn != "") ? withoutAnn : null,
        left_hand_driving: leftHandDriving ? true : null,
        search: (searchTerm != "") ? searchTerm : null,
        caption_extra_queries: (extraQueries.length && searchTerm) ? extraQueries.join("||") : null,
        caption_embed_extra_queries: (captionEmbedExtraQueries.length && captionEmbedSearchText) ? captionEmbedExtraQueries.join("||") : null,
        semantic_extra_queries: (semanticExtraQueries.length && semanticSearchText) ? semanticExtraQueries.join("||") : null,
        visual_extra_queries: (visualExtraQueries.length && visualSearchText) ? visualExtraQueries.join("||") : null,
        search_speed: (speedSearchTerm != "") ? speedSearchTerm : null,
        search_country: (countryQuery != "") ? countryQuery : null,
        search_clipid: (clipIDQuery != "") ? clipIDQuery : null,
        with_ego_data: withEgoData ? true : null,
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
        sil_apis: null,
        ...clusterFilterPayload(),
        model_name: null,
        reduction: (reduction != "") ? reduction : null,
        with_same_clips: withSameClips ? true : null,
        rank_mode: document.getElementById("rrf-toggle-checkbox")?.checked ? "rrf" : null,
    });
}

function renderFiltersForTab(currentSearch) {
    let options = currentSearch.options;
    addAnnotationOptions(options, currentSearch.filter);
    addClassifierSearchOptions(options);

    let metricNameOptions = currentSearch.metricNames;
    addMetricOptions(metricNameOptions, currentSearch.numericFilter);

    let dataSourceOptions = currentSearch.dataSourceOptions;
    addDataSourceOptions(dataSourceOptions, currentSearch.dataSource);
    addLabelsToExclude(options, currentSearch.labelsToExclude);

    let projectSourceOptions = currentSearch.projectOptions;
    addProjectSourceOptions(projectSourceOptions, currentSearch.projectSource);

    let labelTypeOptions = currentSearch.labelTypeOptions;
    addLabelTypeOptions(labelTypeOptions, currentSearch.labelTypes);

    document.getElementById("with-times").checked = currentSearch.times === true;
    document.getElementById("without-times").checked = currentSearch.times === false;
    document.getElementById("without-ann").checked = currentSearch.withoutAnn === true;
    document.getElementById("left-hand-driving").checked = currentSearch.leftHandDriving === true;
    document.getElementById("with-ego-data").checked = currentSearch.withEgoData === true;
    window.currentFilterMode = currentSearch.filterMode || "any";
    syncModeControl("filter-mode-row", "#annotation-tag", "currentFilterMode");
    window.currentDataSourceMode = currentSearch.dataSourceMode || "any";
    syncModeControl("data-source-mode-row", "#data-source", "currentDataSourceMode");
    window.currentLabelsToExcludeMode = currentSearch.labelsToExcludeMode || "any";
    syncModeControl("labels-to-exclude-mode-row", "#labels-to-exclude-choices", "currentLabelsToExcludeMode");
    document.getElementById("search-term").value = currentSearch.search;
    window._rewriteOriginalQuery = window._rewriteOriginalQuery || {};
    var rewriteRestores = [
        ['caption', currentSearch.search, currentSearch.extraQueries],
        ['caption-embed', currentSearch.captionEmbedSearchText, currentSearch.captionEmbedExtraQueries],
        ['semantic', currentSearch.semanticSearchText, currentSearch.semanticExtraQueries],
        ['visual', currentSearch.visualSearchText, currentSearch.visualExtraQueries],
    ];
    rewriteRestores.forEach(function([type, query, queries]) {
        if (queries && queries.length) {
            window._rewriteOriginalQuery[type] = query;
            renderRewriteTags(queries, type);
        } else {
            clearRewriteTags(type);
        }
    });
    document.getElementById("speed-search-term").value = currentSearch.speedQuery;
    document.getElementById("search-country").value = currentSearch.countryQuery;
    document.getElementById("search-clipid").value = currentSearch.clipIDQuery;
    document.getElementById("trajectory-pattern").value = currentSearch.trajectoryPattern || "";
    document.getElementById("semantic-search-text").value = currentSearch.semanticSearchText;
    document.getElementById("search-comment-term").value = currentSearch.searchTermInComments || "";
    document.getElementById("visual-search-text").value = currentSearch.visualSearchText;
    document.getElementById("caption-embed-search-text").value = currentSearch.captionEmbedSearchText || "";

    document.getElementById("reduction-method").value = currentSearch.reduction || "";
    document.getElementById("with-same-clips").checked = currentSearch.sameClips === true;

    showContextualFilters();

    // Restore clustering panel state if a cluster search is active.
    // loadClusteringResults is idempotent: a no-op if the panel is already
    // populated for this run.
    const cs = window.currentClusterSearch;
    if (cs.run_id) {
        loadClusteringResults(cs.run_id, cs.cluster_ids);
    }
}

function renderLeaderboardFilters(leaderboard, selectedModels) {
    const lbSel = document.getElementById("leaderboard-select");
    const modelSel = document.getElementById("models-select");

    // Destroy the previous leaderboard selection elements
    lbSel.innerHTML = "";
    if ($(modelSel).data("select2")) {
        $(modelSel).select2("destroy");
    }
    modelSel.innerHTML = "";

    // Build the leaderboard selection first
    addOptionsToSelect(
        lbSel,
        Object.keys(window.modelsByLeaderboard).sort(),
        leaderboard
    );

    // Make a function to build the model selection based on a list of possible
    // models
    let buildModelSection = function (modelList) {
        if ($(modelSel).data("select2")) {
            $(modelSel).select2("destroy");
        }
        modelSel.innerHTML = "";
        $(modelSel).select2({
            placeholder: "Select models...",
            allowClear: false, // must always have at least one model
            multiple: true,
            width: "100%"
        });
        addOptionsToSelect(modelSel, modelList, null);
        $(modelSel).val(modelList).trigger("change.select2");
        $(modelSel).off("change.enforceOne").on("change.enforceOne", function() {
            const vals = $(this).val() || [];
            if (vals.length === 0) {
                $(this).val([modelList[0]]).trigger("change.select2");
            }
        });
    };

    // When a new leaderboard is chosen rebuild the model selection. If we move
    // back to the current leaderboard the same models as before
    lbSel.onchange = function () {
        buildModelSection(window.modelsByLeaderboard[lbSel.value]);
        if (lbSel.value == leaderboard && selectedModels !== null) {
            $(modelSel).val(selectedModels).trigger("change.select2");
        }
    };

    // Build the current model selection
    lbSel.onchange();
}

function makeInnerTableHTML(modelName) {
    return (
       '<div class="child-row" style="padding-left: 50px;">' +
           '<p><strong>Clips for ' + modelName + ':</strong></p>' +
           '<table class="display compact" style="width:100%;">' +
               '<thead></thead>' +
               '<tbody></tbody>' +
           '</table>' +
       '</div>'
    );
}

function getLink(clip_id) {
    let link = "/policy_predictions#&clip_id=" + clip_id;
    return `<a href="${link}" target="_blank" rel="noopener noreferrer">${clip_id}</a>`;
}


function createTab(tabInfo) {
    let tabId = tabInfo.id;
    let label = tabInfo.label;
    let path = tabInfo.path;

    const tabList = document.getElementById("search-tabs");
    const tab = document.createElement("li");
    tab.className = "tab-entry";
    tab.dataset.tabId = tabId;

    // Create inner span elements
    const labelSpan = document.createElement("a");
    labelSpan.className = "tab-label";
    labelSpan.textContent = label;
    // Show the full search path on hover for discoverability
    if (path) {
        labelSpan.title = path;
        labelSpan.setAttribute('aria-label', path);
    }

    const closeBtn = document.createElement("span");
    closeBtn.className = "close-btn";
    closeBtn.textContent = "✕";
    closeBtn.onclick = () => removeTab(tabInfo);

    tab.appendChild(labelSpan);
    tab.appendChild(closeBtn);
    tab.onclick = (e) => {
        if (e.target.classList.contains("close-btn")) {
            // Don't activate tab if the close button was clicked
            return;
        }
        activateTab(tabInfo);
    };

    // Insert new tab before the persistent "+" add-tab item if present
    const addItem = document.getElementById('add-tab-item');
    if (addItem) {
        tabList.insertBefore(tab, addItem);
    } else {
        tabList.appendChild(tab);
    }

    const content = document.createElement("div");
    content.id = tabId;
    content.className = "tab-content";
    document.getElementById("tab-content-container").appendChild(content);
}

function activateTab(tabInfo) {
    let tabId = tabInfo.id;
    document.querySelectorAll(".tab-content").forEach(div => div.style.display = "none");
    document.querySelectorAll(".tab-entry").forEach(tab => tab.classList.remove("active"));
    document.getElementById(tabId).style.display = "block";
    document.querySelector(`[data-tab-id="${tabId}"]`)?.classList.add("active");
    window.activeTab = tabKey(tabInfo);
    setGlobalSearchFlags(tabInfo.search);
    // rank_mode lives in the tab's path query string (not the server
    // search echo); parse it so the RRF toggle reflects tab state.
    const qs = (tabInfo.path || "").split("?")[1] || "";
    const rm = new URLSearchParams(qs).get("rank_mode");
    window.currentRankMode = rm === "rrf" ? "rrf" : "priority";
    renderFiltersForTab(tabInfo.search);
    renderLeaderboardFilters(tabInfo.leaderboard, tabInfo.selectedModels);
    renderActiveLeaderboardPill(tabInfo.leaderboard);
    updateHashFromTabs(tabInfo);
}

function removeTab(tabInfo) {
    const key = tabKey(tabInfo);
    // Check if the tab to be removed was active
    const wasActive = (window.activeTab === key);

    // Determine candidate next tab before removal
    // Get all the tab keys before deletion
    const keys = Object.keys(window.currentTabs);
    // Get the index of the tab to be closed.
    const currentIndex = keys.indexOf(key);
    // Picks the next tab if available, otherwise the previous tab.
    const plannedNeighborKey = currentIndex >= 0
        ? (currentIndex + 1 < keys.length ? keys[currentIndex + 1] : keys[currentIndex - 1])
        : null;

    // Remove tab UI
    const tabId = tabInfo.id;
    document.querySelector(`[data-tab-id="${tabId}"]`)?.remove();
    document.getElementById(tabId)?.remove();

    // Update state
    delete window.currentTabs[key];

    // If no tabs remain, create/activate a default /metrics? tab
    if (Object.keys(window.currentTabs).length === 0) {
        addNewTab();
        return;
    }

    // If we closed the active tab, activate neighbor; else just refresh hash
    if (wasActive) {
        const neighborKey = plannedNeighborKey && window.currentTabs[plannedNeighborKey]
          ? plannedNeighborKey
          : Object.keys(window.currentTabs)[0];
        activateTab(window.currentTabs[neighborKey]);
    } else {
        if (window.activeTab && window.currentTabs[window.activeTab]) {
          updateHashFromTabs(window.currentTabs[window.activeTab]);
        }
    }
}


function renderTable(currentPageDiv, tab) {
    // Filter models by current selection (leaderboard/models)
    let modelNames = Object.keys(tab.metrics);
    if (tab.selectedModels !== null) {
        modelNames = modelNames.filter(m => tab.selectedModels.includes(m));
    } else {
        const allowed = window.modelsByLeaderboard?.[tab.leaderboard];
        modelNames = allowed != null
            ? modelNames.filter(m => allowed.includes(m))
            : modelNames;
    }

    let metricNames = new Set();
    modelNames.forEach(modelName => {
        Object.keys(tab.metrics[modelName]).forEach(metricName => {
            metricNames.add(metricName);
        });
    });
    metricNames = Array.from(metricNames);
    // Reorder s.t. the name is always the first to appear.
    metricNames = ["num_clips", ...metricNames.filter(m => m !== "num_clips")];
    // Hide metadata-like columns in the outer table; pred reasoning shown in inner table only
    const hidenMetricNames = new Set([
        "gt_category",
        "pred_category",
        "question",
        "gt_timestamp",
        "pred_timestamp",
        "benchmark_start_time",
        "benchmark_end_time",
        "pred_reasoning",
        "pred reasoning",
        "pred-reasoning",
    ]);
    const visibleMetricNames = metricNames.filter(m => !hidenMetricNames.has(m))

    const mainTableColumns = [
        {
            className: "dt-control",
            orderable: false,
            data: null,
            defaultContent: ''
        },
        {
            title: "Model",
            data: "model",
            className: "model-col",
            render: function (data, type, row) {
                // Show full model name on hover; clipped via CSS on small screens
                const text = data == null ? '' : String(data);
                return `<span class="model-text" title="${text}">${text}</span>`;
            }
        }
    ];

    // Helper to safely escape HTML in cell renderers
    function escapeHtml(s) {
        if (s == null) return '';
        return String(s).replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;','\'':'&#39;'}[c]));
    }

    // Escape HTML and convert newlines to <br> for display in table cells
    function renderStringCell(s) {
        if (s == null) return '';
        return escapeHtml(String(s)).replace(/\n/g, '<br>');
    }

    visibleMetricNames.forEach(metricName => {
        mainTableColumns.push({
            title: metricName.replace(/_/g, ' '),
            data: metricName,
            render: function(data, type) {
                if (type !== 'display' || typeof data !== 'string') return data;
                return renderStringCell(data);
            }
        });
    });

    const mainTableData = [];
    for (const modelName of modelNames) {
        const rowData = {
             model: modelName
        };

        metricNames.forEach(metricName => {
            let value = tab.metrics[modelName][metricName];
            if (typeof value === "number" && Number.isFinite(value)) {
                // Only apply toFixed for non-integers; keep integers unformatted
                if (metricName === "num_clips" || Number.isInteger(value)) {
                    rowData[metricName] = Math.round(value);
                } else {
                    rowData[metricName] = Number(value).toFixed(3);
                }
            } else {
                rowData[metricName] = value;
            }
        });
        mainTableData.push(rowData);
    }

    // Create a unique table ID for this div
    const tableId = `leaderboard-table-${currentPageDiv.id}`;
    currentPageDiv.innerHTML = "";

    const tableElement = document.createElement("table");
    tableElement.id = tableId;
    tableElement.className = "display compact";
    tableElement.style.width = "100%";
    currentPageDiv.appendChild(tableElement);

    const dt = $(`#${tableId}`).DataTable({
        data: mainTableData,
        columns: mainTableColumns,
        paging: true,
        ordering: true,
        info: true,
        searching: true,
        scrollX: true,
    });
    // Adjust columns on window resize to keep header/body aligned with scrollX
    $(window).off(`resize.${tableId}`).on(`resize.${tableId}`, () => {
        dt.columns.adjust();
    });
    console.log(tab.path);

    $(`#${tableId} tbody`).off("click", "td.dt-control").on("click", "td.dt-control", function () {
        const tr = $(this).closest("tr");
        const row = dt.row(tr);
        const rowData = row.data(); 

        if (row.child.isShown()) {
            row.child.hide();
            tr.removeClass("shown");
        } else {
            row.child(makeInnerTableHTML(rowData.model)).show();
            tr.addClass("shown");

            const childTableElement = row.child().find("table");
            let modelName = rowData.model
            let path = buildCurrentEndpoint("/per_clip_metrics", modelName);
            if (tab.path !== null) {
                path = tab.path.replace("/metrics", "/per_clip_metrics");
                path += (modelName !== null) ? "&model_name=" + encodeURIComponent(modelName) : "";
            }
            fetch(path).then((r) => r.json()).then(function (data) {
                const excludeMetrics = ['question'];
                let subTableColumns = [];
                subTableColumns.push({title: "Clip ID", data: "clip_id"});
                data.metrics.forEach((m) => {
                    if (!excludeMetrics.includes(m)) {
                        const norm = String(m).toLowerCase().replace(/[\s_-]+/g, '');
                        if (norm === 'predreasoning') {
                            subTableColumns.push({
                                title: m.replace(/_/g, ' '),
                                data: m,
                                className: 'pred-reasoning-col',
                                render: function (data, type) {
                                    if (type !== 'display') return data;
                                    const text = data == null ? '' : String(data);
                                    const truncated = text.length > 120 ? (text.slice(0, 120) + '…') : text;
                                    const fullEsc = escapeHtml(text);
                                    const truncEsc = renderStringCell(truncated);
                                    return `<span class=\"reasoning-cell\" data-fulltext=\"${fullEsc}\" title=\"Click to view full reasoning\">${truncEsc}</span>`;
                                }
                            });
                        } else {
                            subTableColumns.push({
                                title: m.replace(/_/g, ' '),
                                data: m,
                                render: function(data, type) {
                                    if (type !== 'display' || typeof data !== 'string') return data;
                                    return renderStringCell(data);
                                }
                            });
                        }
                    }

                });
                let subTableData = [];
                zip2(data.clips, data.values).forEach(([c, vs]) => {
                    let row = {clip_id: getLink(c)};
                    zip2(data.metrics, vs).forEach(([m, v]) => {
                        if (typeof v === "number" && Number.isFinite(v)) {
                            if (Number.isInteger(v)) {
                                row[m] = Math.round(v);
                            } else {
                                row[m] = Number(v).toFixed(4);
                            }
                        } else {
                            row[m] = v;
                        }
                    });
                    subTableData.push(row);
                });
                const innerDT = childTableElement.DataTable({
                  data: subTableData,
                  columns: subTableColumns,
                  paging: true,
                  pageLength: 10,
                  ordering: true,
                  info: false,
                  searching: false,
                  dom: "tip",
                  scrollX: true,
                  autoWidth: false,
                });
                // Click to open full pred reasoning in the inner table only
                childTableElement.off('click', '.reasoning-cell').on('click', '.reasoning-cell', function () {
                    const full = this.getAttribute('data-fulltext') || '';
                    try { if (typeof closeCaptionBoxes === 'function') closeCaptionBoxes(); } catch (e) {}
                    let box = document.querySelector('.caption-box.lb-reasoning-box');
                    if (!box) {
                        box = document.createElement('div');
                        box.className = 'caption-box lb-reasoning-box';
                        const closeBtn = document.createElement('span');
                        closeBtn.className = 'clear-annotation-button';
                        closeBtn.title = 'Close';
                        closeBtn.textContent = '×';
                        closeBtn.onclick = function () {hideCaption()};
                        const content = document.createElement('div');
                        content.className = 'caption-text';
                        box.appendChild(closeBtn);
                        box.appendChild(content);
                        document.body.appendChild(box);
                    }
                    const content = box.querySelector('.caption-text');
                    content.innerHTML = renderStringCell(full);
                    box.style.display = 'block';
                });
            });
        }
    });

}

function render() {
    // Clear all tabs
    let tabContainer = document.getElementById("tab-content-container");
    tabContainer.innerHTML = "";
    document.querySelectorAll("[data-tab-id]").forEach((el) => el.remove());

    // Make all tabs
    Object.keys(window.currentTabs).forEach(key => {
        createTab(window.currentTabs[key]);
        const tableDiv = document.getElementById(window.currentTabs[key].id);
        renderTable(tableDiv, window.currentTabs[key]);
    });

    // Actually show the activeTab
    let defaultKey = Object.keys(window.currentTabs)[0];
    let key = (window.activeTab !== null) ? window.activeTab : defaultKey;
    activateTab(window.currentTabs[key]);
}

function renderActiveLeaderboardPill(leaderboard) {
    const container = document.getElementById('active-leaderboard-pill');
    container.querySelector("button").textContent = leaderboard;
}

function fetchTabData(tabInfo) {
    const key = tabKey(tabInfo);

    if (!(key in window.currentTabs)) {
        return Promise.reject(new Error(`Tab ${key} not found`));
    }

    if (window.currentTabs[key].metrics !== null) {
        return Promise.resolve(true);
    }

    return fetch(tabInfo.path).then(function (response) {
        return response.json();
    }).then(function (data) {
        // Set the search parameters
        window.currentTabs[key].search = {
            options: data.options,
            filter: data.filter,
            times: data.times,
            leftHandDriving: data.left_hand_driving,
            withoutAnn: data.without_ann,
            search: data.search,
            speedQuery: data.search_speed,
            countryQuery: data.search_country,
            clipIDQuery: data.search_clipid,
            withEgoData: data.with_ego_data,
            trajectoryPattern: data.trajectory_pattern,
            trajectoryShapeClipID: data.trajectory_shape_clipid,
            trajectoryShapeStartT: data.trajectory_shape_start_t,
            trajectoryShapeEndT: data.trajectory_shape_end_t,
            semanticSearchClipID: data.semantic_search_clipid,
            semanticSearchText: data.semantic_search_text,
            visualSearchText: data.visual_search_text,
            visualSearchImageId: data.visual_search_image_id || null,
            captionEmbedSearchText: data.caption_embed_search,
            classifierSearch: {
                run_id: data.classifier_run_id || null,
                expression: data.probability_expression || null,
            },
            clusterSearch: {
                runId: data.cluster_run_id || null,
                clusterIds: clusterIdsFromData(data.cluster_ids),
            },
            clipIdListHash: data.clip_id_list_hash || null,
            clusterDistanceMin: data.cluster_distance_min ?? null,
            clusterDistanceMax: data.cluster_distance_max ?? null,
            labelTypeOptions: data.label_type_options,
            labelTypes: data.label_types,
            searchTermInComments: data.search_comments,
            wmClassName: data.wm_class_name,
            wmMinCount: data.wm_min_count,
            wmMaxCount: data.wm_max_count,
            wmMaxDist: data.wm_max_dist,
            wmMinTime: data.wm_min_time,
            wmAngleRange: data.wm_angle_range,
            metrics: data.metrics,
            dataSourceOptions: data.data_source_options,
            datasetMetadata: data.dataset_metadata,
            dataSource: data.data_source,
            projectOptions: data.project_options,
            projectSource: data.project_source,
            labelsToExclude: data.labels_to_exclude,
            reduction: data.reduction,
            sameClips: data.with_same_clips,
            silAPIs: data.sil_apis,
            metricNames: data.metric_names,
            numericFilter: data.numeric_filter,
            extraQueries: data.caption_extra_queries || [],
            captionEmbedExtraQueries: data.caption_embed_extra_queries || [],
            semanticExtraQueries: data.semantic_extra_queries || [],
            visualExtraQueries: data.visual_extra_queries || [],
            filterMode: data.filter_mode || "any",
            dataSourceMode: data.data_source_mode || "any",
            labelsToExcludeMode: data.labels_to_exclude_mode || "any",
        };

        // Set the actual results/metrics to show
        window.currentTabs[key].metrics = data.metrics;

        // Set the models and available leaderboard options
        window.modelsByLeaderboard = data.models_by_leaderboard;

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

        return true;
    });
}

function showPage(pageNum, filters) {
    let path = buildEndpoint("/metrics", { page: null, ...filters });

    // Apply new search to the active tab instead of opening a new one. We
    // achieve this by keeping the same `id` as the old tab
    const oldKey = window.activeTab || Object.keys(window.currentTabs || {})[0];
    const tabInfo = window.currentTabs[oldKey];
    delete window.currentTabs[oldKey];

    // Set the new search
    tabInfo.path = path;
    tabInfo.leaderboard = document.getElementById("leaderboard-select").value;
    tabInfo.selectedModels = getSelectedModels(tabInfo.leaderboard);
    tabInfo.label = tabLabel(tabInfo);

    // Remove the key and metrics to enable recomputation of the key and
    // refetching of the metrics from the server
    tabInfo.key = null;
    tabInfo.metrics = null;

    // Add the tab back to the tab map and set it as current
    window.currentTabs[tabKey(tabInfo)] = tabInfo;
    window.activeTab = tabKey(tabInfo);

    // Fetch the new data
    showLoading("Loading...");
    fetchTabData(tabInfo).then(() => {
        render();
        hideLoading();
    });
}

function updateHashFromTabs(activeTab) {
    const encodedPaths = Object.values(window.currentTabs).map(tabToURLString);
    let hash = "#tabs=" + encodeURIComponent(encodedPaths.join("||"));
    hash += "&active=" + encodeURIComponent(tabKey(activeTab));
    history.replaceState(null, "", hash);
}

window.onhashchange = function () {
    const hash = window.location.hash.slice(1);
    const params = getQueryParams(hash);

    if (params.tabs) {
        params.tabs.split("||").forEach((tabString, i) => {
            const tab = tabFromURLString(tabString);
            if (!(tabKey(tab) in window.currentTabs)) {
                window.currentTabs[tabKey(tab)] = tab;
            }
        });
    }

    // If no tabs were provided/restored, create a single default /metrics? tab
    if (Object.keys(window.currentTabs).length === 0) {
        const tab = makeDefaultTab();
        window.currentTabs[tabKey(tab)] = tab;
        window.activeTab = tabKey(tab);
    }

    if (params.active) {
        window.activeTab = params.active;
    }

    showLoading("Loading...");
    let dataPromises = Object.values(window.currentTabs).map(fetchTabData);
    Promise.all(dataPromises).then(() => {
        render();
        hideLoading();
    });
};

function toggleEvaluationPanel(el) {
    const adv = document.getElementById("evaluation-container");
    let activate = adv.style.display === "none";
    adv.style.display = (activate) ? "block" : "none";
    el.classList.toggle("selected", activate);
}

function onReductionChange() {
    search();
}

document.addEventListener('DOMContentLoaded', function() {
    initClassifierPanel({canTrain: false});
    bindModeControl("filter-mode-row", "currentFilterMode");
    bindModeControl("data-source-mode-row", "currentDataSourceMode");
    bindModeControl("labels-to-exclude-mode-row", "currentLabelsToExcludeMode");
    renderSectors();
    updateSelectedAngles();

    // Make angle selector functions globally accessible for reset
    window.wmAngleSelector = {
        renderSectors: renderSectors,
        updateSelectedAngles: updateSelectedAngles
    };

    // Wire up Add Tab button
    const addBtn = document.getElementById('add-tab-btn');
    addBtn.addEventListener('click', addNewTab);
});

function hideCaption() {
    let box = document.querySelector('.caption-box.lb-reasoning-box');
    if (box) box.style.display = "none";
}

// Event listener for Escape key
document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
        hideSpeedSearchHelp();
        hideCaption();
        hideCaptionSearchHelp();
        hideSemanticSearchHelp();
        hideVisualSearchHelp();
        hideVisualSearchImageHelp();
        hideCaptionEmbedSearchHelp();
        hideVideoToVideoSearchHelp();
        hideTrajectoryShapeSearchHelp();
        hideCommentSearchHelp();
        hideClusteringHelp();
        hideClosestClustersHelp();
    } else if (event.key == "Enter" && event.target.closest(".sidebar")) {
        search();
    }
});
