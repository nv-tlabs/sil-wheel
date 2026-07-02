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

/* Arena page logic */

window.arenas = [];
window.currentArena = null;    // arena name
window.currentManifest = null;
window.currentMatch = null;
window.matchCount = 0;
window.eloTable = null;
window.historyTable = null;
window.eloChart = null;
window.votingActive = false;   // true when vote buttons are enabled
window.currentCriteria = [];       // [{name, description}, ...] from match
window.currentCriterionIndex = 0;  // which criterion is being voted on
window.criterionVotes = [];        // [{criterion, rating_a, rating_b, rating_change_*, ...}, ...] collected results

// Per-vote timer. Sequential anchor: lastEventAt resets to "now" when assets are ready
// and after every submitted vote/skip. Each criterion's duration_ms is (now - lastEventAt)
// minus time the tab was hidden. Durations longer than HARD_CAP are sent as null.
window.matchTimer = { lastEventAt: 0, accumulatedHidden: 0, hiddenSince: 0 };
var VOTE_DURATION_HARD_CAP_MS = 10 * 60 * 1000;  // 10 minutes
window.currentCriterionFilter = null; // leaderboard filter: null = aggregate, or criterion name
window.currentLabelFilter = null;    // leaderboard filter: null = all items, or label string
window.arenaLoadGen = 0;             // generation counter — discard stale async responses

// Item labels — parsed from manifest.item_labels (CSV per item)
window.itemLabelMap = {};            // item_id → Set of label strings
window.allLabels = [];               // sorted unique labels across all items

// Analytics state
window.analyticsVotes = null;          // raw votes from /arena/votes
window.analyticsNormalized = null;     // canonicalized votes (with direction, canonical pair/winner)
window.analyticsVotesArena = null;     // which arena the cache is for
window.analyticsVotesFetchedAt = 0;    // timestamp of last fetch (ms)
window.analyticsFilters = { excludedUsers: new Set(), excludedItems: new Set(), criterion: null, labels: null };
window.analyticsEloTable = null;       // DataTable instance for filtered leaderboard
window.analyticsAnnotatorTable = null; // DataTable instance for annotator table
window.analyticsItemTable = null;      // DataTable instance for item table
window.analyticsChart = null;          // Chart.js instance for filtered rating chart

// ── Init ──

document.addEventListener("DOMContentLoaded", function () {
    // Logout
    var logoutBtn = document.getElementById("logout-btn");
    if (logoutBtn) {
        logoutBtn.addEventListener("click", function () {
            fetch("/", { method: "POST", body: "logout::" }).catch(function () {});
            window.location.replace("/login");
        });
    }

    // Keyboard shortcuts — mapped dynamically based on current criterion mode
    document.addEventListener("keydown", function (e) {
        if (!window.votingActive || !window.currentMatch) return;
        if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

        var keyMap = getVoteKeyMap();
        if (keyMap[e.key]) { submitVote(keyMap[e.key]); return; }
        if (e.key === "0") { submitVote("both_bad"); return; }
        if (e.key === "s" || e.key === "S") { skipCriterion(); return; }
    });

    // "N" or Enter for next after reveal (works in both voting and review modes)
    document.addEventListener("keydown", function (e) {
        if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
        if (e.key === "Escape") { hideArenaHelp(); return; }
        if (e.key !== "n" && e.key !== "N" && e.key !== "Enter") return;
        var nextBtn = document.getElementById("arena-next-btn");
        if (nextBtn && nextBtn.style.display !== "none" && document.getElementById("arena-reveal").style.display !== "none") {
            nextBtn.click();
        }
    });

    loadArenaList();
});

// ── Per-vote timer ──
// Anchor reset on match ready + after each submitted vote/skip.
// Page Visibility API used to subtract time the user was on another tab.
function resetMatchTimer() {
    window.matchTimer.lastEventAt = performance.now();
    window.matchTimer.accumulatedHidden = 0;
    window.matchTimer.hiddenSince = 0;
}
function consumeVoteDurationMs() {
    var t = window.matchTimer;
    if (!t.lastEventAt) return null;  // timer not yet started
    var now = performance.now();
    // If currently hidden, fold the pending hidden interval before reading.
    if (t.hiddenSince) {
        t.accumulatedHidden += now - t.hiddenSince;
        t.hiddenSince = now;  // keep the clock ticking from "now" for any future hidden time
    }
    var elapsed = Math.round((now - t.lastEventAt) - t.accumulatedHidden);
    // Reset anchor for the next vote
    t.lastEventAt = now;
    t.accumulatedHidden = 0;
    if (elapsed < 0 || elapsed > VOTE_DURATION_HARD_CAP_MS) return null;
    return elapsed;
}
document.addEventListener("visibilitychange", function () {
    var t = window.matchTimer;
    if (!t.lastEventAt) return;
    if (document.hidden) {
        t.hiddenSince = performance.now();
    } else if (t.hiddenSince) {
        t.accumulatedHidden += performance.now() - t.hiddenSince;
        t.hiddenSince = 0;
    }
});

// ── Arena list ──

window.arenaFolderState = {};  // folder path → true (expanded) / false (collapsed)

function loadArenaList() {
    fetch("/arena/list")
        .then(function (r) { return r.json(); })
        .then(function (data) {
            window.arenas = data.arenas || [];
            renderArenaList();
            applyHash();
        })
        .catch(function (e) { console.error("Failed to load arenas:", e); });
}

function buildArenaTree(arenas) {
    // Build a nested tree from flat arena list.  Names with "/" denote folder paths.
    // Returns { folders: { name: subtree, ... }, arenas: [arena, ...] }
    var root = { folders: {}, arenas: [] };

    arenas.forEach(function (a) {
        var parts = a.name.split("/");
        var node = root;
        for (var i = 0; i < parts.length - 1; i++) {
            if (!node.folders[parts[i]]) {
                node.folders[parts[i]] = { folders: {}, arenas: [] };
            }
            node = node.folders[parts[i]];
        }
        node.arenas.push(a);
    });
    return root;
}

function isArenaInSubtree(node, arenaName) {
    // Check if a given arena lives anywhere in this subtree
    for (var i = 0; i < node.arenas.length; i++) {
        if (node.arenas[i].name === arenaName) return true;
    }
    var folderNames = Object.keys(node.folders);
    for (var j = 0; j < folderNames.length; j++) {
        if (isArenaInSubtree(node.folders[folderNames[j]], arenaName)) return true;
    }
    return false;
}

function renderArenaCard(a, depth) {
    var card = document.createElement("div");
    card.className = "arena-card" + (window.currentArena === a.name ? " active" : "");
    if (depth > 0) card.style.marginLeft = (depth * 8) + "px";
    card.onclick = function () {
        window.location.hash = "name=" + encodeURIComponent(a.name);
    };
    var badge = (!a.published) ? '<span class="arena-draft-badge">Draft</span>' : '';
    var desc = depth > 0 ? '' : '<div class="arena-card-desc">' + escapeHtml(a.description) + '</div>';
    card.innerHTML =
        '<div class="arena-card-name">' + escapeHtml(a.display_name) + badge + '</div>' +
        desc +
        '<div class="arena-card-meta">' + a.num_models + ' models · ' + a.total_votes + ' votes</div>';
    return card;
}

function countArenasInSubtree(node) {
    var n = node.arenas.length;
    Object.keys(node.folders).forEach(function (k) { n += countArenasInSubtree(node.folders[k]); });
    return n;
}

function renderArenaTreeNode(node, container, depth, pathPrefix) {
    var folderNames = Object.keys(node.folders).sort();

    folderNames.forEach(function (name) {
        var folderPath = pathPrefix ? pathPrefix + "/" + name : name;
        var subtree = node.folders[name];

        // Auto-expand if active arena is inside this folder
        var hasActive = window.currentArena && isArenaInSubtree(subtree, window.currentArena);
        if (hasActive) window.arenaFolderState[folderPath] = true;

        var isOpen = window.arenaFolderState[folderPath] !== false;
        if (!(folderPath in window.arenaFolderState)) {
            // Default: expand top-level, collapse deeper
            isOpen = depth < 1;
            window.arenaFolderState[folderPath] = isOpen;
        }

        var folder = document.createElement("div");
        folder.className = "arena-folder";

        var header = document.createElement("div");
        header.className = "arena-folder-header" + (depth > 0 ? " nested" : "");
        if (depth > 0) header.style.marginLeft = (depth * 12) + "px";
        var count = countArenasInSubtree(subtree);
        header.innerHTML = '<span class="arena-folder-icon' + (isOpen ? ' open' : '') + '">&#9656;</span>' +
            '<span class="arena-folder-name">' + escapeHtml(name) + '</span>' +
            '<span class="arena-folder-count">' + count + '</span>';

        var content = document.createElement("div");
        content.className = "arena-folder-content";
        content.style.display = isOpen ? "" : "none";

        header.onclick = function () {
            var nowOpen = content.style.display === "none";
            content.style.display = nowOpen ? "" : "none";
            var icon = header.querySelector(".arena-folder-icon");
            if (nowOpen) { icon.classList.add("open"); } else { icon.classList.remove("open"); }
            window.arenaFolderState[folderPath] = nowOpen;
        };

        renderArenaTreeNode(subtree, content, depth + 1, folderPath);

        folder.appendChild(header);
        folder.appendChild(content);
        container.appendChild(folder);
    });

    // Render arenas at this level
    node.arenas.forEach(function (a) {
        container.appendChild(renderArenaCard(a, depth));
    });
}

function renderArenaList(filter) {
    var container = document.getElementById("arena-list");
    container.innerHTML = "";
    var term = (filter || "").toLowerCase();

    if (term) {
        // Search active → flat filtered list
        window.arenas.forEach(function (a) {
            if (a.display_name.toLowerCase().indexOf(term) === -1 && a.name.toLowerCase().indexOf(term) === -1) {
                return;
            }
            container.appendChild(renderArenaCard(a, 0));
        });
    } else {
        // No search → tree view
        var tree = buildArenaTree(window.arenas);
        renderArenaTreeNode(tree, container, 0, "");
    }
}

// Wire up search filter
document.addEventListener("DOMContentLoaded", function () {
    var input = document.getElementById("arena-search");
    if (input) {
        input.addEventListener("input", function () {
            renderArenaList(this.value);
        });
    }
});

// ── Arena selection ──

function selectArena(name, matchId) {
    window.currentArena = name;
    window.currentCriterionFilter = null;  // reset to aggregate
    window.currentLabelFilter = null;      // reset label filter
    var gen = ++window.arenaLoadGen;       // invalidate any in-flight fetches
    renderArenaList(document.getElementById("arena-search").value);

    document.getElementById("arena-empty").style.display = "none";
    document.getElementById("arena-leaderboard-view").style.display = "";
    document.getElementById("arena-annotation-view").style.display = "none";

    // Reset to leaderboard tab
    switchArenaTab("leaderboard");

    // Update navbar title
    document.getElementById("arena-title").textContent = "SIL-Wheel Arena";

    // Fetch manifest + leaderboard + history + confidence intervals in parallel
    showLoading("Loading arena...");
    var q = encodeURIComponent(name);
    Promise.all([
        fetch("/arena/manifest?name=" + q).then(function (r) { return r.json(); }),
        fetch("/arena/leaderboard?name=" + q).then(function (r) { return r.json(); }),
        fetch("/arena/history?name=" + q + "&limit=100").then(function (r) { return r.json(); }),
    ]).then(function (results) {
        if (gen !== window.arenaLoadGen) { hideLoading(); return; }
        window.currentManifest = results[0];
        renderLeaderboard(results[0], results[1], results[2]);
        hideLoading();
        if (matchId) reviewMatch(matchId);
    }).catch(function (e) { hideLoading(); console.error("Failed to load arena:", e); });
}

function switchArenaTab(tab) {
    document.getElementById("arena-tab-leaderboard").style.display = tab === "leaderboard" ? "" : "none";
    document.getElementById("arena-tab-analytics").style.display = tab === "analytics" ? "" : "none";
    document.getElementById("arena-tab-btn-leaderboard").classList.toggle("active", tab === "leaderboard");
    document.getElementById("arena-tab-btn-analytics").classList.toggle("active", tab === "analytics");
    if (tab === "analytics") loadAnalytics();
}

function renderLeaderboard(manifest, leaderboardData, historyData) {
    document.getElementById("arena-display-name").textContent = manifest.display_name || manifest.name;
    document.getElementById("arena-description").textContent = manifest.description || "";
    document.getElementById("arena-total-matches").textContent = leaderboardData.total_matches + " total votes";
    document.getElementById("arena-title").textContent = "SIL-Wheel Arena: " + (manifest.display_name || manifest.name);

    // Info button with owners
    var infoBtn = document.getElementById("arena-info-btn");
    var owners = manifest.owners || [];
    if (owners.length > 0) {
        infoBtn.style.display = "";
        document.getElementById("arena-info-tooltip").textContent = "Owners: " + owners.join(", ");
    } else {
        infoBtn.style.display = "none";
    }

    // Owner controls
    var ownerControls = document.getElementById("arena-owner-controls");
    var currentArenaInfo = window.arenas.find(function (a) { return a.name === window.currentArena; });
    if (currentArenaInfo && currentArenaInfo.is_owner) {
        ownerControls.style.display = "";
        var publishBtn = document.getElementById("arena-publish-btn");
        if (currentArenaInfo.published) {
            publishBtn.textContent = "Unpublish";
            publishBtn.onclick = function () { arenaUnpublish(window.currentArena); };
        } else {
            publishBtn.textContent = "Publish";
            publishBtn.onclick = function () { arenaPublish(window.currentArena); };
        }
        document.getElementById("arena-refresh-btn").onclick = function () { arenaRefreshManifest(window.currentArena); };
    } else {
        ownerControls.style.display = "none";
    }

    // VLM Judge controls — show if owner/admin AND server has VLM judge
    var vlmControls = document.getElementById("arena-vlm-judge-controls");
    if (currentArenaInfo && currentArenaInfo.is_owner && manifest.vlm_judge_available) {
        vlmControls.style.display = "";
        document.getElementById("arena-vlm-judge-btn").onclick = function () { arenaRunVLMJudge(window.currentArena); };
    } else {
        vlmControls.style.display = "none";
    }

    // Criteria chips — show if manifest has >1 criterion
    var criteriaSection = document.getElementById("arena-criteria-section");
    var chipsEl = document.getElementById("arena-criteria-chips");
    var criteria = manifest.criteria || [];
    if (criteria.length > 1) {
        criteriaSection.style.display = "";
        chipsEl.innerHTML = "";
        // Aggregate chip
        var aggChip = document.createElement("span");
        aggChip.className = "arena-criterion-chip" + (window.currentCriterionFilter === null ? " active" : "");
        aggChip.textContent = "Aggregate";
        aggChip.onclick = function () { loadCriterionLeaderboard(null); };
        chipsEl.appendChild(aggChip);
        // Per-criterion chips
        criteria.forEach(function (c) {
            var chip = document.createElement("span");
            chip.className = "arena-criterion-chip" + (window.currentCriterionFilter === c.name ? " active" : "");
            chip.textContent = c.name;
            chip.title = c.description || "";
            chip.onclick = function () { loadCriterionLeaderboard(c.name); };
            chipsEl.appendChild(chip);
        });
    } else {
        criteriaSection.style.display = "none";
    }

    // Label chips — show if manifest has item_labels
    parseItemLabels(manifest);
    var labelsSection = document.getElementById("arena-labels-section");
    var labelChipsEl = document.getElementById("arena-label-chips");
    if (window.allLabels.length > 0) {
        labelsSection.style.display = "";
        labelChipsEl.innerHTML = "";
        var allLabelChip = document.createElement("span");
        allLabelChip.className = "arena-criterion-chip" + (window.currentLabelFilter === null ? " active" : "");
        allLabelChip.textContent = "All";
        allLabelChip.onclick = function () { loadLabelLeaderboard(null); };
        labelChipsEl.appendChild(allLabelChip);
        window.allLabels.forEach(function (label) {
            var chip = document.createElement("span");
            chip.className = "arena-criterion-chip" + (window.currentLabelFilter === label ? " active" : "");
            chip.textContent = label;
            chip.onclick = function () { loadLabelLeaderboard(label); };
            labelChipsEl.appendChild(chip);
        });
    } else {
        labelsSection.style.display = "none";
    }

    renderEloTable(leaderboardData);

    // History table — compact, last 5 with pagination, clickable rows
    if (window.historyTable) {
        window.historyTable.destroy();
        document.querySelector("#arena-history-table tbody").innerHTML = "";
    }
    var canDelete = !!(currentArenaInfo && currentArenaInfo.is_owner);
    var hrows = (historyData || []).map(function (h) {
        var d = new Date(h.created_at * 1000);
        var ts = d.getFullYear() + "-" +
            String(d.getMonth() + 1).padStart(2, "0") + "-" +
            String(d.getDate()).padStart(2, "0") + " " +
            String(d.getHours()).padStart(2, "0") + ":" +
            String(d.getMinutes()).padStart(2, "0") + ":" +
            String(d.getSeconds()).padStart(2, "0");
        // Build winner summary from criterion_votes (grouped by match)
        var votes = h.criterion_votes || [];
        var winLabel = votes.map(function (cv) {
            return fmtWinner(cv.winner);
        }).join(" / ") || "—";
        // columns: 0=vlm_tag, 1=ts, 2=input_id, 3=model_a, 4=model_b, 5=winLabel, 6=username, 7=match_id, 8=delete_btn
        var isVLM = h.username && h.username.indexOf("vlm_judge:") === 0;
        var vlmTag = isVLM ? '<span style="font-size:0.7em;background:#6f42c1;color:#fff;padding:1px 4px;border-radius:2px;">VLM</span>' : '';
        var delBtn = canDelete ? '<button class="arena-history-delete-btn" title="Delete this vote (admin/owner only)">🗑</button>' : '';
        return [vlmTag, ts, h.input_id, h.model_a, h.model_b, winLabel, h.username, h.match_id, delBtn];
    });
    window.historyTable = $("#arena-history-table").DataTable({
        data: hrows,
        pageLength: 5,
        lengthMenu: [5, 10, 25, 50],
        paging: true,
        searching: false,
        info: true,
        order: [[1, "desc"]],
        language: { info: "Showing _START_-_END_ of _TOTAL_ matches" },
        columnDefs: [
            { targets: [7], visible: false },
            { targets: 0, width: "30px", orderable: false, className: "dt-center" },
            { targets: 8, width: "32px", orderable: false, className: "dt-center" },
        ],
        createdRow: function (row, data) {
            // Allow HTML in the VLM tag and delete-button columns
            $("td:eq(0)", row).html(data[0]);
            $("td:eq(7)", row).html(data[8]);  // last visible col index after match_id is hidden
        },
    });
    // Row click reviews the match — but ignore clicks on the delete button itself
    $("#arena-history-table tbody").off("click");
    $("#arena-history-table tbody").on("click", "tr", function (e) {
        if (e.target && e.target.closest && e.target.closest(".arena-history-delete-btn")) return;
        var data = window.historyTable.row(this).data();
        if (data) reviewMatch(data[7], data[6]);
    });
    $("#arena-history-table tbody").on("click", ".arena-history-delete-btn", function (e) {
        e.stopPropagation();
        var data = window.historyTable.row($(this).closest("tr")).data();
        if (!data) return;
        deleteMatch(data[7]);
    });

    // Rating convergence chart — use currentArena (path-based) not manifest.name (may be stale flat name)
    renderEloChart(window.currentArena, window.currentCriterionFilter);
}

function renderEloChart(arenaName, criterion) {
    var gen = window.arenaLoadGen;
    var cParam = criterion ? "&criterion=" + encodeURIComponent(criterion) : "";
    fetch("/arena/elo_history?name=" + encodeURIComponent(arenaName) + cParam)
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (gen !== window.arenaLoadGen) return;
            createEloLineChart("arena-elo-chart", "eloChart", data);
        })
        .catch(function (e) { console.error("Failed to load rating history:", e); });
}

function loadCriterionLeaderboard(criterion) {
    window.currentCriterionFilter = criterion;
    // If a label is active, use client-side replay (label filtering needs votes)
    if (window.currentLabelFilter) {
        reloadLeaderboardFromVotes();
        return;
    }
    // No label filter — use server-computed leaderboard
    var gen = ++window.arenaLoadGen;
    var q = encodeURIComponent(window.currentArena);
    var cParam = criterion ? "&criterion=" + encodeURIComponent(criterion) : "";
    fetch("/arena/leaderboard?name=" + q + cParam)
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (gen !== window.arenaLoadGen) return;
            updateLeaderboardChips();
            document.getElementById("arena-total-matches").textContent = data.total_matches + " total votes";
            renderEloTable(data);
            renderEloChart(window.currentArena, criterion);
        }).catch(function (e) { console.error("Failed to load criterion leaderboard:", e); });
}

function loadLabelLeaderboard(label) {
    window.currentLabelFilter = label;
    if (!label) {
        // Revert to server-computed leaderboard
        loadCriterionLeaderboard(window.currentCriterionFilter);
        return;
    }
    reloadLeaderboardFromVotes();
}

// Replay ratings client-side with current criterion + label filters.
// Used when label filtering is active (server doesn't support per-item filtering).
function reloadLeaderboardFromVotes() {
    var gen = ++window.arenaLoadGen;
    var arenaName = window.currentArena;
    var manifest = window.currentManifest || {};
    var cfg = readRatingConfig(manifest);

    // Reuse analytics vote cache if available, otherwise fetch
    var votesReady = (window.analyticsVotesArena === arenaName && window.analyticsVotes)
        ? Promise.resolve(window.analyticsVotes)
        : fetch("/arena/votes?name=" + encodeURIComponent(arenaName))
            .then(function (r) { return r.json(); })
            .then(function (data) {
                window.analyticsVotes = data.votes || [];
                window.analyticsVotesArena = arenaName;
                window.analyticsVotesFetchedAt = Date.now();
                return data.votes || [];
            });

    votesReady.then(function (votes) {
        if (gen !== window.arenaLoadGen) return;
        var normalized = normalizeVotes(votes);
        if (window.currentCriterionFilter) {
            normalized = normalized.filter(function (v) { return v.criterion === window.currentCriterionFilter; });
        }
        normalized = filterVotesByLabel(normalized, window.currentLabelFilter);
        var replayed = replayRatingsAggregate(normalized, cfg);
        updateLeaderboardChips();
        document.getElementById("arena-total-matches").textContent = normalized.length + " total votes";
        renderEloTable({ rankings: replayed.rankings, total_matches: normalized.length });
        createEloLineChart("arena-elo-chart", "eloChart", replayed.history);
    }).catch(function (e) { console.error("Failed to load votes:", e); });
}

// Update active state on both criterion and label chip rows
function updateLeaderboardChips() {
    document.querySelectorAll("#arena-criteria-chips .arena-criterion-chip").forEach(function (chip) {
        var isAgg = chip.textContent === "Aggregate" && window.currentCriterionFilter === null;
        var isMatch = chip.textContent === window.currentCriterionFilter;
        chip.classList.toggle("active", isAgg || isMatch);
    });
    document.querySelectorAll("#arena-label-chips .arena-criterion-chip").forEach(function (chip) {
        var isAll = chip.textContent === "All" && window.currentLabelFilter === null;
        var isMatch = chip.textContent === window.currentLabelFilter;
        chip.classList.toggle("active", isAll || isMatch);
    });
}

// ── Annotation flow ──

function showLeaderboardView() {
    document.getElementById("arena-leaderboard-view").style.display = "";
    document.getElementById("arena-annotation-view").style.display = "none";
    window.votingActive = false;
    window.reviewIndex = -1;
    window.reviewHistory = [];
    window.currentCriteria = [];
    window.currentCriterionIndex = 0;
    window.criterionVotes = [];
    // Refresh leaderboard
    selectArena(window.currentArena);
}

function exportVotes() {
    if (!window.currentArena) return;
    fetch("/arena/votes?name=" + encodeURIComponent(window.currentArena))
        .then(function (r) { return r.json(); })
        .then(function (data) {
            var blob = new Blob([JSON.stringify(data.votes, null, 2)], { type: "application/json" });
            var a = document.createElement("a");
            a.href = URL.createObjectURL(blob);
            a.download = window.currentArena + "_votes.json";
            a.click();
            URL.revokeObjectURL(a.href);
        });
}

function startEvaluating() {
    window.matchCount = 0;
    document.getElementById("arena-leaderboard-view").style.display = "none";
    document.getElementById("arena-annotation-view").style.display = "";
    nextMatch();
}

function nextMatch() {
    document.getElementById("arena-reveal").style.display = "none";
    document.getElementById("arena-criterion-panels").innerHTML = "";
    showLoading("Loading match...");

    fetch("/", {
        method: "POST",
        headers: { "Content-Type": "text/plain" },
        body: "arena_next_match::" + window.currentArena,
    })
        .then(function (r) { return r.json(); })
        .then(function (match) {
            hideLoading();
            window.currentMatch = match;
            window.matchCount++;
            document.getElementById("arena-match-counter").textContent = "Vote #" + window.matchCount;
            renderMatch(match);

            // Set up criteria panels
            var criteria = getMatchCriteria(match);
            window.currentCriteria = criteria;
            window.currentCriterionIndex = 0;
            window.criterionVotes = [];
            buildCriterionPanels(criteria, "annotate");
        })
        .catch(function (e) { hideLoading(); console.error("Failed to get match:", e); });
}

// ── Vote button configuration ──
// Returns the list of vote buttons for a criterion based on its mode.
// "preference" (default): 5 preference levels (A++, A+, Tie, B+, B++)
// "passfail": 4 binary options (Only A, Only B, Both Good, Both Bad)
// Each button has: cls (CSS class), code (vote value stored in DB), label, key (keyboard shortcut).

var VOTE_BUTTONS = {
    preference: [
        {cls: "pick-a-strong", code: "a_strong", label: "A is much better", key: "1"},
        {cls: "pick-a",        code: "a",        label: "A is better",      key: "2"},
        {cls: "tie",           code: "tie",      label: "Tie",              key: "3"},
        {cls: "pick-b",        code: "b",        label: "B is better",      key: "4"},
        {cls: "pick-b-strong", code: "b_strong", label: "B is much better", key: "5"},
    ],
    passfail: [
        {cls: "pick-a",  code: "a",   label: "Only A", key: "1"},
        {cls: "tie",     code: "tie", label: "Both Good", key: "2"},
        {cls: "pick-b",  code: "b",   label: "Only B", key: "3"},
    ],
};

// "Both Bad" and "Skip" are shared across all modes (rendered separately in the skip row).

function getVoteButtons(criterion) {
    var mode = (criterion && criterion.mode) || "preference";
    return VOTE_BUTTONS[mode] || VOTE_BUTTONS.preference;
}

function getVoteKeyMap() {
    // Build key → vote code map for the current criterion's mode
    var criterion = (window.currentCriteria || [])[window.currentCriterionIndex] || {};
    var buttons = getVoteButtons(criterion);
    var map = {};
    buttons.forEach(function (b) { map[b.key] = b.code; });
    return map;
}

function buildCriterionPanels(criteria, mode) {
    // mode: "annotate" (one active at a time) or "review" (all shown with results)
    var container = document.getElementById("arena-criterion-panels");
    container.innerHTML = "";

    criteria.forEach(function (c, idx) {
        var panel = document.createElement("div");
        panel.className = "arena-criterion-panel";
        panel.id = "criterion-panel-" + idx;

        // Header
        var header = document.createElement("div");
        header.className = "arena-criterion-panel-header";
        var nameSpan = document.createElement("span");
        nameSpan.className = "arena-criterion-panel-name";
        nameSpan.textContent = (criteria.length > 1 ? (idx + 1) + ". " : "") + c.name;
        header.appendChild(nameSpan);
        // Badge placeholder — filled after voting
        var badge = document.createElement("span");
        badge.className = "arena-criterion-panel-badge";
        badge.style.display = "none";
        badge.id = "criterion-badge-" + idx;
        header.appendChild(badge);
        panel.appendChild(header);

        // Body
        var body = document.createElement("div");
        body.className = "arena-criterion-panel-body";
        body.id = "criterion-body-" + idx;

        // Description
        var desc = document.createElement("div");
        desc.className = "arena-criterion-panel-desc";
        desc.textContent = c.description || "";
        body.appendChild(desc);

        if (mode === "annotate") {
            // Vote buttons — layout depends on criterion mode (preference vs passfail)
            var voteRow = document.createElement("div");
            voteRow.className = "arena-vote-row";
            getVoteButtons(c).forEach(function (b) {
                var btn = document.createElement("button");
                btn.className = "arena-vote-btn " + b.cls;
                btn.innerHTML = b.label + " <kbd>" + b.key + "</kbd>";
                btn.onclick = function () { submitVote(b.code); };
                voteRow.appendChild(btn);
            });
            body.appendChild(voteRow);

            // Both Bad + Skip (shared across all modes)
            var skipRow = document.createElement("div");
            skipRow.className = "arena-skip-row";
            var bothBadLabel = (c.mode === "passfail") ? "Both Bad" : "Both are bad";
            var bothBadBtn = document.createElement("button");
            bothBadBtn.className = "arena-skip-btn both-bad";
            bothBadBtn.innerHTML = bothBadLabel + " <kbd>0</kbd>";
            bothBadBtn.onclick = function () { submitVote("both_bad"); };
            skipRow.appendChild(bothBadBtn);
            var skipBtn = document.createElement("button");
            skipBtn.className = "arena-skip-btn";
            skipBtn.innerHTML = "Skip / N/A <kbd>S</kbd>";
            skipBtn.onclick = function () { skipCriterion(); };
            skipRow.appendChild(skipBtn);
            body.appendChild(skipRow);

            // Reasoning
            var reasonRow = document.createElement("div");
            reasonRow.className = "arena-reasoning-row";
            reasonRow.innerHTML =
                '<label class="arena-reasoning-label">Reasoning <span class="arena-reasoning-optional">(optional)</span></label>' +
                '<textarea class="arena-reasoning-input" rows="2" placeholder="Why did you pick this one?" id="criterion-reasoning-' + idx + '"></textarea>';
            body.appendChild(reasonRow);
        } else {
            // Review mode: show vote result + reasoning as text
            var cv = c._vote;  // attached by caller
            if (cv) {
                var reasonText = document.createElement("div");
                reasonText.className = "arena-criterion-panel-reasoning-text";
                reasonText.textContent = cv.reasoning || "";
                body.appendChild(reasonText);
            }
        }

        panel.appendChild(body);
        container.appendChild(panel);
    });

    if (mode === "annotate") {
        showCriterionPanel(0);
        window.votingActive = true;
        // Start per-vote timer now that the match is rendered and votable.
        resetMatchTimer();
    } else {
        window.votingActive = false;
    }
}

function showCriterionPanel(activeIdx) {
    var panels = document.querySelectorAll(".arena-criterion-panel");
    panels.forEach(function (panel, idx) {
        var body = document.getElementById("criterion-body-" + idx);
        if (idx === activeIdx) {
            panel.classList.remove("collapsed");
            body.style.display = "";
        } else {
            panel.classList.add("collapsed");
            body.style.display = "none";
        }
    });
    // Update progress dots
    var progressEl = document.getElementById("arena-criterion-progress");
    if (window.currentCriteria.length > 1) {
        progressEl.style.display = "";
        document.getElementById("arena-criterion-label").textContent =
            "Criterion " + (activeIdx + 1) + "/" + window.currentCriteria.length;
        var dotsEl = document.getElementById("arena-criterion-dots");
        dotsEl.innerHTML = "";
        for (var i = 0; i < window.currentCriteria.length; i++) {
            var dot = document.createElement("span");
            dot.className = "arena-criterion-dot" + (i < activeIdx ? " done" : (i === activeIdx ? " current" : ""));
            dotsEl.appendChild(dot);
        }
    } else {
        progressEl.style.display = "none";
    }
}

function renderMatch(match) {
    // Clear previous
    document.getElementById("arena-inputs").innerHTML = "";
    document.getElementById("arena-content-a").innerHTML = "";
    document.getElementById("arena-content-b").innerHTML = "";
    document.getElementById("arena-output-a").className = "arena-output";
    document.getElementById("arena-output-b").className = "arena-output";
    document.querySelector("#arena-output-a .arena-output-label").textContent = "Model A";
    document.querySelector("#arena-output-b .arena-output-label").textContent = "Model B";

    // Inputs
    var inputsEl = document.getElementById("arena-inputs");
    match.inputs.forEach(function (inp) {
        var block = document.createElement("div");
        block.className = "arena-input-block";
        var label = document.createElement("div");
        label.className = "arena-input-label";
        label.textContent = inp.label;
        block.appendChild(label);

        if (inp.type === "text") {
            block.appendChild(makeClippedText(inp.content, "arena-input-text"));
        } else if (inp.type === "json") {
            block.appendChild(renderJsonTree(inp.content));
        } else if (inp.type === "video") {
            var vid = document.createElement("video");
            vid.controls = true;
            vid.preload = "auto";
            vid.autoplay = true;
            vid.loop = true;
            vid.muted = true;
            if (inp.start_time != null || inp.end_time != null) {
                clampVideo(vid, inp.start_time, inp.end_time);
            }
            vid.src = inp.url;
            block.appendChild(vid);
        } else if (inp.type === "image") {
            block.appendChild(makeZoomableImage(inp.url));
        }
        inputsEl.appendChild(block);
    });

    // Outputs — render each output in the array
    renderOutputs("arena-content-a", match.outputs_a);
    renderOutputs("arena-content-b", match.outputs_b);

    // Sync videos — either all together or inputs/outputs separately
    var manifest = window.currentManifest || {};
    if (manifest.sync_all_videos) {
        syncVideoGroup("#arena-annotation-view video");
    } else {
        syncVideoGroup("#arena-inputs video");
        syncVideoGroup(".arena-comparison video");
    }
    showVideoControls();
}

function renderOutputs(containerId, outputs) {
    var el = document.getElementById(containerId);
    var showLabels = outputs.length > 1;
    outputs.forEach(function (output) {
        if (showLabels) {
            var lbl = document.createElement("div");
            lbl.className = "arena-input-label";
            lbl.textContent = output.label || output.name;
            el.appendChild(lbl);
        }
        if (output.type === "text") {
            var div = document.createElement("div");
            div.className = "arena-text-output";
            div.textContent = output.content || "";
            el.appendChild(div);
        } else if (output.type === "json") {
            el.appendChild(renderJsonTree(output.content));
        } else if (output.type === "video") {
            var vid = document.createElement("video");
            vid.controls = true;
            vid.preload = "auto";
            vid.autoplay = true;
            vid.loop = true;
            vid.muted = true;
            vid.src = output.url;
            el.appendChild(vid);
        } else if (output.type === "image") {
            el.appendChild(makeZoomableImage(output.url));
        }
    });
}

// ── Synchronized video playback ──

function syncVideoGroup(selector) {
    var vids = Array.from(document.querySelectorAll(selector));
    if (vids.length < 2) return;

    // Disable per-video loop and autoplay — we start them together
    vids.forEach(function (v) {
        v.loop = false;
        v.pause();
    });

    // Wait for all videos to be ready, then play in sync
    var readyCount = 0;
    function onReady() {
        readyCount++;
        if (readyCount >= vids.length) {
            vids.forEach(function (v) { v.currentTime = 0; v.play(); });
        }
    }
    vids.forEach(function (v) {
        if (v.readyState >= 3) { onReady(); }
        else { v.addEventListener("canplay", onReady, { once: true }); }
    });

    var syncing = false;

    function syncFrom(source) {
        if (syncing) return;
        // Ignore events from a video that has reached its end
        var srcDur = source.duration || Infinity;
        if (source.paused && source.currentTime >= srcDur - 0.1) return;

        syncing = true;
        vids.forEach(function (v) {
            if (v === source) return;
            var dur = v.duration || Infinity;
            if (source.currentTime >= dur) {
                v.currentTime = Math.max(0, dur - 0.1);
                if (!v.paused) v.pause();
            } else {
                if (!source.paused && v.paused) v.play();
                if (source.paused && !v.paused) v.pause();
                if (Math.abs(source.currentTime - v.currentTime) > 0.3) {
                    v.currentTime = source.currentTime;
                }
            }
        });
        syncing = false;
    }

    vids.forEach(function (v) {
        v.addEventListener("play", function () { syncFrom(v); });
        v.addEventListener("pause", function () { syncFrom(v); });
        v.addEventListener("seeked", function () { syncFrom(v); });
    });
}

// ── Video playback controls ──

function showVideoControls() {
    var ctrl = document.getElementById("arena-video-controls");
    var hasVideo = document.querySelectorAll("#arena-annotation-view video").length > 0;
    ctrl.style.display = hasVideo ? "" : "none";
    // Restore saved speed (default 1x)
    var saved = parseFloat(localStorage.getItem("arena-vc-speed")) || 1;
    setSpeed(saved);
    updatePlayPauseIcon(true);
}

function setSpeed(rate) {
    document.querySelectorAll("#arena-annotation-view video").forEach(function (v) {
        v.playbackRate = rate;
    });
    var ctrl = document.getElementById("arena-video-controls");
    Array.from(ctrl.querySelectorAll(".arena-vc-speed")).forEach(function (b) {
        b.classList.toggle("active", parseFloat(b.textContent) === rate);
    });
    localStorage.setItem("arena-vc-speed", rate);
}

function togglePlayPause() {
    var vids = document.querySelectorAll("#arena-annotation-view video");
    var playing = Array.from(vids).some(function (v) { return !v.paused; });
    vids.forEach(function (v) { playing ? v.pause() : v.play(); });
    updatePlayPauseIcon(!playing);
}

function updatePlayPauseIcon(playing) {
    var btn = document.getElementById("arena-vc-playpause");
    if (btn) btn.innerHTML = playing ? "&#9646;&#9646;" : "&#9654;";
}

function stepFrame(dir) {
    var dt = dir / 24;  // ~1 frame at 24fps, clamped to [0, duration]
    document.querySelectorAll("#arena-annotation-view video").forEach(function (v) {
        var dur = v.duration || 0;
        if (!dur) return;
        v.pause();
        v.currentTime = Math.min(Math.max(0, v.currentTime + dt), dur);
    });
    updatePlayPauseIcon(false);
}

// ── Voting ──

function submitVote(winner) {
    if (!window.currentMatch || !window.votingActive) return;
    window.votingActive = false;  // prevent double-click

    var idx = window.currentCriterionIndex;
    var m = window.currentMatch;
    var criterion = window.currentCriteria[idx].name;

    // Read reasoning from this panel's textarea
    var reasoningEl = document.getElementById("criterion-reasoning-" + idx);
    var reasoning = reasoningEl ? (reasoningEl.value || "").trim() : "";

    var durationMs = consumeVoteDurationMs();
    var durationStr = durationMs == null ? "" : String(durationMs);
    var payload = "arena_submit_vote::" + m.arena_name + "::" + m.match_id + "::" + m.item_id + "::" + m.model_a + "::" + m.model_b + "::" + winner + "::" + reasoning + "::" + criterion + "::" + durationStr;

    fetch("/", {
        method: "POST",
        headers: { "Content-Type": "text/plain" },
        body: payload,
    })
        .then(function (r) { return r.json(); })
        .then(function (result) {
            window.criterionVotes.push(result);

            // Collapse this panel with a badge showing the vote
            collapsePanel(idx, winner);

            // If more criteria, advance; otherwise reveal
            if (idx < window.currentCriteria.length - 1) {
                window.currentCriterionIndex = idx + 1;
                showCriterionPanel(idx + 1);
                window.votingActive = true;
            } else {
                document.getElementById("arena-criterion-progress").style.display = "none";
                showReveal(window.criterionVotes);
            }
        })
        .catch(function (e) { console.error("Vote failed:", e); window.votingActive = true; });
}

function collapsePanel(idx, winner) {
    var panel = document.getElementById("criterion-panel-" + idx);
    var body = document.getElementById("criterion-body-" + idx);
    var badge = document.getElementById("criterion-badge-" + idx);

    // Hide body, show badge
    body.style.display = "none";
    panel.classList.add("collapsed");

    badge.textContent = fmtWinner(winner);
    badge.className = "arena-criterion-panel-badge " + winnerBadgeClass(winner);
    badge.style.display = "";
}

function skipCriterion() {
    // Routed through submitVote so the skip is persisted (with duration_ms) and
    // de-duplicated by (match_id, criterion). Backend records the row but does
    // not move ratings for winner == "skip".
    submitVote("skip");
}

function showReveal(results) {
    // results is an array of per-criterion vote results
    var first = results[0] || {};
    document.getElementById("arena-reveal").style.display = "";

    var nextBtn = document.getElementById("arena-next-btn");
    nextBtn.style.display = "";
    nextBtn.textContent = "Next Vote →";
    nextBtn.onclick = nextMatch;

    document.getElementById("reveal-name-a").textContent = first.model_a_name;
    document.getElementById("reveal-name-b").textContent = first.model_b_name;

    // Show per-criterion rating changes in the panel badges (skips render as +0 / +0).
    results.forEach(function (r, idx) {
        var badge = document.getElementById("criterion-badge-" + idx);
        if (badge) {
            var delta = (r.rating_change_a >= 0 ? "+" : "") + r.rating_change_a + " / " +
                        (r.rating_change_b >= 0 ? "+" : "") + r.rating_change_b;
            badge.textContent = badge.textContent + "  " + delta;
        }
    });

    // Aggregate rating changes for the reveal header
    var totalChangeA = results.reduce(function (s, r) { return s + r.rating_change_a; }, 0);
    var totalChangeB = results.reduce(function (s, r) { return s + r.rating_change_b; }, 0);
    var eloA = document.getElementById("reveal-elo-a");
    var eloB = document.getElementById("reveal-elo-b");
    if (results.length === 1) {
        eloA.textContent = (first.rating_change_a >= 0 ? "+" : "") + first.rating_change_a + " (" + first.rating_a + ")";
        eloB.textContent = (first.rating_change_b >= 0 ? "+" : "") + first.rating_change_b + " (" + first.rating_b + ")";
    } else {
        eloA.textContent = "";
        eloB.textContent = "";
    }
    eloA.className = "arena-elo-change " + (totalChangeA > 0 ? "positive" : totalChangeA < 0 ? "negative" : "neutral");
    eloB.className = "arena-elo-change " + (totalChangeB > 0 ? "positive" : totalChangeB < 0 ? "negative" : "neutral");

    // Update output labels with real names
    document.querySelector("#arena-output-a .arena-output-label").textContent = first.model_a_name;
    document.querySelector("#arena-output-b .arena-output-label").textContent = first.model_b_name;
}

// ── URL hash ──
// Supported hash params:
//   #name=<arena>             — select and load an arena
//   #name=<arena>&match=<id>  — open a specific match in review mode (permalink)

function parseHash() {
    var hash = window.location.hash.replace("#", "");
    var params = {};
    hash.split("&").forEach(function (part) {
        var kv = part.split("=");
        if (kv.length === 2) params[kv[0]] = decodeURIComponent(kv[1]);
    });
    return params;
}

function applyHash() {
    var params = parseHash();
    if (params.name) {
        selectArena(params.name, params.match || null);
    }
}

window.addEventListener("hashchange", applyHash);

// ── Util ──

var WIN_CODE_TO_LABEL = {"a_strong": "A++", "a": "A+", "b_strong": "B++", "b": "B+", "both_bad": "Both Bad", "skip": "Skip / N/A", "tie": "Tie"};

function fmtWinner(code) {
    return WIN_CODE_TO_LABEL[code] || "Tie";
}

function winnerBadgeClass(winner) {
    if (winner === "a" || winner === "a_strong") return "pick-a";
    if (winner === "b" || winner === "b_strong") return "pick-b";
    if (winner === "both_bad") return "both-bad";
    return "tie";
}

function getMatchCriteria(match) {
    return match.criteria || [{"name": "overall", "description": match.instructions || ""}];
}

// ── Shared rating line chart ──
// Renders a rating-over-time chart on the given canvas.
// chartRefName is a window property name for the Chart instance (for destroy on re-render).
// historyData is {model: [{match, rating}, ...]}

function createEloLineChart(canvasId, chartRefName, historyData) {
    if (window[chartRefName]) { window[chartRefName].destroy(); }
    var canvas = document.getElementById(canvasId);
    var section = canvas.parentElement;
    var models = Object.keys(historyData);
    if (models.length === 0) { section.style.display = "none"; return; }
    section.style.display = "";

    var maxPts = Math.max.apply(null, models.map(function (m) { return historyData[m].length; }));
    var dotRadius = maxPts > 50 ? 0 : 3;
    var datasets = models.map(function (model) {
        var pts = historyData[model];
        return {
            label: model,
            data: pts.map(function (p) { return { x: p.match, y: p.rating }; }),
            borderColor: modelColor(model),
            backgroundColor: "transparent",
            borderWidth: 2,
            pointRadius: dotRadius,
            tension: 0.1,
        };
    });

    window[chartRefName] = new Chart(canvas.getContext("2d"), {
        type: "line",
        data: { datasets: datasets },
        options: {
            responsive: true,
            animation: false,
            interaction: { mode: "index", intersect: false },
            scales: {
                x: { type: "linear", title: { display: true, text: "Vote #" } },
                y: { title: { display: true, text: "Rating" } },
            },
            plugins: {
                tooltip: { mode: "index", intersect: false, itemSort: function (a, b) { return b.raw.y - a.raw.y; } },
                legend: { position: "top" },
            },
        },
    });
}

function modelColor(model) {
    var models = (window.currentManifest || {}).models || [];
    var idx = models.indexOf(model);
    if (idx === -1) idx = models.length;
    var hue = (idx * 137) % 360;  // golden angle spread
    return "hsl(" + hue + ", 70%, 50%)";
}

function fmtRating(r) {
    // 95% CI = rating ± 1.96 · RD. RD is carried on every ranking row by the backend.
    var pm = Math.round(1.96 * r.rd);
    var low = Math.round((r.rating - 1.96 * r.rd) * 10) / 10;
    var high = Math.round((r.rating + 1.96 * r.rd) * 10) / 10;
    return r.rating + ' <span class="arena-ci" title="95% CI: ' + low + ' \u2013 ' + high + '">\u00B1' + pm + '</span>';
}

function renderEloTable(leaderboardData) {
    if (window.eloTable) {
        window.eloTable.destroy();
        document.querySelector("#arena-elo-table tbody").innerHTML = "";
    }
    var numModels = leaderboardData.rankings.length;
    var ranked = leaderboardData.rankings.filter(function (r) { return r.matches >= numModels; });
    var provisional = leaderboardData.rankings.filter(function (r) { return r.matches < numModels; });

    var rows = [];
    ranked.forEach(function (r, i) {
        rows.push([i + 1, r.model, fmtRating(r), r.matches, r.wins, r.losses, r.ties, r.win_rate + "%", ""]);
    });
    provisional.forEach(function (r) {
        rows.push(["-", r.model, fmtRating(r), r.matches, r.wins, r.losses, r.ties, r.win_rate + "%", "provisional"]);
    });
    window.eloTable = $("#arena-elo-table").DataTable({
        data: rows,
        paging: false,
        searching: false,
        info: false,
        ordering: false,
        columnDefs: [{ targets: 8, visible: false }],
        createdRow: function (row, data) {
            if (data[8] === "provisional") {
                $(row).addClass("arena-provisional-row");
                row.title = "Provisional — needs at least " + numModels + " votes (" + data[3] + " so far) for a stable ranking";
            }
            $("td:eq(2)", row).html(data[2]);
        },
    });
}

function clampVideo(vid, startTime, endTime) {
    if (startTime == null && endTime == null) return;
    var start = startTime != null ? startTime : 0;
    var end = endTime != null ? endTime : Infinity;

    vid.addEventListener("loadedmetadata", function () {
        vid.currentTime = start;
        vid.play();
    });
    vid.addEventListener("timeupdate", function () {
        if (vid.currentTime < start) vid.currentTime = start;
        if (vid.currentTime >= end) {
            vid.currentTime = start;
            vid.play();
        }
    });
    vid.addEventListener("seeking", function () {
        if (vid.currentTime < start) vid.currentTime = start;
        if (vid.currentTime > end) vid.currentTime = start;
    });
}

// ── Image lightbox ──
// Creates an <img> wrapped in a container with a zoom button.
// Clicking the zoom button opens the image in a fullscreen <dialog> lightbox.
// Close with Escape, clicking the backdrop, or the close button.

function makeZoomableImage(src) {
    var img = document.createElement("img");
    img.src = src;
    img.className = "arena-img-zoomable";
    img.onclick = function () {
        var dialog = document.getElementById("arena-lightbox");
        document.getElementById("arena-lightbox-img").src = src;
        dialog.showModal();
    };
    return img;
}

// ── Item labels ──
// Parses manifest.item_labels (CSV strings) into lookup maps.
// Called once per arena load in renderLeaderboard.

function parseItemLabels(manifest) {
    window.itemLabelMap = {};
    window.allLabels = [];
    var raw = (manifest && manifest.item_labels) || {};
    var labelSet = new Set();
    Object.keys(raw).forEach(function (id) {
        var labels = raw[id].split(",").map(function (s) { return s.trim(); }).filter(Boolean);
        window.itemLabelMap[id] = new Set(labels);
        labels.forEach(function (l) { labelSet.add(l); });
    });
    window.allLabels = Array.from(labelSet).sort();
}

function filterVotesByLabel(votes, label) {
    if (!label) return votes;
    return votes.filter(function (v) {
        var labels = window.itemLabelMap[v.input_id];
        return labels && labels.has(label);
    });
}

function escapeHtml(str) {
    var div = document.createElement("div");
    div.appendChild(document.createTextNode(str || ""));
    return div.innerHTML;
}

// Collapsible JSON tree renderer — reused for both arena-input-json and arena-json-output.
// Falls back to a <pre> with raw text if the content isn't valid JSON.
function renderJsonTree(content) {
    var data;
    try { data = JSON.parse(content || "null"); }
    catch (e) {
        var pre = document.createElement("pre");
        pre.className = "arena-json-output";
        pre.textContent = content || "";
        return pre;
    }

    var container = document.createElement("div");
    container.className = "arena-json-tree";
    if (data !== null && typeof data === "object") {
        var isArr = Array.isArray(data);
        var keys = isArr ? data.map(function (_, i) { return i; }) : Object.keys(data);
        keys.forEach(function (k) {
            container.appendChild(jsonTreeNode(isArr ? data[k] : data[k], k));
        });
    } else {
        container.appendChild(jsonTreeNode(data, null));
    }
    return container;
}

function jsonTreeNode(value, key) {
    var isContainer = (value !== null && typeof value === "object");

    if (!isContainer) {
        var leaf = document.createElement("div");
        leaf.className = "arena-json-leaf";
        if (key !== null) {
            var keyEl = document.createElement("span");
            keyEl.className = "arena-json-key";
            keyEl.textContent = key + ":";
            leaf.appendChild(keyEl);
            leaf.appendChild(document.createTextNode(" "));
        }
        var valEl = document.createElement("span");
        valEl.className = "arena-json-val";
        valEl.textContent = (typeof value === "string") ? value : JSON.stringify(value);
        leaf.appendChild(valEl);
        return leaf;
    }

    var details = document.createElement("details");
    details.className = "arena-json-node";

    var summary = document.createElement("summary");
    summary.className = "arena-json-summary";
    var isArr = Array.isArray(value);
    var count = isArr ? value.length : Object.keys(value).length;
    if (key !== null) {
        var ks = document.createElement("span");
        ks.className = "arena-json-key";
        ks.textContent = key + ":";
        summary.appendChild(ks);
        summary.appendChild(document.createTextNode(" "));
    }
    var meta = document.createElement("span");
    meta.className = "arena-json-meta";
    meta.textContent = isArr ? "[ " + count + " items ]" : "{ " + count + " keys }";
    summary.appendChild(meta);
    details.appendChild(summary);

    var children = document.createElement("div");
    children.className = "arena-json-children";
    if (isArr) {
        value.forEach(function (v, i) { children.appendChild(jsonTreeNode(v, i)); });
    } else {
        Object.keys(value).forEach(function (k) { children.appendChild(jsonTreeNode(value[k], k)); });
    }
    details.appendChild(children);
    return details;
}

var TEXT_CLIP_LENGTH = 600;

function makeClippedText(content, className) {
    var container = document.createElement("div");
    container.className = "arena-clipped-container";

    var el = document.createElement("div");
    el.className = className;

    if ((content || "").length <= TEXT_CLIP_LENGTH) {
        el.textContent = content || "";
        container.appendChild(el);
        return container;
    }

    el.textContent = content.slice(0, TEXT_CLIP_LENGTH) + "...";
    container.appendChild(el);

    var btn = document.createElement("button");
    btn.className = "arena-expand-btn";
    btn.textContent = "Show more";
    var expanded = false;
    btn.onclick = function () {
        expanded = !expanded;
        el.textContent = expanded ? content : content.slice(0, TEXT_CLIP_LENGTH) + "...";
        btn.textContent = expanded ? "Show less" : "Show more";
    };
    container.appendChild(btn);
    return container;
}

// ── Review past match ──

// Review navigation state
window.reviewHistory = [];
window.reviewIndex = -1;

function reviewMatch(matchId, username) {
    // Build review list from current history table data if not already navigating
    if (window.reviewIndex === -1 && window.historyTable) {
        window.reviewHistory = window.historyTable.rows({ order: "current" }).data().toArray();
        for (var i = 0; i < window.reviewHistory.length; i++) {
            if (window.reviewHistory[i][7] === matchId) {
                window.reviewIndex = i;
                break;
            }
        }
    }

    document.getElementById("arena-leaderboard-view").style.display = "none";
    document.getElementById("arena-annotation-view").style.display = "";
    var isVLM = username && username.indexOf("vlm_judge:") === 0;
    document.getElementById("arena-match-counter").textContent =
        "Reviewing match" + (window.reviewHistory.length > 0 ? " " + (window.reviewIndex + 1) + "/" + window.reviewHistory.length : "")
        + (isVLM ? " — " + username : "");
    window.currentMatch = null;
    window.votingActive = false;
    document.getElementById("arena-criterion-progress").style.display = "none";

    fetch("/", {
        method: "POST",
        headers: { "Content-Type": "text/plain" },
        body: "arena_review_match::" + window.currentArena + "::" + matchId,
    })
        .then(function (r) { return r.json(); })
        .then(function (match) {
            renderMatch(match);

            // Build criterion panels in review mode — attach vote data to each criterion
            var cvotes = match.criterion_votes || [];
            var criteria = getMatchCriteria(match).map(function (c) {
                var vote = cvotes.find(function (v) { return v.criterion === c.name; });
                return Object.assign({}, c, {_vote: vote || null});
            });
            buildCriterionPanels(criteria, "review");

            // Show vote badges on each panel (panels stay expanded in review mode)
            criteria.forEach(function (c, idx) {
                if (c._vote) {
                    var badge = document.getElementById("criterion-badge-" + idx);
                    badge.textContent = fmtWinner(c._vote.winner);
                    badge.className = "arena-criterion-panel-badge " + winnerBadgeClass(c._vote.winner);
                    badge.style.display = "";
                }
            });

            // Show reveal with model names
            document.getElementById("arena-reveal").style.display = "";
            document.querySelector("#arena-output-a .arena-output-label").textContent = match.model_a;
            document.querySelector("#arena-output-b .arena-output-label").textContent = match.model_b;
            document.getElementById("reveal-name-a").textContent = match.model_a;
            document.getElementById("reveal-name-b").textContent = match.model_b;
            document.getElementById("reveal-elo-a").textContent = "";
            document.getElementById("reveal-elo-b").textContent = "";
            document.getElementById("reveal-elo-a").className = "arena-elo-change";
            document.getElementById("reveal-elo-b").className = "arena-elo-change";

            // Set up Review Next button
            var nextBtn = document.getElementById("arena-next-btn");
            if (window.reviewIndex >= 0 && window.reviewIndex < window.reviewHistory.length - 1) {
                nextBtn.style.display = "";
                nextBtn.textContent = "Review Next →";
                nextBtn.onclick = function () {
                    window.reviewIndex++;
                    var next = window.reviewHistory[window.reviewIndex];
                    reviewMatch(next[7], next[6]);
                };
            } else {
                nextBtn.style.display = "none";
            }
        })
        .catch(function (e) { console.error("Failed to load match for review:", e); });
}

// ── Owner controls ──

function reloadArena(arenaName) {
    // Refresh the arena list, then re-select the current arena
    fetch("/arena/list")
        .then(function (r) { return r.json(); })
        .then(function (data) {
            window.arenas = data.arenas || [];
            renderArenaList();
            selectArena(arenaName);
        });
}

function arenaPublish(arenaName) {
    fetch("/", { method: "POST", headers: { "Content-Type": "text/plain" }, body: "arena_publish::" + arenaName })
        .then(function (r) { return r.json(); })
        .then(function () { reloadArena(arenaName); })
        .catch(function (e) { console.error("Publish failed:", e); });
}

function arenaUnpublish(arenaName) {
    fetch("/", { method: "POST", headers: { "Content-Type": "text/plain" }, body: "arena_unpublish::" + arenaName })
        .then(function (r) { return r.json(); })
        .then(function () { reloadArena(arenaName); })
        .catch(function (e) { console.error("Unpublish failed:", e); });
}

function arenaRefreshManifest(arenaName) {
    var btn = document.getElementById("arena-refresh-btn");
    btn.disabled = true;
    btn.textContent = "Refreshing...";
    fetch("/", { method: "POST", headers: { "Content-Type": "text/plain" }, body: "arena_refresh_manifest::" + arenaName })
        .then(function (r) { return r.json(); })
        .then(function () {
            btn.disabled = false;
            btn.textContent = "Refresh Manifest";
            reloadArena(arenaName);
        })
        .catch(function (e) {
            console.error("Refresh failed:", e);
            btn.disabled = false;
            btn.textContent = "Refresh Manifest";
        });
}

function deleteMatch(matchId) {
    var arenaName = window.currentArena;
    if (!arenaName || !matchId) return;
    if (!confirm("Delete this vote and replay ratings?\n\nMatch ID: " + matchId + "\n\nThis cannot be undone.")) return;

    fetch("/", {
        method: "POST",
        headers: { "Content-Type": "text/plain" },
        body: "arena_delete_match::" + arenaName + "::" + matchId,
    })
        .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, body: j }; }); })
        .then(function (res) {
            if (!res.ok) {
                alert("Delete failed: " + (res.body && res.body.error || "unknown error"));
                return;
            }
            // Reload everything: leaderboard, rating chart, history, CI — ratings were just rebuilt.
            reloadArena(arenaName);
        })
        .catch(function (e) {
            console.error("Delete failed:", e);
            alert("Delete failed: " + e);
        });
}

function arenaRunVLMJudge(arenaName) {
    var countInput = document.getElementById("arena-vlm-judge-count");
    var num = Math.max(1, Math.min(100, parseInt(countInput.value) || 10));
    countInput.value = num;

    if (!confirm("Run VLM judge for " + num + " matches on this arena?\n\nThis will submit automated votes that affect model ratings.")) {
        return;
    }

    var btn = document.getElementById("arena-vlm-judge-btn");
    btn.disabled = true;
    btn.textContent = "🔁 Judging...";
    countInput.disabled = true;

    fetch("/", {
        method: "POST",
        headers: { "Content-Type": "text/plain" },
        body: "arena_vlm_judge_batch::" + arenaName + "::" + num,
    })
        .then(function (r) { return r.json(); })
        .then(function (result) {
            btn.disabled = false;
            btn.textContent = "🔁 VLM Judge";
            countInput.disabled = false;

            if (result.error) {
                alert("VLM Judge error: " + result.error);
                return;
            }

            alert("VLM Judge started: " + result.num_matches + " matches queued.\nVotes will appear in the history as they complete — refresh to check progress.");
        })
        .catch(function (e) {
            console.error("VLM Judge failed:", e);
            btn.disabled = false;
            btn.textContent = "🔁 VLM Judge";
            countInput.disabled = false;
            alert("VLM Judge request failed. Check console for details.");
        });
}

function showArenaHelp() {
    document.getElementById("arena-help-content").style.display = "block";
}

function hideArenaHelp() {
    document.getElementById("arena-help-content").style.display = "none";
}

// ── Sidebar collapse ──

function toggleArenaSidebar() {
    var sidebar = document.getElementById("arena-sidebar");
    var expand = document.getElementById("arena-sidebar-expand");
    var content = document.querySelector(".main-content");
    var collapsed = sidebar.classList.toggle("collapsed");
    expand.style.display = collapsed ? "" : "none";
    content.classList.toggle("arena-sidebar-collapsed", collapsed);
}

// Auto-collapse sidebar on small screens and on resize
(function () {
    function autoCollapse() {
        var sidebar = document.getElementById("arena-sidebar");
        if (!sidebar) return;
        var isCollapsed = sidebar.classList.contains("collapsed");
        if (window.innerWidth < 900 && !isCollapsed) {
            toggleArenaSidebar();
        } else if (window.innerWidth >= 900 && isCollapsed) {
            toggleArenaSidebar();
        }
    }
    window.addEventListener("DOMContentLoaded", autoCollapse);
    window.addEventListener("resize", autoCollapse);
})();

// ── Metric formatting helpers ──

function formatTauCell(tau) {
    if (tau === null) return '<span class="arena-analytics-na">N/A</span>';
    return tau.toFixed(3);
}

function formatInfluenceCell(value) {
    if (value === null) return '<span class="arena-analytics-na">N/A</span>';
    return value.toFixed(1);
}

function formatRankShiftsCell(shifts, totalModels) {
    // Show "X of N" for rank shifts so users understand scale
    if (shifts === null) return '<span class="arena-analytics-na">N/A</span>';
    if (totalModels !== null) return shifts + '<span class="arena-analytics-context"> of ' + totalModels + '</span>';
    return String(shifts);
}

// ── Analytics ──
// Fetches all votes once, computes agreement metrics and filtered ratings client-side.
// All filter changes recompute instantly (no server round-trip).

var ANALYTICS_CACHE_TTL = 30000; // 30s — refetch on tab switch if stale

// Register custom DataTables sort types once — used by both annotator and data tables.
// "nullable": normal numeric sort, nulls always last.
// "abs-nullable": sort by absolute value, nulls always last.
(function () {
    var ext = $.fn.dataTable.ext.type.order;
    ext["nullable-pre"] = function (d) { return d; };
    ext["nullable-asc"] = function (a, b) {
        if (a === null && b === null) return 0;
        if (a === null) return 1;
        if (b === null) return -1;
        return a - b;
    };
    ext["nullable-desc"] = function (a, b) {
        if (a === null && b === null) return 0;
        if (a === null) return 1;
        if (b === null) return -1;
        return b - a;
    };
    ext["abs-nullable-pre"] = function (d) { return d; };
    ext["abs-nullable-asc"] = function (a, b) {
        if (a === null && b === null) return 0;
        if (a === null) return 1;
        if (b === null) return -1;
        return Math.abs(a) - Math.abs(b);
    };
    ext["abs-nullable-desc"] = function (a, b) {
        if (a === null && b === null) return 0;
        if (a === null) return 1;
        if (b === null) return -1;
        return Math.abs(b) - Math.abs(a);
    };
})();

// ── Analytics: data loading ──

function loadAnalytics(forceRefresh) {
    var arenaName = window.currentArena;
    if (!arenaName) return;
    var now = Date.now();
    var cacheValid = !forceRefresh
        && arenaName === window.analyticsVotesArena
        && now - window.analyticsVotesFetchedAt < ANALYTICS_CACHE_TTL
        && window.analyticsNormalized;
    if (cacheValid) { renderAnalytics(); return; }

    fetch("/arena/votes?name=" + encodeURIComponent(arenaName))
        .then(function (r) { return r.json(); })
        .then(function (data) {
            window.analyticsVotes = data.votes || [];
            window.analyticsVotesArena = arenaName;
            window.analyticsVotesFetchedAt = Date.now();
            window.analyticsNormalized = normalizeVotes(window.analyticsVotes);
            renderAnalytics();
        })
        .catch(function (e) { console.error("Failed to load analytics votes:", e); });
}

function resetAnalyticsFilters() {
    window.analyticsFilters = { excludedUsers: new Set(), excludedItems: new Set(), criterion: null, labels: null };
    renderAnalytics();
}

// ── Analytics: vote normalization ──
// Canonicalizes A/B assignment so agreement can be computed across matches
// where the same item+pair appeared with different A/B ordering.

var WINNER_FLIP = { a: "b", b: "a", a_strong: "b_strong", b_strong: "a_strong" };
var DIRECTION_MAP = { a: "left", a_strong: "left", b: "right", b_strong: "right", tie: "draw", both_bad: "draw" };

function normalizeVotes(votes) {
    return votes.filter(function (v) {
        return v.winner !== "skip";
    }).map(function (v) {
        var left = v.model_a < v.model_b ? v.model_a : v.model_b;
        var right = v.model_a < v.model_b ? v.model_b : v.model_a;
        var swapped = v.model_a !== left;
        var canonicalWinner = swapped ? (WINNER_FLIP[v.winner] || v.winner) : v.winner;
        return {
            match_id: v.match_id,
            input_id: v.input_id,
            model_a: v.model_a,
            model_b: v.model_b,
            winner: v.winner,
            username: v.username,
            created_at: v.created_at,
            criterion: v.criterion,
            reasoning: v.reasoning,
            canonicalPair: left + "|" + right,
            canonicalWinner: canonicalWinner,
            direction: DIRECTION_MAP[canonicalWinner] || "draw",
        };
    });
}

// ── Analytics: ranking utilities ──

function kendallTau(rankA, rankB) {
    // Kendall's τ-b between two rating maps (model → rating).
    // Only considers models present in both.
    var models = Object.keys(rankA).filter(function (m) { return m in rankB; });
    if (models.length < 2) return null;
    var concordant = 0, discordant = 0;
    for (var i = 0; i < models.length; i++) {
        for (var j = i + 1; j < models.length; j++) {
            var product = (rankA[models[i]] - rankA[models[j]]) * (rankB[models[i]] - rankB[models[j]]);
            if (product > 0) concordant++;
            else if (product < 0) discordant++;
        }
    }
    var n = concordant + discordant;
    return n === 0 ? null : (concordant - discordant) / n;
}

function ratingsMap(replayed) {
    // Convert a replay result to {model: rating} map (only models with matches)
    var map = {};
    replayed.rankings.forEach(function (r) { if (r.matches > 0) map[r.model] = r.rating; });
    return map;
}

function ranksMap(replayed) {
    // Convert a replay result to {model: rank} map (1-based, only models with matches)
    var map = {};
    replayed.rankings.forEach(function (r) { if (r.matches > 0) map[r.model] = r.rank; });
    return map;
}

// ── Analytics: solo τ ──
// Replay ratings using only one entity's votes, compare to consensus ranking.
// Works for both annotators (groupKey = "username") and data points (groupKey = "input_id").

function computeSoloTau(votes, groupKey, consensusRatings, cfg) {
    var byEntity = {};
    votes.forEach(function (v) {
        var key = v[groupKey];
        if (!byEntity[key]) byEntity[key] = [];
        byEntity[key].push(v);
    });

    var result = {}; // entity → tau
    var MIN_VOTES = 3;
    Object.keys(byEntity).forEach(function (entity) {
        if (byEntity[entity].length < MIN_VOTES) return;
        var soloRatings = ratingsMap(replayRatings(byEntity[entity], cfg));
        if (Object.keys(soloRatings).length < 2) return;
        var tau = kendallTau(soloRatings, consensusRatings);
        if (tau !== null) result[entity] = Math.round(tau * 1000) / 1000;
    });
    return result;
}

// ── Analytics: pairwise annotator agreement ──
// Computes Kendall's τ between each pair of annotators' solo Glicko-2 rankings.
// No direct item overlap needed — each annotator's votes independently produce a ranking.

function computePairwiseTau(votes, cfg, orderedUsernames) {
    // Group votes by annotator and replay solo rating for each.
    // orderedUsernames controls display order (e.g. from annotator table sort).
    var byUser = {};
    votes.forEach(function (v) {
        if (!byUser[v.username]) byUser[v.username] = [];
        byUser[v.username].push(v);
    });
    var MIN_VOTES = 3;
    var soloRatings = {}; // username → {model: rating}
    var usernames = [];
    (orderedUsernames || Object.keys(byUser)).forEach(function (u) {
        if (!byUser[u] || byUser[u].length < MIN_VOTES) return;
        var ratings = ratingsMap(replayRatings(byUser[u], cfg));
        if (Object.keys(ratings).length < 2) return;
        soloRatings[u] = ratings;
        usernames.push(u);
    });

    // Pairwise τ matrix
    var matrix = {}; // "userA|userB" → tau (or null)
    for (var i = 0; i < usernames.length; i++) {
        for (var j = i + 1; j < usernames.length; j++) {
            var tau = kendallTau(soloRatings[usernames[i]], soloRatings[usernames[j]]);
            var key = usernames[i] + "|" + usernames[j];
            matrix[key] = tau !== null ? Math.round(tau * 1000) / 1000 : null;
        }
    }
    return { usernames: usernames, matrix: matrix };
}

function renderAgreementMatrix(pairwise) {
    var el = document.getElementById("analytics-agreement-matrix");
    var users = pairwise.usernames;
    if (users.length < 2) { el.innerHTML = ""; return; }

    var excluded = window.analyticsFilters.excludedUsers;
    var html = '<div class="arena-agreement-matrix-wrap">' +
        '<h4 class="arena-agreement-matrix-title">Pairwise Alignment</h4>' +
        '<table class="arena-agreement-matrix">' +
        '<thead><tr><th></th>';
    users.forEach(function (u) {
        var cls = excluded.has(u) ? ' class="arena-matrix-excluded"' : '';
        html += '<th' + cls + ' title="' + escapeHtml(u) + '">' + escapeHtml(u) + '</th>';
    });
    html += '</tr></thead><tbody>';

    users.forEach(function (rowUser, i) {
        var rowExcluded = excluded.has(rowUser);
        html += '<tr><th' + (rowExcluded ? ' class="arena-matrix-excluded"' : '') + ' title="' + escapeHtml(rowUser) + '">' + escapeHtml(rowUser) + '</th>';
        users.forEach(function (colUser, j) {
            var cellExcluded = rowExcluded || excluded.has(colUser);
            if (i === j) {
                html += '<td class="arena-matrix-diag' + (cellExcluded ? ' arena-matrix-excluded' : '') + '">&mdash;</td>';
            } else {
                var key = i < j ? (rowUser + "|" + colUser) : (colUser + "|" + rowUser);
                var tau = pairwise.matrix[key];
                if (tau === null) {
                    html += '<td class="arena-analytics-na' + (cellExcluded ? ' arena-matrix-excluded' : '') + '" title="Insufficient shared models">N/A</td>';
                } else {
                    var bg = cellExcluded ? 'transparent' : tauToColor(tau);
                    var cls = cellExcluded ? ' class="arena-matrix-excluded"' : '';
                    html += '<td' + cls + ' style="background:' + bg + ';" title="τ = ' + tau.toFixed(3) + '">' + tau.toFixed(2) + '</td>';
                }
            }
        });
        html += '</tr>';
    });
    html += '</tbody></table></div>';
    el.innerHTML = html;
}

function tauToColor(tau) {
    // Map τ ∈ [-1, 1] to a background color: red ← white → green
    var intensity = Math.min(Math.abs(tau), 1);
    var alpha = Math.round(intensity * 0.35 * 100) / 100;
    if (tau >= 0) return 'rgba(46, 125, 50, ' + alpha + ')';   // green
    return 'rgba(198, 40, 40, ' + alpha + ')';                  // red
}

// ── Analytics: LOO influence ──
// For each entity (annotator or data point), remove its votes, replay ratings,
// and compute: Δτ (change in avg solo-vs-consensus τ), Σ|Δrating|, rank changes.
// Reusable for both annotators (groupKey = "username") and data points (groupKey = "input_id").

function computeLOOInfluence(votes, groupKey, cfg) {
    // Baseline: consensus ratings and per-entity solo τ averaged
    var baseline = replayRatings(votes, cfg);
    var baselineRatings = ratingsMap(baseline);
    var baselineRanks = ranksMap(baseline);
    var baselineSoloTaus = computeSoloTau(votes, "username", baselineRatings, cfg);
    var baselineAvgTau = avgOfValues(baselineSoloTaus);

    // Collect unique entities
    var entities = {};
    votes.forEach(function (v) { entities[v[groupKey]] = true; });

    var result = {}; // entity → {deltaTau, sumAbsDeltaRating, rankChanges}
    Object.keys(entities).forEach(function (entity) {
        var remaining = votes.filter(function (v) { return v[groupKey] !== entity; });
        if (remaining.length === 0) return;

        // Replay consensus ratings without this entity
        var loo = replayRatings(remaining, cfg);
        var looRatings = ratingsMap(loo);
        var looRanks = ranksMap(loo);

        // Δτ: recompute per-annotator solo τ vs new consensus, average, compare to baseline
        var looSoloTaus = computeSoloTau(remaining, "username", looRatings, cfg);
        var looAvgTau = avgOfValues(looSoloTaus);
        var deltaTau = (baselineAvgTau !== null && looAvgTau !== null)
            ? Math.round((looAvgTau - baselineAvgTau) * 1000) / 1000
            : null;

        // Σ|ΔRating|: sum of absolute rating changes across all models
        var sumAbsDelta = 0;
        Object.keys(baselineRatings).forEach(function (m) {
            if (m in looRatings) sumAbsDelta += Math.abs(looRatings[m] - baselineRatings[m]);
        });
        sumAbsDelta = Math.round(sumAbsDelta * 10) / 10;

        // Rank changes: number of models whose rank position changed
        var rankChanges = 0;
        Object.keys(baselineRanks).forEach(function (m) {
            if (m in looRanks && looRanks[m] !== baselineRanks[m]) rankChanges++;
        });

        result[entity] = { deltaTau: deltaTau, sumAbsDeltaRating: sumAbsDelta, rankChanges: rankChanges };
    });
    return result;
}

function avgOfValues(obj) {
    var vals = Object.keys(obj).map(function (k) { return obj[k]; });
    if (vals.length === 0) return null;
    return vals.reduce(function (a, b) { return a + b; }, 0) / vals.length;
}

// ── Analytics: annotator stats ──
// Per annotator: count of votes, and majority agreement across comparison groups with >= 2 raters.

// ── Analytics: entity stats ──
// Per annotator: vote count. Per item: vote count + per-group vote breakdown (for expansion).

function computeAnnotatorStats(votes) {
    var stats = {};
    votes.forEach(function (v) {
        if (!stats[v.username]) stats[v.username] = 0;
        stats[v.username]++;
    });
    return Object.keys(stats).map(function (username) {
        return { username: username, totalVotes: stats[username] };
    }).sort(function (a, b) { return b.totalVotes - a.totalVotes; });
}

// ── Analytics: per-vote timing ──
// rawVotes is the unmodified array from /arena/votes (includes skip rows and duration_ms).
// We compute timing PER MATCH (not per criterion row):
//   TTFV = duration_ms of the chronologically-first criterion vote for that match
//          (= duration_ms of the first row encountered for that match_id, since the
//           backend returns rows ORDER BY id)
//   TTPM = sum of duration_ms across every row for that match — wall time of engagement
// Legacy rows with NULL duration_ms contribute nothing; entire-NULL matches yield NULL.

function computeMatchTimings(rawVotes) {
    var byMatch = {};
    rawVotes.forEach(function (v) {
        var m = byMatch[v.match_id];
        if (!m) {
            m = { ttfv: null, ttpm: 0, hasTpm: false, username: v.username, input_id: v.input_id };
            byMatch[v.match_id] = m;
            if (typeof v.duration_ms === "number") m.ttfv = v.duration_ms;
        }
        if (typeof v.duration_ms === "number") {
            m.ttpm += v.duration_ms;
            m.hasTpm = true;
        }
    });
    Object.keys(byMatch).forEach(function (k) {
        if (!byMatch[k].hasTpm) byMatch[k].ttpm = null;
    });
    return byMatch;
}

function _medianMs(arr) {
    var v = arr.filter(function (x) { return typeof x === "number" && !isNaN(x); });
    if (v.length === 0) return null;
    v.sort(function (a, b) { return a - b; });
    var n = v.length;
    return n % 2 === 0 ? (v[n/2 - 1] + v[n/2]) / 2 : v[(n-1)/2];
}

function formatDurationMs(ms) {
    if (ms == null) return "—";
    if (ms < 60000) return (ms / 1000).toFixed(1) + "s";
    var totalSec = Math.round(ms / 1000);
    var m = Math.floor(totalSec / 60);
    var s = totalSec % 60;
    return m + "m " + s + "s";
}

// Aggregate per-match timings → per-entity medians.
function aggregateTimingsByEntity(matchTimings, getKey) {
    var bucket = {};
    Object.keys(matchTimings).forEach(function (mid) {
        var m = matchTimings[mid];
        var k = getKey(m);
        if (!bucket[k]) bucket[k] = { ttfv: [], ttpm: [] };
        if (m.ttfv != null) bucket[k].ttfv.push(m.ttfv);
        if (m.ttpm != null) bucket[k].ttpm.push(m.ttpm);
    });
    var result = {};
    Object.keys(bucket).forEach(function (k) {
        result[k] = { medianTTFV: _medianMs(bucket[k].ttfv), medianTTPM: _medianMs(bucket[k].ttpm) };
    });
    return result;
}

function computeItemStats(votes) {
    // Group by (input_id, canonicalPair, criterion) for per-group vote breakdown
    var groups = {};
    votes.forEach(function (v) {
        var key = v.input_id + "|" + v.canonicalPair + "|" + v.criterion;
        if (!groups[key]) groups[key] = { item_id: v.input_id, pair: v.canonicalPair, criterion: v.criterion, votes: [] };
        groups[key].votes.push(v);
    });

    // Aggregate by item
    var byItem = {};
    Object.keys(groups).forEach(function (key) {
        var g = groups[key];
        // Deduplicate by username within each group
        var byUser = {};
        g.votes.forEach(function (v) { byUser[v.username] = v; });
        var dedupVotes = Object.keys(byUser).map(function (u) { return byUser[u]; });

        if (!byItem[g.item_id]) byItem[g.item_id] = { evalCount: 0, groups: [] };
        byItem[g.item_id].evalCount += dedupVotes.length;
        byItem[g.item_id].groups.push({
            pair: g.pair,
            criterion: g.criterion,
            votes: dedupVotes.map(function (v) { return { username: v.username, direction: v.direction, match_id: v.match_id }; }),
        });
    });

    return Object.keys(byItem).map(function (item_id) {
        var item = byItem[item_id];
        return { itemId: item_id, evalCount: item.evalCount, groups: item.groups };
    }).sort(function (a, b) { return b.evalCount - a.evalCount; });
}

// ── Analytics: composable filter pipeline ──
// Filters compose sequentially: exclude users → exclude items → criterion → labels.

function applyAnalyticsFilters(votes, filters) {
    var filtered = votes;

    // 1. Exclude annotators
    if (filters.excludedUsers.size > 0) {
        filtered = filtered.filter(function (v) {
            return !filters.excludedUsers.has(v.username);
        });
    }

    // 2. Exclude manually unchecked items
    if (filters.excludedItems.size > 0) {
        filtered = filtered.filter(function (v) {
            return !filters.excludedItems.has(v.input_id);
        });
    }

    // 3. Criterion filter
    if (filters.criterion) {
        filtered = filtered.filter(function (v) {
            return v.criterion === filters.criterion;
        });
    }

    // 4. Label filter — keep only votes for items with at least one matching label
    if (filters.labels && filters.labels.size > 0) {
        filtered = filtered.filter(function (v) {
            var itemLabels = window.itemLabelMap[v.input_id];
            if (!itemLabels) return false;
            var match = false;
            filters.labels.forEach(function (l) { if (itemLabels.has(l)) match = true; });
            return match;
        });
    }

    return filtered;
}

// ── Analytics: client-side Glicko-2 replay (one vote = one period) ──
// Mirrors Python arena_store._glicko2_update / _apply_match. Each model's state is
// [rating, rd, volatility] on the display scale (rating around 1500). See
// http://www.glicko.net/glicko/glicko2.pdf.

var SCORE_MAP = { a_strong: 1.0, a: 0.75, tie: 0.5, b: 0.25, b_strong: 0.0 };
var GLICKO2_SCALE = 400.0 / Math.log(10);  // ≈ 173.7178 (display units per internal unit)
// Defaults mirror Python DEFAULT_* constants in arena_store.py. Manifest values override.
var DEFAULT_CFG = { initial: 1500, initialRd: 350, initialVol: 0.02, tau: 0.5 };

function readRatingConfig(manifest) {
    var cfg = (manifest && manifest.rating_config) || {};
    return {
        initial: cfg.initial_rating != null ? cfg.initial_rating : DEFAULT_CFG.initial,
        initialRd: cfg.initial_rd != null ? cfg.initial_rd : DEFAULT_CFG.initialRd,
        initialVol: cfg.initial_volatility != null ? cfg.initial_volatility : DEFAULT_CFG.initialVol,
        tau: cfg.tau != null ? cfg.tau : DEFAULT_CFG.tau,
    };
}

function glicko2Update(state, oppState, score, tau) {
    // state, oppState: [rating, rd, vol] on display scale. score in [0, 1] for *this* player.
    // Returns new [rating, rd, vol] on display scale.
    var r = state[0], rd = state[1], vol = state[2];
    var oppR = oppState[0], oppRd = oppState[1];

    var mu = (r - 1500) / GLICKO2_SCALE;
    var phi = rd / GLICKO2_SCALE;
    var oppMu = (oppR - 1500) / GLICKO2_SCALE;
    var oppPhi = oppRd / GLICKO2_SCALE;

    var g = 1.0 / Math.sqrt(1.0 + 3.0 * oppPhi * oppPhi / (Math.PI * Math.PI));
    var E = 1.0 / (1.0 + Math.exp(-g * (mu - oppMu)));
    var E1mE = Math.max(E * (1.0 - E), 1e-12);
    var v = 1.0 / (g * g * E1mE);
    var delta = v * g * (score - E);

    // Illinois root-find for new volatility (Glickman 2013 §5)
    var a = Math.log(vol * vol);
    var f = function (x) {
        var ex = Math.exp(x);
        var num = ex * (delta * delta - phi * phi - v - ex);
        var den = 2.0 * Math.pow(phi * phi + v + ex, 2);
        return num / den - (x - a) / (tau * tau);
    };
    var A = a, B;
    if (delta * delta > phi * phi + v) {
        B = Math.log(delta * delta - phi * phi - v);
    } else {
        var k = 1;
        while (f(a - k * tau) < 0) k++;
        B = a - k * tau;
    }
    var fA = f(A), fB = f(B);
    for (var iter = 0; iter < 100 && Math.abs(B - A) > 1e-6; iter++) {
        var C = A + (A - B) * fA / (fB - fA);
        var fC = f(C);
        if (fC * fB <= 0) { A = B; fA = fB; }
        else { fA /= 2.0; }
        B = C; fB = fC;
    }
    var newVol = Math.exp(A / 2.0);

    var phiStar = Math.sqrt(phi * phi + newVol * newVol);
    var newPhi = 1.0 / Math.sqrt(1.0 / (phiStar * phiStar) + 1.0 / v);
    var newMu = mu + newPhi * newPhi * g * (score - E);

    return [1500 + GLICKO2_SCALE * newMu, GLICKO2_SCALE * newPhi, newVol];
}

function applyMatchJS(stateA, stateB, winner, tau) {
    var sA, sB;
    if (winner === "both_bad") { sA = 0.0; sB = 0.0; }
    else { sA = SCORE_MAP[winner] != null ? SCORE_MAP[winner] : 0.5; sB = 1.0 - sA; }
    return [glicko2Update(stateA, stateB, sA, tau), glicko2Update(stateB, stateA, sB, tau)];
}

// Replay a stream of votes under Glicko-2. State per model is [rating, rd, volatility].
// One vote = one period; history records one point per vote so charts render smoothly.
function replayRatings(votes, cfg) {
    cfg = cfg || DEFAULT_CFG;
    var state = {};    // model → [rating, rd, vol]
    var stats = {};    // model → {matches, wins, losses, ties}
    var history = {};  // model → [{match, rating, rd}]
    var matchNum = {};

    votes.forEach(function (v) {
        var ma = v.model_a, mb = v.model_b, w = v.winner;
        [ma, mb].forEach(function (m) {
            if (!state[m]) {
                state[m] = [cfg.initial, cfg.initialRd, cfg.initialVol];
                stats[m] = { matches: 0, wins: 0, losses: 0, ties: 0 };
                history[m] = [{ match: 0, rating: cfg.initial, rd: cfg.initialRd }];
                matchNum[m] = 0;
            }
        });

        if (w === "skip") return;

        var res = applyMatchJS(state[ma], state[mb], w, cfg.tau);
        state[ma] = res[0];
        state[mb] = res[1];

        stats[ma].matches++; stats[mb].matches++;
        matchNum[ma]++; matchNum[mb]++;

        if (w === "both_bad") { stats[ma].losses++; stats[mb].losses++; }
        else if (w === "a" || w === "a_strong") { stats[ma].wins++; stats[mb].losses++; }
        else if (w === "b" || w === "b_strong") { stats[mb].wins++; stats[ma].losses++; }
        else { stats[ma].ties++; stats[mb].ties++; }

        history[ma].push({ match: matchNum[ma], rating: Math.round(state[ma][0] * 10) / 10, rd: Math.round(state[ma][1] * 10) / 10 });
        history[mb].push({ match: matchNum[mb], rating: Math.round(state[mb][0] * 10) / 10, rd: Math.round(state[mb][1] * 10) / 10 });
    });

    var models = Object.keys(state).sort(function (a, b) { return state[b][0] - state[a][0]; });
    var rankings = models.map(function (m, i) {
        return {
            rank: i + 1,
            model: m,
            rating: Math.round(state[m][0] * 10) / 10,
            rd: Math.round(state[m][1] * 10) / 10,
            matches: stats[m].matches,
            wins: stats[m].wins,
            losses: stats[m].losses,
            ties: stats[m].ties,
            winRate: stats[m].matches > 0 ? Math.round(stats[m].wins / stats[m].matches * 100 * 10) / 10 : 0,
        };
    });

    return { rankings: rankings, history: history };
}

// Probability-space aggregation of ratings across criteria. Ratings are logit-scaled;
// arithmetic mean over-weights extremes. Convert each rating to its implied win probability
// against an initial-rated reference, average those, convert back. Reduces to the
// arithmetic mean only when all per-criterion ratings are equal.
function aggregateRatingsProbSpace(ratings, initial) {
    if (!ratings || ratings.length === 0) return initial;
    var sum = 0;
    for (var i = 0; i < ratings.length; i++) {
        sum += 1.0 / (1.0 + Math.pow(10, (initial - ratings[i]) / 400.0));
    }
    var p = sum / ratings.length;
    p = Math.max(Math.min(p, 1.0 - 1e-9), 1e-9);
    return initial + 400.0 * Math.log10(p / (1.0 - p));
}

// Aggregate replay: per-criterion Glicko-2 replays combined via probability-space rating
// average and mean RD. Sums count fields. Behaves identically to replayRatings when there's
// only one criterion.
function replayRatingsAggregate(votes, cfg) {
    cfg = cfg || DEFAULT_CFG;
    var byCriterion = {};
    votes.forEach(function (v) {
        var c = v.criterion || "overall";
        if (!byCriterion[c]) byCriterion[c] = [];
        byCriterion[c].push(v);
    });

    var criteria = Object.keys(byCriterion);
    if (criteria.length <= 1) return replayRatings(votes, cfg);

    var perCriterion = {};
    criteria.forEach(function (c) { perCriterion[c] = replayRatings(byCriterion[c], cfg); });

    var allModels = new Set();
    criteria.forEach(function (c) { perCriterion[c].rankings.forEach(function (r) { allModels.add(r.model); }); });

    var agg = {};
    allModels.forEach(function (m) { agg[m] = { ratings: [], rds: [], matches: 0, wins: 0, losses: 0, ties: 0 }; });
    criteria.forEach(function (c) {
        perCriterion[c].rankings.forEach(function (r) {
            var a = agg[r.model];
            a.ratings.push(r.rating);
            a.rds.push(r.rd);
            a.matches += r.matches;
            a.wins += r.wins;
            a.losses += r.losses;
            a.ties += r.ties;
        });
    });

    var models = Array.from(allModels).sort(function (a, b) {
        return aggregateRatingsProbSpace(agg[b].ratings, cfg.initial)
             - aggregateRatingsProbSpace(agg[a].ratings, cfg.initial);
    });
    var rankings = models.map(function (m, i) {
        var a = agg[m];
        // Aggregate RD = arithmetic mean of per-criterion RDs. Mirrors the server
        // (arena_store.get_leaderboard). Conservative — independent-Gaussian mean would
        // give sqrt(Σφ²)/n, but criteria are correlated so mean-of-RDs is safer.
        var meanRd = a.rds.reduce(function (s, x) { return s + x; }, 0) / a.rds.length;
        return {
            rank: i + 1, model: m,
            rating: Math.round(aggregateRatingsProbSpace(a.ratings, cfg.initial) * 10) / 10,
            rd: Math.round(meanRd * 10) / 10,
            matches: a.matches, wins: a.wins, losses: a.losses, ties: a.ties,
            winRate: a.matches > 0 ? Math.round(a.wins / a.matches * 100 * 10) / 10 : 0,
        };
    });

    var history = {};
    allModels.forEach(function (m) {
        var critHistories = criteria.map(function (c) { return (perCriterion[c].history || {})[m]; }).filter(Boolean);
        if (critHistories.length === 0) return;
        var longest = critHistories.reduce(function (a, b) { return a.length >= b.length ? a : b; });
        history[m] = longest.map(function (pt, i) {
            var ratingsAtI = [], rdsAtI = [];
            critHistories.forEach(function (ch) {
                var p = ch[Math.min(i, ch.length - 1)] || pt;
                ratingsAtI.push(p.rating);
                rdsAtI.push(p.rd);
            });
            return {
                match: pt.match,
                rating: Math.round(aggregateRatingsProbSpace(ratingsAtI, cfg.initial) * 10) / 10,
                rd: Math.round(rdsAtI.reduce(function (s, x) { return s + x; }, 0) / rdsAtI.length * 10) / 10,
            };
        });
    });

    return { rankings: rankings, history: history };
}

// ── Analytics: master render ──

function renderAnalytics() {
    var allVotes = window.analyticsNormalized;
    if (!allVotes) return;

    var manifest = window.currentManifest || {};
    var cfg = readRatingConfig(manifest);

    // Criterion chips
    renderAnalyticsCriterionChips(manifest);
    renderAnalyticsLabelChips();

    // Apply filters
    // Two-stage filtering:
    // 1. "userFiltered" = after annotator + criterion exclusions (but before item exclusions)
    //    Used for LOO tau so item exclusions don't make excluded items show N/A,
    //    but annotator changes DO affect the per-item metrics.
    // 2. "filtered" = full pipeline (annotator + item + criterion + labels)
    //    Used for rating replay and the leaderboard.
    var userFiltered = applyAnalyticsFilters(allVotes, {
        excludedUsers: window.analyticsFilters.excludedUsers,
        excludedItems: new Set(),
        criterion: window.analyticsFilters.criterion,
        labels: window.analyticsFilters.labels,
    });
    var filtered = applyAnalyticsFilters(allVotes, window.analyticsFilters);

    // itemFiltered = item + criterion + label filters (no annotator exclusions) → for per-annotator metrics
    var itemFiltered = applyAnalyticsFilters(allVotes, {
        excludedUsers: new Set(),
        excludedItems: window.analyticsFilters.excludedItems,
        criterion: window.analyticsFilters.criterion,
        labels: window.analyticsFilters.labels,
    });

    // Entity lists from ALL votes (so excluded ones still appear as rows in the tables)
    var allAnnotatorStats = computeAnnotatorStats(allVotes);
    var allItemStats = computeItemStats(allVotes);

    // Consensus ratings from filtered votes — alignment is relative to the current selection,
    // so values shift when annotators/items are excluded. This is intentional: it answers
    // "how does this entity compare to the currently selected group?"
    var filteredReplay = replayRatingsAggregate(filtered, cfg);
    var unfilteredReplay = replayRatingsAggregate(allVotes, cfg);
    var consensusRatings = ratingsMap(filteredReplay);

    // Solo τ: each entity's solo ranking vs consensus (computed on ALL votes so excluded entities keep values)
    var annotatorSoloTau = computeSoloTau(allVotes, "username", consensusRatings, cfg);
    var itemSoloTau = computeSoloTau(allVotes, "input_id", consensusRatings, cfg);

    // LOO influence: per-annotator on itemFiltered, per-data-point on userFiltered
    var annotatorLOO = computeLOOInfluence(itemFiltered, "username", cfg);
    var itemLOO = computeLOOInfluence(userFiltered, "input_id", cfg);

    // Model count for "X of N" rank shift display
    var modelCount = Object.keys(consensusRatings).length;
    var activeAnnotatorCount = allAnnotatorStats.filter(function (s) {
        return !window.analyticsFilters.excludedUsers.has(s.username);
    }).length;

    // Per-vote timing — computed from raw (un-normalized) votes so skip rows are included.
    var rawVotes = window.analyticsVotes || [];
    var matchTimings = computeMatchTimings(rawVotes);
    var annotatorTimings = aggregateTimingsByEntity(matchTimings, function (m) { return m.username; });
    var itemTimings = aggregateTimingsByEntity(matchTimings, function (m) { return m.input_id; });
    // Summary medians: restrict to matches that appear in the current filter set.
    var filteredMatchIds = new Set();
    filtered.forEach(function (v) { filteredMatchIds.add(v.match_id); });
    var summaryTtfv = [], summaryTtpm = [];
    Object.keys(matchTimings).forEach(function (mid) {
        if (!filteredMatchIds.has(mid)) return;
        var t = matchTimings[mid];
        if (t.ttfv != null) summaryTtfv.push(t.ttfv);
        if (t.ttpm != null) summaryTtpm.push(t.ttpm);
    });
    var summaryTimings = { medianTTFV: _medianMs(summaryTtfv), medianTTPM: _medianMs(summaryTtpm) };

    // Render all sections
    renderAnalyticsSummary(filtered.length, allVotes.length, activeAnnotatorCount, summaryTimings);
    renderAnnotatorTable(allAnnotatorStats, annotatorSoloTau, annotatorLOO, modelCount, annotatorTimings);
    var annotatorOrder = allAnnotatorStats.map(function (s) { return s.username; });
    renderAgreementMatrix(computePairwiseTau(filtered, cfg, annotatorOrder));
    renderItemTable(allItemStats, itemSoloTau, itemLOO, modelCount, itemTimings);
    renderAnalyticsLeaderboard(filteredReplay, unfilteredReplay, filtered.length);
    createEloLineChart("analytics-elo-chart", "analyticsChart", filteredReplay.history);

    // Show filter bar only when criterion or label chips are visible
    var hasCriteria = (manifest.criteria || []).length > 1;
    var hasLabels = window.allLabels.length > 0;
    document.getElementById("analytics-filter-bar").style.display = (hasCriteria || hasLabels) ? "" : "none";

    // Active filters banner
    var f = window.analyticsFilters;
    var parts = [];
    if (f.excludedUsers.size > 0) parts.push(f.excludedUsers.size + " annotator" + (f.excludedUsers.size > 1 ? "s" : "") + " excluded");
    if (f.excludedItems.size > 0) parts.push(f.excludedItems.size + " data point" + (f.excludedItems.size > 1 ? "s" : "") + " excluded");
    if (f.criterion) parts.push("criterion: " + escapeHtml(f.criterion));
    if (f.labels && f.labels.size > 0) parts.push("labels: " + Array.from(f.labels).map(escapeHtml).join(", "));

    var banner = document.getElementById("analytics-active-banner");
    if (parts.length > 0) {
        banner.style.display = "";
        document.getElementById("analytics-active-text").innerHTML =
            "Filters active: " + parts.join(" &bull; ") +
            " &mdash; <strong>" + filtered.length + " of " + allVotes.length + " votes</strong>";
    } else {
        banner.style.display = "none";
    }
}

function renderAnalyticsCriterionChips(manifest) {
    var container = document.getElementById("analytics-criterion-chips");
    var group = document.getElementById("analytics-criterion-group");
    container.innerHTML = "";
    var criteria = manifest.criteria || [];
    if (criteria.length <= 1) {
        group.style.display = "none";
        return;
    }
    group.style.display = "";
    // "All" chip
    var allChip = document.createElement("span");
    allChip.className = "arena-criterion-chip" + (window.analyticsFilters.criterion === null ? " active" : "");
    allChip.textContent = "All";
    allChip.onclick = function () { window.analyticsFilters.criterion = null; renderAnalytics(); };
    container.appendChild(allChip);
    criteria.forEach(function (c) {
        var chip = document.createElement("span");
        chip.className = "arena-criterion-chip" + (window.analyticsFilters.criterion === c.name ? " active" : "");
        chip.textContent = c.name;
        chip.title = c.description || "";
        chip.onclick = function () { window.analyticsFilters.criterion = c.name; renderAnalytics(); };
        container.appendChild(chip);
    });
}

function renderAnalyticsLabelChips() {
    var container = document.getElementById("analytics-label-chips");
    var group = document.getElementById("analytics-label-group");
    container.innerHTML = "";
    if (window.allLabels.length === 0) {
        group.style.display = "none";
        return;
    }
    group.style.display = "";
    var selectedLabels = window.analyticsFilters.labels;
    // Multi-select: clicking toggles a label in the set. No selection = all items.
    window.allLabels.forEach(function (label) {
        var isSelected = selectedLabels && selectedLabels.has(label);
        var chip = document.createElement("span");
        chip.className = "arena-criterion-chip" + (isSelected ? " active" : "");
        chip.textContent = label;
        chip.onclick = function () {
            if (!window.analyticsFilters.labels) window.analyticsFilters.labels = new Set();
            if (window.analyticsFilters.labels.has(label)) {
                window.analyticsFilters.labels.delete(label);
                if (window.analyticsFilters.labels.size === 0) window.analyticsFilters.labels = null;
            } else {
                window.analyticsFilters.labels.add(label);
            }
            renderAnalytics();
        };
        container.appendChild(chip);
    });
}

function renderAnalyticsSummary(filteredCount, totalCount, annotatorCount, timings) {
    var el = document.getElementById("analytics-summary");
    var text = filteredCount < totalCount
        ? filteredCount + " of " + totalCount + " votes (filtered)"
        : totalCount + " votes";
    text += " from " + annotatorCount + " annotator" + (annotatorCount !== 1 ? "s" : "");
    if (timings && (timings.medianTTFV != null || timings.medianTTPM != null)) {
        text += " · median time to first vote " + formatDurationMs(timings.medianTTFV);
        text += " · median time per match " + formatDurationMs(timings.medianTTPM);
    }
    el.innerHTML = '<div class="arena-analytics-subtitle">' + text + '</div>';
}

function renderAnnotatorTable(stats, soloTau, loo, modelCount, timings) {
    if (window.analyticsAnnotatorTable) {
        window.analyticsAnnotatorTable.destroy();
        document.querySelector("#analytics-annotator-table tbody").innerHTML = "";
    }

    // Row layout: [0] checkbox, [1] username, [2] votes,
    //   [3] alignment (solo τ), [4] influence (LOO Σ|ΔRating|/vote), [5] rank shifts,
    //   [6] median time-to-first-vote (ms), [7] median time-per-match (ms)
    var rows = stats.map(function (s) {
        var excluded = window.analyticsFilters.excludedUsers.has(s.username);
        var checkbox = '<input type="checkbox"' + (excluded ? '' : ' checked') + ' />';
        var tau = soloTau[s.username] != null ? soloTau[s.username] : null;
        var l = loo[s.username] || {};
        var perVoteRating = (l.sumAbsDeltaRating != null && s.totalVotes > 0)
            ? Math.round(l.sumAbsDeltaRating / s.totalVotes * 10) / 10 : null;
        var t = (timings && timings[s.username]) || {};
        return [checkbox, s.username, s.totalVotes, tau, perVoteRating,
                l.rankChanges != null ? l.rankChanges : null,
                t.medianTTFV != null ? t.medianTTFV : null,
                t.medianTTPM != null ? t.medianTTPM : null];
    });
    window.analyticsAnnotatorTable = $("#analytics-annotator-table").DataTable({
        data: rows,
        paging: false,
        searching: false,
        info: false,
        layout: { topStart: null, topEnd: null, bottomStart: null, bottomEnd: null },
        order: [[2, "desc"]],
        columnDefs: [
            { targets: 0, width: "30px", orderable: false, className: "dt-center" },
            { targets: [3, 5, 6, 7], type: "nullable" },
            { targets: 4, type: "abs-nullable" },
        ],
        createdRow: function (row, data) {
            $("td:eq(0)", row).html(data[0]);
            $("td:eq(3)", row).html(formatTauCell(data[3]));
            $("td:eq(4)", row).html(formatInfluenceCell(data[4]));
            $("td:eq(5)", row).html(formatRankShiftsCell(data[5], modelCount));
            $("td:eq(6)", row).text(formatDurationMs(data[6]));
            $("td:eq(7)", row).text(formatDurationMs(data[7]));
            if (window.analyticsFilters.excludedUsers.has(data[1])) {
                $(row).addClass("arena-analytics-row-excluded");
            }
        },
    });
    // Update header select-all state
    var allIncluded = stats.every(function (s) { return !window.analyticsFilters.excludedUsers.has(s.username); });
    document.getElementById("analytics-annotator-select-all").checked = allIncluded;

    // Unbind previous handlers before rebinding (table is destroyed+rebuilt on each render)
    $("#analytics-annotator-table tbody").off("change");

    // Row checkbox toggle
    $("#analytics-annotator-table tbody").on("change", "input[type=checkbox]", function () {
        var data = window.analyticsAnnotatorTable.row($(this).closest("tr")).data();
        if (!data) return;
        var username = data[1];
        if (this.checked) {
            window.analyticsFilters.excludedUsers.delete(username);
        } else {
            window.analyticsFilters.excludedUsers.add(username);
        }
        renderAnalytics();
    });

    // Select-all toggle
    $("#analytics-annotator-select-all").off("change").on("change", function () {
        var checked = this.checked;
        if (checked) {
            window.analyticsFilters.excludedUsers.clear();
        } else {
            stats.forEach(function (s) { window.analyticsFilters.excludedUsers.add(s.username); });
        }
        renderAnalytics();
    });
}

function renderItemTable(stats, soloTau, loo, modelCount, timings) {
    if (window.analyticsItemTable) {
        window.analyticsItemTable.destroy();
        document.querySelector("#analytics-item-table tbody").innerHTML = "";
    }

    // Row layout: [0] checkbox, [1] data point ID, [2] votes,
    //   [3] alignment (solo τ), [4] consensus Δ (LOO Δτ), [5] influence (LOO Σ|ΔRating|/vote),
    //   [6] rank shifts, [7] groups JSON (hidden), [8] median TTFV ms, [9] median TTPM ms
    var rows = stats.map(function (item) {
        var excluded = window.analyticsFilters.excludedItems.has(item.itemId);
        var checkbox = '<input type="checkbox"' + (excluded ? '' : ' checked') + ' />';
        var tau = soloTau[item.itemId] != null ? soloTau[item.itemId] : null;
        var l = loo[item.itemId] || {};
        var perVoteRating = (l.sumAbsDeltaRating != null && item.evalCount > 0)
            ? Math.round(l.sumAbsDeltaRating / item.evalCount * 10) / 10 : null;
        var t = (timings && timings[item.itemId]) || {};
        return [checkbox, item.itemId, item.evalCount,
                tau,
                l.deltaTau != null ? l.deltaTau : null,
                perVoteRating,
                l.rankChanges != null ? l.rankChanges : null,
                JSON.stringify(item.groups),
                t.medianTTFV != null ? t.medianTTFV : null,
                t.medianTTPM != null ? t.medianTTPM : null];
    });

    window.analyticsItemTable = $("#analytics-item-table").DataTable({
        data: rows,
        paging: true,
        pageLength: 10,
        searching: false,
        info: true,
        order: [[4, "desc"]],
        columnDefs: [
            { targets: 0, width: "30px", orderable: false, className: "dt-center" },
            { targets: 7, visible: false },
            { targets: [3, 4, 6, 8, 9], type: "nullable" },
            { targets: 5, type: "abs-nullable" },
        ],
        createdRow: function (row, data) {
            $("td:eq(0)", row).html(data[0]);
            // Data point name + label tags
            var itemLabels = window.itemLabelMap[data[1]];
            if (itemLabels && itemLabels.size > 0) {
                var tags = "";
                itemLabels.forEach(function (l) { tags += ' <span class="arena-label-tag">' + escapeHtml(l) + '</span>'; });
                $("td:eq(1)", row).html(escapeHtml(data[1]) + tags);
            }
            $("td:eq(3)", row).html(formatTauCell(data[3]));
            if (data[4] !== null) {
                $("td:eq(4)", row).text((data[4] > 0 ? "+" : "") + data[4].toFixed(3));
            } else {
                $("td:eq(4)", row).html('<span class="arena-analytics-na">N/A</span>');
            }
            $("td:eq(5)", row).html(formatInfluenceCell(data[5]));
            $("td:eq(6)", row).html(formatRankShiftsCell(data[6], modelCount));
            // td:eq(7) is the first visible column after the hidden groups JSON — that's data[8] (TTFV).
            $("td:eq(7)", row).text(formatDurationMs(data[8]));
            $("td:eq(8)", row).text(formatDurationMs(data[9]));
            if (window.analyticsFilters.excludedItems.has(data[1])) {
                $(row).addClass("arena-analytics-row-excluded");
            }
        },
    });

    // Unbind previous handlers before rebinding (table is destroyed+rebuilt on each render)
    $("#analytics-item-table tbody").off("click change");

    // Click to expand: show per-group detail with review links
    $("#analytics-item-table tbody").on("click", "tr", function (e) {
        if (e.target.tagName === "INPUT" || e.target.tagName === "A") return;
        var tr = $(this);
        var row = window.analyticsItemTable.row(tr);
        if (row.child.isShown()) {
            row.child.hide();
            tr.removeClass("shown");
        } else {
            var groupsJson = row.data()[7];
            row.child(formatItemDetail(JSON.parse(groupsJson)));
            row.child.show();
            tr.addClass("shown");
        }
    });

    // Update header select-all state
    var allItemsIncluded = stats.every(function (s) { return !window.analyticsFilters.excludedItems.has(s.itemId); });
    document.getElementById("analytics-item-select-all").checked = allItemsIncluded;

    // Checkbox toggle: exclude/include items
    $("#analytics-item-table tbody").on("change", "input[type=checkbox]", function () {
        var data = window.analyticsItemTable.row($(this).closest("tr")).data();
        if (!data) return;
        var itemId = data[1];
        if (this.checked) {
            window.analyticsFilters.excludedItems.delete(itemId);
        } else {
            window.analyticsFilters.excludedItems.add(itemId);
        }
        renderAnalytics();
    });

    // Select-all toggle
    $("#analytics-item-select-all").off("change").on("change", function () {
        var checked = this.checked;
        if (checked) {
            window.analyticsFilters.excludedItems.clear();
        } else {
            stats.forEach(function (s) { window.analyticsFilters.excludedItems.add(s.itemId); });
        }
        renderAnalytics();
    });
}

function formatItemDetail(groups) {
    var arenaName = window.currentArena;
    var html = '<div style="padding:8px 16px;">';
    groups.forEach(function (g) {
        var pairModels = g.pair.split("|");
        var dirLabel = { left: pairModels[0], right: pairModels[1], draw: "Tie" };
        html += '<div style="margin-bottom:8px;">';
        html += '<strong>' + escapeHtml(pairModels[0]) + ' vs ' + escapeHtml(pairModels[1]) + '</strong>';
        if (g.criterion !== "overall") html += ' <span style="color:var(--muted);">(' + escapeHtml(g.criterion) + ')</span>';
        html += '<div style="margin-left:12px; font-size:0.88em;">';
        g.votes.forEach(function (v) {
            html += escapeHtml(v.username) + ': <strong>' + escapeHtml(dirLabel[v.direction]) + '</strong>';
            if (v.match_id) {
                html += ' <a href="/arena#name=' + encodeURIComponent(arenaName) + '&match=' + encodeURIComponent(v.match_id) +
                    '" target="_blank" style="font-size:0.85em;">Review</a>';
            }
            html += '<br>';
        });
        html += '</div></div>';
    });
    html += '</div>';
    return html;
}

function renderAnalyticsLeaderboard(filteredRatings, unfilteredRatings, voteCount) {
    if (window.analyticsEloTable) {
        window.analyticsEloTable.destroy();
        document.querySelector("#analytics-elo-table tbody").innerHTML = "";
    }

    var summary = "Using " + voteCount + " votes";
    document.getElementById("analytics-filter-summary").textContent = summary;

    // Build unfiltered lookups: model → rating, model → rank position
    var unfilteredRatingMap = {};
    var unfilteredRankMap = {};
    unfilteredRatings.rankings.forEach(function (r) {
        unfilteredRatingMap[r.model] = r.rating;
        unfilteredRankMap[r.model] = r.rank;
    });

    var rows = filteredRatings.rankings.map(function (r) {
        // Delta rating
        var unfilteredRating = unfilteredRatingMap[r.model] || r.rating;
        var deltaRating = Math.round((r.rating - unfilteredRating) * 10) / 10;
        var deltaRatingStr = deltaRating > 0 ? "+" + deltaRating : deltaRating === 0 ? "0" : String(deltaRating);

        // Delta rank position (negative = moved up, so we flip sign for display: +2 = "moved up 2 spots")
        var unfilteredRank = unfilteredRankMap[r.model] || r.rank;
        var deltaRank = unfilteredRank - r.rank;  // positive if rank improved (lower number = better)
        var deltaRankStr = deltaRank > 0 ? "+" + deltaRank : deltaRank === 0 ? "0" : String(deltaRank);

        return [r.rank, r.model, r.rating, deltaRatingStr, deltaRankStr,
                r.matches, r.wins, r.losses, r.ties, r.winRate + "%"];
    });

    window.analyticsEloTable = $("#analytics-elo-table").DataTable({
        data: rows,
        paging: false,
        searching: false,
        info: false,
        ordering: false,
        createdRow: function (row, data) {
            // Color Δ Rating (col 3) and Δ Rank (col 4) green/red based on sign
            [3, 4].forEach(function (col) {
                var val = parseFloat(data[col]);
                if (val > 0) $("td:eq(" + col + ")", row).addClass("arena-delta-pos");
                else if (val < 0) $("td:eq(" + col + ")", row).addClass("arena-delta-neg");
            });
        },
    });
}


