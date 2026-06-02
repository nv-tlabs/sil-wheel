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

    // Keyboard shortcuts
    document.addEventListener("keydown", function (e) {
        if (!window.votingActive || !window.currentMatch) return;
        // Ignore if user is typing in an input/textarea
        if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

        switch (e.key) {
            case "1": case "a": case "A": submitVote("a_strong"); break;
            case "2": case "q": case "Q": submitVote("a"); break;
            case "3": case "t": case "T": submitVote("tie"); break;
            case "4": case "p": case "P": submitVote("b"); break;
            case "5": case "b": case "B": submitVote("b_strong"); break;
            case "0": submitVote("both_bad"); break;
            case "s": case "S": skipMatch(); break;
        }
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

// ── Arena list ──

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

function renderArenaList(filter) {
    var container = document.getElementById("arena-list");
    container.innerHTML = "";
    var term = (filter || "").toLowerCase();

    window.arenas.forEach(function (a) {
        if (term && a.display_name.toLowerCase().indexOf(term) === -1 && a.name.toLowerCase().indexOf(term) === -1) {
            return;
        }
        var card = document.createElement("div");
        card.className = "arena-card" + (window.currentArena === a.name ? " active" : "");
        card.onclick = function () { selectArena(a.name); };
        var badge = (!a.published) ? '<span class="arena-draft-badge">Draft</span>' : '';
        card.innerHTML =
            '<div class="arena-card-name">' + escapeHtml(a.display_name) + badge + '</div>' +
            '<div class="arena-card-desc">' + escapeHtml(a.description) + '</div>' +
            '<div class="arena-card-meta">' + a.num_models + ' models · ' + a.total_votes + ' votes</div>';
        container.appendChild(card);
    });
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

function selectArena(name) {
    window.currentArena = name;
    window.location.hash = "name=" + encodeURIComponent(name);
    renderArenaList(document.getElementById("arena-search").value);

    document.getElementById("arena-empty").style.display = "none";
    document.getElementById("arena-leaderboard-view").style.display = "";
    document.getElementById("arena-annotation-view").style.display = "none";

    // Update navbar title
    document.getElementById("arena-title").textContent = "SIL-Wheel Arena";

    // Fetch manifest + leaderboard + history + confidence intervals in parallel
    var q = encodeURIComponent(name);
    Promise.all([
        fetch("/arena/manifest?name=" + q).then(function (r) { return r.json(); }),
        fetch("/arena/leaderboard?name=" + q).then(function (r) { return r.json(); }),
        fetch("/arena/history?name=" + q + "&limit=100").then(function (r) { return r.json(); }),
        fetch("/arena/elo_confidence?name=" + q).then(function (r) { return r.json(); }),
    ]).then(function (results) {
        window.currentManifest = results[0];
        renderLeaderboard(results[0], results[1], results[2], results[3]);
    }).catch(function (e) { console.error("Failed to load arena:", e); });
}

function renderLeaderboard(manifest, leaderboardData, historyData, ciData) {
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

    ciData = ciData || {};

    // ELO table — models with < N matches (N = #models) shown as unranked/gray
    if (window.eloTable) {
        window.eloTable.destroy();
        document.querySelector("#arena-elo-table tbody").innerHTML = "";
    }
    var numModels = leaderboardData.rankings.length;
    var ranked = leaderboardData.rankings.filter(function (r) { return r.matches >= numModels; });
    var provisional = leaderboardData.rankings.filter(function (r) { return r.matches < numModels; });

    function fmtRating(r) {
        var ci = ciData[r.model];
        if (ci) {
            var pm = Math.round((ci.ci_high - ci.ci_low) / 2);
            return r.rating + ' <span class="arena-ci" title="95% CI: ' + ci.ci_low + ' – ' + ci.ci_high + '">\u00B1' + pm + '</span>';
        }
        return String(r.rating);
    }

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
            // Allow HTML in the rating column
            $("td:eq(2)", row).html(data[2]);
        },
    });

    // History table — compact, last 5 with pagination, clickable rows
    if (window.historyTable) {
        window.historyTable.destroy();
        document.querySelector("#arena-history-table tbody").innerHTML = "";
    }
    var hrows = (historyData || []).map(function (h) {
        var d = new Date(h.created_at * 1000);
        var ts = d.getFullYear() + "-" +
            String(d.getMonth() + 1).padStart(2, "0") + "-" +
            String(d.getDate()).padStart(2, "0") + " " +
            String(d.getHours()).padStart(2, "0") + ":" +
            String(d.getMinutes()).padStart(2, "0") + ":" +
            String(d.getSeconds()).padStart(2, "0");
        var winLabel = h.winner === "a_strong" ? "A++" :
                       h.winner === "a" ? "A+" :
                       h.winner === "b_strong" ? "B++" :
                       h.winner === "b" ? "B+" :
                       h.winner === "both_bad" ? "Both Bad" :
                       h.winner === "skip" ? "Skip" : "Tie";
        // columns: 0=vlm_tag, 1=ts, 2=input_id, 3=model_a, 4=model_b, 5=winLabel, 6=username, 7=match_id, 8=reasoning
        var isVLM = h.username && h.username.indexOf("vlm_judge:") === 0;
        var vlmTag = isVLM ? '<span style="font-size:0.7em;background:#6f42c1;color:#fff;padding:1px 4px;border-radius:2px;">VLM</span>' : '';
        return [vlmTag, ts, h.input_id, h.model_a, h.model_b, winLabel, h.username, h.match_id, h.reasoning || ""];
    });
    window.historyTable = $("#arena-history-table").DataTable({
        data: hrows,
        pageLength: 5,
        lengthMenu: [5, 10, 25, 50],
        paging: true,
        searching: false,
        info: true,
        order: [[1, "desc"]],
        language: { info: "Showing _START_-_END_ of _TOTAL_ votes" },
        columnDefs: [
            { targets: [7, 8], visible: false },
            { targets: 0, width: "30px", orderable: false, className: "dt-center" },
        ],
        createdRow: function (row, data) {
            // Allow HTML in the VLM tag column
            $("td:eq(0)", row).html(data[0]);
        },
    });
    // Clickable rows to review past matches
    $("#arena-history-table tbody").on("click", "tr", function () {
        var data = window.historyTable.row(this).data();
        if (data) reviewMatch(data[7], data[5], data[6], data[8]);
    });

    // ELO convergence chart
    renderEloChart(manifest.name);
}

function renderEloChart(arenaName) {
    fetch("/arena/elo_history?name=" + encodeURIComponent(arenaName))
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (window.eloChart) { window.eloChart.destroy(); }

            var models = Object.keys(data);
            if (models.length === 0) {
                document.getElementById("arena-elo-chart-section").style.display = "none";
                return;
            }
            document.getElementById("arena-elo-chart-section").style.display = "";

            var colors = ["#007bff", "#6f42c1", "#76B900", "#fd7e14", "#e83e8c", "#20c997", "#6610f2", "#dc3545"];
            var datasets = models.map(function (model, i) {
                var pts = data[model];
                return {
                    label: model,
                    data: pts.map(function (p) { return { x: p.match, y: p.rating }; }),
                    borderColor: colors[i % colors.length],
                    backgroundColor: "transparent",
                    borderWidth: 2,
                    pointRadius: pts.length > 50 ? 0 : 3,
                    tension: 0.1,
                };
            });

            var ctx = document.getElementById("arena-elo-chart").getContext("2d");
            // Vertical crosshair line at hover position (not available as a built-in Chart.js option)
            var crosshairPlugin = {
                id: "crosshair",
                afterDraw: function (chart) {
                    if (chart.tooltip && chart.tooltip.opacity > 0) {
                        var x = chart.tooltip.caretX;
                        var area = chart.chartArea;
                        var c = chart.ctx;
                        c.save();
                        c.beginPath();
                        c.moveTo(x, area.top);
                        c.lineTo(x, area.bottom);
                        c.lineWidth = 1;
                        c.strokeStyle = "rgba(0,0,0,0.2)";
                        c.stroke();
                        c.restore();
                    }
                },
            };

            window.eloChart = new Chart(ctx, {
                type: "line",
                data: { datasets: datasets },
                plugins: [crosshairPlugin],
                options: {
                    responsive: true,
                    animation: false,
                    interaction: { mode: "index", intersect: false },
                    scales: {
                        x: { type: "linear", title: { display: true, text: "Vote #" } },
                        y: { title: { display: true, text: "ELO Rating" } },
                    },
                    plugins: {
                        tooltip: {
                            mode: "index",
                            intersect: false,
                            itemSort: function (a, b) { return b.raw.y - a.raw.y; },
                        },
                        legend: { position: "top" },
                    },
                },
            });
        })
        .catch(function (e) { console.error("Failed to load ELO history:", e); });
}

// ── Annotation flow ──

function showLeaderboardView() {
    document.getElementById("arena-leaderboard-view").style.display = "";
    document.getElementById("arena-annotation-view").style.display = "none";
    window.votingActive = false;
    window.reviewIndex = -1;
    window.reviewHistory = [];
    // Refresh leaderboard
    selectArena(window.currentArena);
}

function exportVotes() {
    if (!window.currentArena) return;
    window.location.href = "/arena/export?name=" + encodeURIComponent(window.currentArena);
}

function startEvaluating() {
    window.matchCount = 0;
    document.getElementById("arena-leaderboard-view").style.display = "none";
    document.getElementById("arena-annotation-view").style.display = "";
    nextMatch();
}

function nextMatch() {
    document.getElementById("arena-vote-row").style.display = "";
    document.getElementById("arena-skip-row").style.display = "";
    document.getElementById("arena-reveal").style.display = "none";
    // Reset reasoning field
    var reasoningEl = document.getElementById("arena-reasoning");
    reasoningEl.value = "";
    reasoningEl.readOnly = false;
    reasoningEl.placeholder = "Why did you pick this one?";
    document.getElementById("arena-reasoning-badge").style.display = "none";
    document.querySelector(".arena-reasoning-optional").style.display = "";
    setVoteButtonsEnabled(true);
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
        })
        .catch(function (e) { hideLoading(); console.error("Failed to get match:", e); });
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

    // Instructions
    var instrEl = document.getElementById("arena-instructions");
    instrEl.textContent = match.instructions || "";

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
            block.appendChild(makeClippedText(inp.content, "arena-input-json"));
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
            var img = document.createElement("img");
            img.src = inp.url;
            block.appendChild(img);
        }
        inputsEl.appendChild(block);
    });

    // Sync input videos together
    syncVideoGroup("#arena-inputs video");

    // Outputs — render each output in the array
    renderOutputs("arena-content-a", match.outputs_a);
    renderOutputs("arena-content-b", match.outputs_b);
    syncVideoGroup(".arena-comparison video");
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
            var pre = document.createElement("pre");
            pre.className = "arena-json-output";
            try { pre.textContent = JSON.stringify(JSON.parse(output.content || "{}"), null, 2); }
            catch (e) { pre.textContent = output.content || ""; }
            el.appendChild(pre);
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
            var img = document.createElement("img");
            img.src = output.url;
            el.appendChild(img);
        }
    });
}

// ── Synchronized video playback ──

function syncVideoGroup(selector) {
    var vids = Array.from(document.querySelectorAll(selector));
    if (vids.length < 2) return;

    var syncing = false;

    function syncFrom(source) {
        if (syncing) return;
        syncing = true;
        vids.forEach(function (v) {
            if (v === source) return;
            if (!source.paused && v.paused) v.play();
            if (source.paused && !v.paused) v.pause();
            if (Math.abs(source.currentTime - v.currentTime) > 0.3) {
                v.currentTime = source.currentTime;
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

// ── Voting ──

function setVoteButtonsEnabled(enabled) {
    window.votingActive = enabled;
    document.querySelectorAll(".arena-vote-btn, .arena-skip-btn").forEach(function (b) {
        b.disabled = !enabled;
        if (enabled) b.classList.remove("voted");
    });
}

function highlightWinner(winner) {
    // Highlight output boxes
    var clsA = "arena-output", clsB = "arena-output";
    if (winner === "a" || winner === "a_strong") { clsA += " selected-a"; }
    else if (winner === "b" || winner === "b_strong") { clsB += " selected-b"; }
    else if (winner === "tie") { clsA += " selected-tie"; clsB += " selected-tie"; }
    document.getElementById("arena-output-a").className = clsA;
    document.getElementById("arena-output-b").className = clsB;

    // Highlight the chosen vote button and disable all
    var WINNER_TO_CLASS = {
        "a_strong": "pick-a-strong", "a": "pick-a", "tie": "tie",
        "b": "pick-b", "b_strong": "pick-b-strong", "both_bad": "both-bad",
    };
    var chosenClass = WINNER_TO_CLASS[winner] || "";
    document.querySelectorAll(".arena-vote-btn, .arena-skip-btn").forEach(function (btn) {
        btn.disabled = true;
        btn.classList.remove("voted");
        if (chosenClass && btn.classList.contains(chosenClass)) {
            btn.classList.add("voted");
        }
    });
    document.getElementById("arena-vote-row").style.display = "";
    document.getElementById("arena-skip-row").style.display = "";
}

function submitVote(winner) {
    if (!window.currentMatch || !window.votingActive) return;
    setVoteButtonsEnabled(false);

    var m = window.currentMatch;
    highlightWinner(winner);

    var reasoning = (document.getElementById("arena-reasoning").value || "").trim();
    var payload = "arena_submit_vote::" + m.arena_name + "::" + m.match_id + "::" + m.item_id + "::" + m.model_a + "::" + m.model_b + "::" + winner + "::" + reasoning;

    fetch("/", {
        method: "POST",
        headers: { "Content-Type": "text/plain" },
        body: payload,
    })
        .then(function (r) { return r.json(); })
        .then(function (result) {
            showReveal(result);
        })
        .catch(function (e) { console.error("Vote failed:", e); setVoteButtonsEnabled(true); });
}

function skipMatch() {
    if (!window.currentMatch || !window.votingActive) return;
    // Skip — no server call, just move to next match
    nextMatch();
}

function showReveal(result) {
    document.getElementById("arena-reveal").style.display = "";

    var nextBtn = document.getElementById("arena-next-btn");
    nextBtn.style.display = "";
    nextBtn.textContent = "Next Vote →";
    nextBtn.onclick = nextMatch;

    document.getElementById("reveal-name-a").textContent = result.model_a_name;
    document.getElementById("reveal-name-b").textContent = result.model_b_name;

    var eloA = document.getElementById("reveal-elo-a");
    var eloB = document.getElementById("reveal-elo-b");

    eloA.textContent = (result.elo_change_a >= 0 ? "+" : "") + result.elo_change_a + " (" + result.elo_a + ")";
    eloB.textContent = (result.elo_change_b >= 0 ? "+" : "") + result.elo_change_b + " (" + result.elo_b + ")";

    eloA.className = "arena-elo-change " + (result.elo_change_a > 0 ? "positive" : result.elo_change_a < 0 ? "negative" : "neutral");
    eloB.className = "arena-elo-change " + (result.elo_change_b > 0 ? "positive" : result.elo_change_b < 0 ? "negative" : "neutral");

    // Update output labels with real names
    document.querySelector("#arena-output-a .arena-output-label").textContent = result.model_a_name;
    document.querySelector("#arena-output-b .arena-output-label").textContent = result.model_b_name;
}

// ── URL hash ──

function applyHash() {
    var hash = window.location.hash.replace("#", "");
    var params = {};
    hash.split("&").forEach(function (part) {
        var kv = part.split("=");
        if (kv.length === 2) params[kv[0]] = decodeURIComponent(kv[1]);
    });
    if (params.name) {
        selectArena(params.name);
    }
}

window.addEventListener("hashchange", applyHash);

// ── Util ──

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

function escapeHtml(str) {
    var div = document.createElement("div");
    div.appendChild(document.createTextNode(str || ""));
    return div.innerHTML;
}

var TEXT_CLIP_LENGTH = 300;

function makeClippedText(content, className) {
    var container = document.createElement("div");
    container.className = "arena-clipped-container";

    var el = document.createElement(className === "arena-input-json" ? "pre" : "div");
    el.className = className;

    if (className === "arena-input-json") {
        try { content = JSON.stringify(JSON.parse(content || "{}"), null, 2); }
        catch (e) { /* keep as-is */ }
    }

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

// Map display labels back to internal winner codes for highlighting
var WIN_LABEL_TO_CODE = {"A++": "a_strong", "A+": "a", "Tie": "tie", "B+": "b", "B++": "b_strong", "Both Bad": "both_bad", "Skip": "skip"};

// Review navigation state
window.reviewHistory = [];  // array of [itemId, modelA, modelB, winLabel]
window.reviewIndex = -1;

function reviewMatch(matchId, winLabel, username, reasoning) {
    // Build review list from current history table data if not already navigating
    if (window.reviewIndex === -1 && window.historyTable) {
        window.reviewHistory = window.historyTable.rows({ order: "current" }).data().toArray();
        for (var i = 0; i < window.reviewHistory.length; i++) {
            if (window.reviewHistory[i][6] === matchId) {
                window.reviewIndex = i;
                break;
            }
        }
    }

    document.getElementById("arena-leaderboard-view").style.display = "none";
    document.getElementById("arena-annotation-view").style.display = "";
    var isVLM = username && username.indexOf("vlm_judge:") === 0;
    document.getElementById("arena-match-counter").textContent =
        "Reviewing vote" + (window.reviewHistory.length > 0 ? " " + (window.reviewIndex + 1) + "/" + window.reviewHistory.length : "")
        + (isVLM ? " — " + username : "");
    window.currentMatch = null;
    window.votingActive = false;

    // Show reasoning (readonly in review mode)
    var reasoningEl = document.getElementById("arena-reasoning");
    reasoningEl.value = reasoning || "";
    reasoningEl.readOnly = true;
    reasoningEl.placeholder = "";
    var badge = document.getElementById("arena-reasoning-badge");
    badge.style.display = isVLM ? "" : "none";
    document.querySelector(".arena-reasoning-optional").style.display = "none";

    fetch("/", {
        method: "POST",
        headers: { "Content-Type": "text/plain" },
        body: "arena_review_match::" + window.currentArena + "::" + matchId,
    })
        .then(function (r) { return r.json(); })
        .then(function (match) {
            renderMatch(match);

            // Switch to review mode: show reveal with model names
            document.getElementById("arena-reveal").style.display = "";

            document.querySelector("#arena-output-a .arena-output-label").textContent = match.model_a;
            document.querySelector("#arena-output-b .arena-output-label").textContent = match.model_b;
            document.getElementById("reveal-name-a").textContent = match.model_a;
            document.getElementById("reveal-name-b").textContent = match.model_b;
            document.getElementById("reveal-elo-a").textContent = "";
            document.getElementById("reveal-elo-b").textContent = "";
            document.getElementById("reveal-elo-a").className = "arena-elo-change";
            document.getElementById("reveal-elo-b").className = "arena-elo-change";

            // Highlight winner
            highlightWinner(WIN_LABEL_TO_CODE[winLabel] || "");

            // Set up Review Next button (hide on last row — header already has Back)
            var nextBtn = document.getElementById("arena-next-btn");
            if (window.reviewIndex >= 0 && window.reviewIndex < window.reviewHistory.length - 1) {
                nextBtn.style.display = "";
                nextBtn.textContent = "Review Next →";
                nextBtn.onclick = function () {
                    window.reviewIndex++;
                    var next = window.reviewHistory[window.reviewIndex];
                    reviewMatch(next[6], next[4], next[5], next[7]);
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

function arenaRunVLMJudge(arenaName) {
    var countInput = document.getElementById("arena-vlm-judge-count");
    var num = Math.max(1, Math.min(100, parseInt(countInput.value) || 10));
    countInput.value = num;

    if (!confirm("Run VLM judge for " + num + " matches on this arena?\n\nThis will submit automated votes that affect ELO ratings.")) {
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
