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

window._ds_entries = window._ds_entries || [];
window.dataStats = {
    msgEl: null,
    selectEl: null,
    gridEl: null,
    dataGridEl: null,
};

function dsOpt(label, value) {
    var o = document.createElement('option');
    o.textContent = label;
    o.value = value;
    return o;
}

// Zoom helpers (no nested function declarations)
function dataStatsCreateZoomState(imgEl, container) {
    return {
        imgEl: imgEl,
        container: container,
        zoomed: false,
        isPanning: false,
        moved: false,
        scale: 1.0,
        MIN_SCALE: 1.0,
        MAX_SCALE: 5.0,
        startX: 0,
        startY: 0,
        startScrollLeft: 0,
        startScrollTop: 0,
        onChange: null,
    };
}

function dataStatsZoomNotify(state) {
    if (typeof state.onChange === 'function') state.onChange(state.scale, state.zoomed);
}

function dataStatsZoomApplyScale(state, keepPoint) {
    if (!state.zoomed) return;
    var imgEl = state.imgEl;
    var container = state.container;
    var prevW = imgEl.offsetWidth;
    var prevH = imgEl.offsetHeight;
    var mouseX = 0, mouseY = 0;
    if (keepPoint && keepPoint.type === 'mouse') {
        var rect = container.getBoundingClientRect();
        mouseX = keepPoint.clientX - rect.left;
        mouseY = keepPoint.clientY - rect.top;
    } else {
        mouseX = container.clientWidth / 2;
        mouseY = container.clientHeight / 2;
    }
    var offsetX = container.scrollLeft + mouseX;
    var offsetY = container.scrollTop + mouseY;

    imgEl.style.width = (state.scale * 100) + '%';

    var newW = imgEl.offsetWidth;
    var newH = imgEl.offsetHeight;
    var fx = newW / Math.max(1, prevW);
    var fy = newH / Math.max(1, prevH);
    var nextScrollLeft = Math.max(0, offsetX * fx - mouseX);
    var nextScrollTop = Math.max(0, offsetY * fy - mouseY);
    container.scrollLeft = nextScrollLeft;
    container.scrollTop = nextScrollTop;
    dataStatsZoomNotify(state);
}

function dataStatsZoomSetZoom(state, z, opts) {
    var imgEl = state.imgEl;
    var container = state.container;
    state.zoomed = z;
    if (state.zoomed) {
        imgEl.classList.add('zoomed');
        if (state.scale <= state.MIN_SCALE) state.scale = 1.2;
        dataStatsZoomApplyScale(state, opts && opts.keepPoint);
    } else {
        imgEl.classList.remove('zoomed');
        imgEl.classList.remove('panning');
        imgEl.style.width = '';
        state.scale = 1.0;
        container.scrollLeft = 0;
        container.scrollTop = 0;
    }
    dataStatsZoomNotify(state);
}

function dataStatsZoomOnClick(e) {
    var state = e.currentTarget._dsZoomState;
    if (state.moved) { e.preventDefault(); state.moved = false; return; }
    dataStatsZoomSetZoom(state, !state.zoomed, { keepPoint: { type: 'mouse', clientX: e.clientX, clientY: e.clientY } });
}

function dataStatsZoomOnDragStart(e) { e.preventDefault(); }

function dataStatsZoomOnMouseDown(e) {
    var state = e.currentTarget._dsZoomState;
    if (!state.zoomed) return;
    state.isPanning = true;
    state.moved = false;
    state.startX = e.clientX;
    state.startY = e.clientY;
    state.startScrollLeft = state.container.scrollLeft;
    state.startScrollTop = state.container.scrollTop;
    state.imgEl.classList.add('panning');
}

function dataStatsZoomOnMouseMove(e) {
    var state = e.currentTarget._dsZoomState;
    if (!state.zoomed || !state.isPanning) return;
    var dx = e.clientX - state.startX;
    var dy = e.clientY - state.startY;
    state.container.scrollLeft = state.startScrollLeft - dx;
    state.container.scrollTop = state.startScrollTop - dy;
    state.moved = true;
}

function dataStatsZoomOnWheel(e) {
    var state = e.currentTarget._dsZoomState;
    e.preventDefault();
    var step = e.ctrlKey ? 1.03 : 1.10;
    if (e.deltaY < 0) {
        state.scale = Math.min(state.MAX_SCALE, (state.zoomed ? state.scale : 1.2) * step);
        if (!state.zoomed) dataStatsZoomSetZoom(state, true, { keepPoint: { type: 'mouse', clientX: e.clientX, clientY: e.clientY } });
        else dataStatsZoomApplyScale(state, { type: 'mouse', clientX: e.clientX, clientY: e.clientY });
    } else {
        state.scale = Math.max(state.MIN_SCALE, state.scale / step);
        if (state.zoomed && state.scale <= state.MIN_SCALE + 0.01) dataStatsZoomSetZoom(state, false);
        else if (state.zoomed) dataStatsZoomApplyScale(state, { type: 'mouse', clientX: e.clientX, clientY: e.clientY });
    }
}

function dataStatsZoomOnMouseUp(e) {
    var state = e.currentTarget._dsZoomState;
    if (!state.zoomed) return;
    state.isPanning = false;
    state.imgEl.classList.remove('panning');
}

function dataStatsZoomOnMouseLeave(e) {
    var state = e.currentTarget._dsZoomState;
    if (!state.zoomed) return;
    state.isPanning = false;
    state.imgEl.classList.remove('panning');
}

function dataStatsEnableZoom(imgEl, container) {
    var state = dataStatsCreateZoomState(imgEl, container);
    // attach state to elements for handler access
    imgEl._dsZoomState = state;
    container._dsZoomState = state;

    imgEl.addEventListener('click', dataStatsZoomOnClick);
    imgEl.addEventListener('dragstart', dataStatsZoomOnDragStart);
    container.addEventListener('mousedown', dataStatsZoomOnMouseDown);
    container.addEventListener('mousemove', dataStatsZoomOnMouseMove);
    container.addEventListener('wheel', dataStatsZoomOnWheel, { passive: false });
    container.addEventListener('mouseup', dataStatsZoomOnMouseUp);
    container.addEventListener('mouseleave', dataStatsZoomOnMouseLeave);

    return {
        zoomIn: function() {
            var step = 1.10;
            state.scale = Math.min(state.MAX_SCALE, (state.zoomed ? state.scale : 1.2) * step);
            if (!state.zoomed) dataStatsZoomSetZoom(state, true);
            else dataStatsZoomApplyScale(state);
        },
        zoomOut: function() {
            var step = 1.10;
            state.scale = Math.max(state.MIN_SCALE, state.scale / step);
            if (state.zoomed && state.scale <= state.MIN_SCALE + 0.01) dataStatsZoomSetZoom(state, false);
            else if (state.zoomed) dataStatsZoomApplyScale(state);
        },
        reset: function() { dataStatsZoomSetZoom(state, false); },
        getScale: function() { return state.scale; },
        setOnChange: function(fn) { state.onChange = fn; if (typeof state.onChange === 'function') state.onChange(state.scale, state.zoomed); },
    };
}

function dataStatsShowImage(img, viewer, ds, which) {
    if (which === 'vis') img.src = ds.visualization;
    else if (which === 'stats') img.src = ds.statistics;
    else img.src = ds.per_clip;
    img.classList.remove('zoomed', 'panning');
    img.style.width = '';
    viewer.scrollLeft = 0; viewer.scrollTop = 0;
}

function dataStatsMakePanel(ds) {
    var panel = document.createElement('div');
    panel.className = 'ds-panel';
    panel.dataset.ds = ds.dataset;

    var h3 = document.createElement('h3');
    var hlink = document.createElement('a');
    hlink.href = '/#page=0&data_source=' + encodeURIComponent(ds.dataset);
    hlink.textContent = ds.dataset;
    hlink.className = 'ds-link';
    h3.appendChild(hlink);

    var toolbar = document.createElement('div');
    toolbar.className = 'ds-toolbar';
    var btnVis = document.createElement('button'); btnVis.textContent = 'Visualization';
    var btnStats = document.createElement('button'); btnStats.textContent = 'Statistics';
    var btnPerClip = document.createElement('button'); btnPerClip.textContent = 'Per-clip';
    toolbar.appendChild(btnVis); toolbar.appendChild(btnStats); toolbar.appendChild(btnPerClip);

    var viewer = document.createElement('div');
    viewer.className = 'image-viewer';
    var img = document.createElement('img');
    img.alt = 'Dataset image';
    viewer.appendChild(img);

    var zoomAPI = dataStatsEnableZoom(img, viewer);

    btnVis.onclick = function() { dataStatsShowImage(img, viewer, ds, 'vis'); };
    btnStats.onclick = function() { dataStatsShowImage(img, viewer, ds, 'stats'); };
    btnPerClip.onclick = function() { dataStatsShowImage(img, viewer, ds, 'per'); };
    dataStatsShowImage(img, viewer, ds, 'vis');

    var summary = document.createElement('div');
    summary.className = 'summary';
    var loading = document.createElement('div');
    loading.className = 'loading';
    loading.textContent = 'Loading trajectory summary…';
    summary.appendChild(loading);

    if (ds.summary) {
        fetch(ds.summary)
            .then(function(r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
            .then(function(j) { dataStatsRenderSummary(j || {}, summary); })
            .catch(function() {
                summary.innerHTML = '';
                var e = document.createElement('div'); e.className = 'error';
                e.textContent = 'Failed to load trajectory summary';
                summary.appendChild(e);
            });
    }


    var zoomControls = document.createElement('div');
    zoomControls.className = 'zoom-controls';
    var zOut = document.createElement('button'); zOut.textContent = '−'; zOut.title = 'Zoom Out';
    var zIn = document.createElement('button'); zIn.textContent = '+'; zIn.title = 'Zoom In';
    var zReset = document.createElement('button'); zReset.textContent = 'Reset Zoom'; zReset.title = 'Reset Zoom';
    var zRatio = document.createElement('span'); zRatio.className = 'zoom-ratio'; zRatio.title = 'Zoom Level';
    zOut.onclick = function() { zoomAPI.zoomOut(); };
    zIn.onclick = function() { zoomAPI.zoomIn(); };
    zReset.onclick = function() { zoomAPI.reset(); };
    zoomAPI.setOnChange(function(scale) { zRatio.textContent = String(Math.round(scale * 100)) + '%'; });
    var spacer = document.createElement('div'); spacer.style.flex = '1 1 auto';
    toolbar.appendChild(spacer);
    zoomControls.appendChild(zOut);
    zoomControls.appendChild(zIn);
    zoomControls.appendChild(zReset);
    zoomControls.appendChild(zRatio);
    toolbar.appendChild(zoomControls);

    panel.appendChild(h3);
    panel.appendChild(summary);
    panel.appendChild(toolbar);
    panel.appendChild(viewer);
    return panel;
}

function dataStatsMakeDataPanel(ds) {
    if (!(ds.data_summary || (ds.data_pngs && ds.data_pngs.length) || ds.data_plot)) return null;
    var panel = document.createElement('div'); panel.className = 'ds-panel'; panel.dataset.ds = ds.dataset;
    var h3 = document.createElement('h3');
    var hlink = document.createElement('a'); hlink.href = '/#page=0&data_source=' + encodeURIComponent(ds.dataset);
    hlink.textContent = ds.dataset; hlink.className = 'ds-link'; h3.appendChild(hlink);
    panel.appendChild(h3);

    if (ds.data_summary) {
        var summary = document.createElement('div'); summary.className = 'summary';
        var loading = document.createElement('div'); loading.className = 'loading'; loading.textContent = 'Loading data statistics…'; summary.appendChild(loading);
        fetch(ds.data_summary)
            .then(function(r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
            .then(function(j) { summary.innerHTML = ''; dataStatsRenderDataSummary(j || {}, summary); })
            .catch(function() { summary.innerHTML = ''; var e = document.createElement('div'); e.className = 'error'; e.textContent = 'Failed to load data statistics'; summary.appendChild(e); });
        panel.appendChild(summary);
    }

    var list = (ds.data_pngs && ds.data_pngs.length) ? ds.data_pngs : (ds.data_plot ? [ds.data_plot] : []);
    if (list && list.length) {
        var imgs = document.createElement('div'); imgs.style.marginTop = '8px';
        var title = document.createElement('div'); title.className = 'summary-meta'; title.textContent = 'Plots'; imgs.appendChild(title);
        list.forEach(function(u) { var im = document.createElement('img'); im.src = u; im.alt = 'Data plot'; im.style.maxWidth = '100%'; im.style.marginBottom = '6px'; im.loading = 'lazy'; imgs.appendChild(im); });
        panel.appendChild(imgs);
    }
    return panel;
}

function dataStatsFmt(x) {
    return (typeof x === 'number' && isFinite(x)) ? x.toFixed(4) : '—';
}

function dataStatsBuildStatsTable(obj) {
    var table = document.createElement('table'); table.className = 'stats-table';
    var thead = document.createElement('thead');
    thead.innerHTML = '<tr><th>Feature</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Median</th></tr>';
    table.appendChild(thead);
    var tbody = document.createElement('tbody');
    var entries = Object.entries(obj || {});
    entries.forEach(function(entry) {
        var name = entry[0];
        var stats = entry[1];
        var tr = document.createElement('tr');
        var vals = [
            name,
            dataStatsFmt(stats && stats.mean),
            dataStatsFmt(stats && stats.std),
            dataStatsFmt(stats && stats.min),
            dataStatsFmt(stats && stats.max),
            dataStatsFmt(stats && stats.median),
        ];
        vals.forEach(function(v) { var td = document.createElement('td'); td.textContent = v; tr.appendChild(td); });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    return table;
}

function dataStatsRenderSummary(data, container) {
    container.innerHTML = '';

    var meta = document.createElement('div');
    meta.className = 'summary-meta';
    var dsName = data.dataset || '—';
    var nClips = (data.n_clips_sampled != null ? data.n_clips_sampled : '—');
    meta.textContent = 'Dataset: ' + dsName + ' • Clips: ' + nClips;
    container.appendChild(meta);

    if (data.features) {
        var det = document.createElement('details');
        det.open = true;
        var sum = document.createElement('summary'); sum.textContent = 'Features'; det.appendChild(sum);
        var block = document.createElement('div'); block.className = 'summary-block';
        block.appendChild(dataStatsBuildStatsTable(data.features));
        det.appendChild(block);
        container.appendChild(det);
    }
    if (data.per_clip_avg) {
        var det2 = document.createElement('details');
        var sum2 = document.createElement('summary'); sum2.textContent = 'Per-Clip Average'; det2.appendChild(sum2);
        var block2 = document.createElement('div'); block2.className = 'summary-block';
        block2.appendChild(dataStatsBuildStatsTable(data.per_clip_avg));
        det2.appendChild(block2);
        container.appendChild(det2);
    }
    if (data.per_clip_max) {
        var det3 = document.createElement('details');
        var sum3 = document.createElement('summary'); sum3.textContent = 'Per-Clip Max'; det3.appendChild(sum3);
        var block3 = document.createElement('div'); block3.className = 'summary-block';
        block3.appendChild(dataStatsBuildStatsTable(data.per_clip_max));
        det3.appendChild(block3);
        container.appendChild(det3);
    }

    var pct = data.percentiles || null;
    if (pct && Object.keys(pct).length) {
        var det4 = document.createElement('details');
        var sum4 = document.createElement('summary'); sum4.textContent = 'Percentiles'; det4.appendChild(sum4);
        var block4 = document.createElement('div'); block4.className = 'summary-block';

        var allKeys = new Set();
        Object.values(pct).forEach(function(obj) { Object.keys(obj).forEach(function(k) { allKeys.add(k); }); });
        var cols = Array.from(allKeys).sort(function(a,b) { return parseInt(a.slice(1)) - parseInt(b.slice(1)); });

        var table = document.createElement('table'); table.className = 'stats-table';
        var thead = document.createElement('thead');
        var headerCells = ['Feature'].concat(cols);
        thead.innerHTML = '<tr>' + headerCells.map(function(h) { return '<th>' + h + '</th>'; }).join('') + '</tr>';
        table.appendChild(thead);
        var tbody = document.createElement('tbody');
        Object.entries(pct).forEach(function(entry) {
            var name = entry[0]; var obj = entry[1];
            var tr = document.createElement('tr');
            var cells = [name].concat(cols.map(function(k) { return dataStatsFmt(obj[k]); }));
            cells.forEach(function(v) { var td = document.createElement('td'); td.textContent = v; tr.appendChild(td); });
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        block4.appendChild(table);
        det4.appendChild(block4);
        container.appendChild(det4);
    }
}

// Render only the core features table for Data Statistics (no per-clip or percentiles)
function dataStatsRenderDataSummary(data, container) {
    container.innerHTML = '';

    var meta = document.createElement('div');
    meta.className = 'summary-meta';
    var dsName = data.dataset || '—';
    var nClips = (data.n_clips_sampled != null ? data.n_clips_sampled : '—');
    meta.textContent = 'Dataset: ' + dsName + ' • Clips: ' + nClips;
    container.appendChild(meta);

    if (data.features) {
        var det = document.createElement('details');
        det.open = true;
        var sum = document.createElement('summary'); sum.textContent = 'Features'; det.appendChild(sum);
        var block = document.createElement('div'); block.className = 'summary-block';
        block.appendChild(dataStatsBuildStatsTable(data.features));
        det.appendChild(block);
        container.appendChild(det);
    }
}

function dataStatsRender(datasets) {
    var grid = window.dataStats.gridEl;
    grid.innerHTML = '';
    datasets.forEach(function(ds) { grid.appendChild(dataStatsMakePanel(ds)); });
    // Render data stats panels
    var grid2 = window.dataStats.dataGridEl;
    if (grid2) {
        grid2.innerHTML = '';
        datasets.forEach(function(ds) {
            var p = dataStatsMakeDataPanel(ds); if (p) grid2.appendChild(p);
        });
    }
}

function dataStatsSyncPanels(datasets) {
    var grid = window.dataStats.gridEl;
    var selectedNames = new Set(datasets.map(function(d) { return d.dataset; }));
    Array.from(grid.querySelectorAll('.ds-panel')).forEach(function(p) {
        var name = p && p.dataset ? p.dataset.ds : null;
        if (!selectedNames.has(name)) p.remove();
    });
    datasets.forEach(function(ds) {
        var existing = Array.from(grid.querySelectorAll('.ds-panel')).find(function(p) { return p.dataset && p.dataset.ds === ds.dataset; });
        if (!existing) grid.appendChild(dataStatsMakePanel(ds));
    });
    // Sync data grid (simple re-render)
    var grid2 = window.dataStats.dataGridEl;
    if (grid2) {
        grid2.innerHTML = '';
        datasets.forEach(function(ds) { var p = dataStatsMakeDataPanel(ds); if (p) grid2.appendChild(p); });
    }
}

function dataStatsSelected() {
    var sel = window.dataStats.selectEl;
    var vals = [];
    if (window.jQuery && jQuery.fn && jQuery.fn.select2) {
        vals = jQuery(sel).val() || [];
    } else {
        vals = Array.from(sel.selectedOptions).map(function(o) { return o.value; });
    }
    var entries = window._ds_entries || [];
    return entries.filter(function(e) { return vals.includes(e.dataset); });
}

function dsParseHash() {
    var h = String(window.location.hash || '');
    if (h.startsWith('#')) h = h.slice(1);
    var out = {};
    if (!h) return out;
    h.split('&').forEach(function(part) {
        if (!part) return;
        var kv = part.split('=');
        var k = decodeURIComponent(kv[0] || '');
        var v = decodeURIComponent(kv.slice(1).join('=') || '');
        if (k) out[k] = v;
    });
    return out;
}

function dsEncodeHash(obj) {
    var parts = [];
    Object.keys(obj).forEach(function(k) {
        var v = obj[k];
        if (v === null || v === undefined || v === '') return;
        parts.push(encodeURIComponent(k) + '=' + encodeURIComponent(String(v)));
    });
    return parts.join('&');
}

function dataStatsUpdateHashFromSelection() {
    var qs = dsParseHash();
    var selected = dataStatsSelected().map(function(e) { return e.dataset; });
    if (selected.length > 0) {
        qs.data_source = selected.join('||');
    } else {
        delete qs.data_source;
    }
    var h = dsEncodeHash(qs);
    window.location.hash = h ? ('#' + h) : '';
}

function render() {
    var sel = window.dataStats.selectEl;
    var qs = dsParseHash();
    var want = (qs.data_source || '')
        .split('||')
        .map(function(s) { return s.trim(); })
        .filter(Boolean);

    if (window.jQuery && jQuery.fn && jQuery.fn.select2) {
        jQuery(sel).val(want).trigger('change');
    } else {
        Array.from(sel.options).forEach(function(o) { o.selected = want.includes(o.value); });
        sel.dispatchEvent(new Event('change'));
    }

    // Ensure panels update even if no user interaction occurred
    dataStatsSyncPanels(dataStatsSelected());
    dataStatsUpdateDsCount();
}


function dataStatsEnableNativeToggle() {
    var sel = window.dataStats.selectEl;
    sel.addEventListener('mousedown', function(e) {
        if (e && e.target && e.target.tagName === 'OPTION') {
            e.preventDefault();
            e.target.selected = !e.target.selected;
            sel.dispatchEvent(new Event('change'));
        }
    });
}

function dataStatsSetAllDatasetsSelected(all) {
    var sel = window.dataStats.selectEl;
    var arr = (window._ds_entries || []).map(function(e) { return e.dataset; });
    if (window.jQuery && jQuery.fn && jQuery.fn.select2) {
        jQuery(sel).val(all ? arr : []).trigger('change');
    } else {
        Array.from(sel.options).forEach(function(o) { o.selected = !!all; });
        sel.dispatchEvent(new Event('change'));
    }
}

function dataStatsUpdateDsCount() {
    var dsCount = document.getElementById('ds-count');
    var total = (window._ds_entries || []).length;
    var n = dataStatsSelected().length;
    dsCount.textContent = total ? (n + '/' + total) : '';
}

function initDataStats() {
    window.dataStats.msgEl = document.getElementById('stats-msg');
    window.dataStats.selectEl = document.getElementById('dataset-select');
    window.dataStats.gridEl = document.getElementById('stats-grid');
    window.dataStats.dataGridEl = document.getElementById('data-stats-grid');

    var sel = window.dataStats.selectEl;
    sel.addEventListener('change', function() {
        dataStatsSyncPanels(dataStatsSelected());
        dataStatsUpdateDsCount();
        dataStatsUpdateHashFromSelection();
    });

  document.getElementById('select-all-ds').addEventListener('click', function() {
    dataStatsSetAllDatasetsSelected(true);
    dataStatsSyncPanels(dataStatsSelected());
    dataStatsUpdateDsCount();
    dataStatsUpdateHashFromSelection();
  });
  document.getElementById('clear-ds').addEventListener('click', function() {
    dataStatsSetAllDatasetsSelected(false);
    dataStatsSyncPanels(dataStatsSelected());
    dataStatsUpdateDsCount();
    dataStatsUpdateHashFromSelection();
  });

    dataStatsLoadDatasetsFromHash();

    document.getElementById('logout-btn').addEventListener('click', function() {
        var btn = document.getElementById('logout-btn');
        var prev = btn.textContent; btn.disabled = true; btn.textContent = 'Logging out…';
        fetch('/', { method: 'POST', body: 'logout::' })
            .catch(function() {})
            .then(function() { btn.disabled = false; btn.textContent = prev; window.location.replace('/login'); });
    });
}

window.onhashchange = function () {
    dataStatsLoadDatasetsFromHash();
}

document.addEventListener('DOMContentLoaded', function() {
    initDataStats();
    // React to hash navigation like other pages
    window.onhashchange = function () { dataStatsLoadDatasetsFromHash(); };
});

function dataStatsLoadDatasetsFromHash() {
    // Resolve elements even if initDataStats hasn't run yet
    var sel = window.dataStats.selectEl || document.getElementById('dataset-select');
    var msgEl = window.dataStats.msgEl || document.getElementById('stats-msg');
    var grid = window.dataStats.gridEl || document.getElementById('stats-grid');
    if (!sel || !msgEl) {
        return; // DOM not ready; DOMContentLoaded handler will call again
    }
    window.dataStats.selectEl = sel;
    window.dataStats.msgEl = msgEl;
    window.dataStats.gridEl = grid;
    var qs = dsParseHash();
    var url = '/data_stats_list';

    fetch(url)
        .then(function(r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then(function(j) {
            var arr = (j && j.datasets) || [];
            window._ds_entries = arr;
            sel.innerHTML = '';
            arr.forEach(function(e) { sel.appendChild(dsOpt(e.dataset, e.dataset)); });

            // Enhance select once if Select2 present
            var enhanced = false;
            if (window.jQuery && jQuery.fn && jQuery.fn.select2) {
                if (!jQuery(sel).data('select2')) {
                    jQuery(sel).select2({ placeholder: 'Choose datasets', width: 'resolve', closeOnSelect: false, templateResult: datasetTemplateResult, templateSelection: datasetTemplateResult });
                    jQuery(sel).on('select2:select select2:unselect', function() { sel.dispatchEvent(new Event('change')); });
                }
                enhanced = true;
            }
            if (!enhanced) dataStatsEnableNativeToggle();

            if (arr.length) {
                render();
            }
            msgEl.textContent = arr.length ? '' : 'No datasets found';
            dataStatsUpdateDsCount();
        })
        .catch(function() {
            msgEl.textContent = 'Failed to load datasets';
            msgEl.style.color = '#b91c1c';
        });
}

// Also trigger on initial parse if DOM is already loaded
if (document.readyState !== 'loading') {
    dataStatsLoadDatasetsFromHash();
} else {
    document.addEventListener('DOMContentLoaded', dataStatsLoadDatasetsFromHash);
}
