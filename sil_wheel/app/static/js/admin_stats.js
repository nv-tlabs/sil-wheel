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

function el(tag, attrs) {
  var e = document.createElement(tag);
  attrs = attrs || {};
  Object.keys(attrs).forEach(function(k) {
    var v = attrs[k];
    if (k === 'class') { e.className = v; } else if (k === 'html') { e.innerHTML = v; } else { e.setAttribute(k, v); }
  });
  for (var i = 2; i < arguments.length; i++) e.append(arguments[i]);
  return e;
}

function fetchSummary() {
  var msg = document.getElementById('stats-msg');
  msg.textContent = 'Computing usage statistics…';
  return fetch('/admin_stats_data')
    .then(function(r) {
      if (!r.ok) { throw new Error('HTTP ' + r.status); }
      return r.json();
    })
    .then(function(j) {
      msg.textContent = '';
      return j.summary || {};
    });
}

function renderSummary(s) {
  const root = document.getElementById('summary');
  root.innerHTML = '';
  const card = function(k, v) {
    return el('div', { class: 'stat panel' }, el('div', { class: 'value' }, String(v)), el('div', { class: 'label' }, k));
  };
  const totalSearches = (s.search_usage || []).reduce(function(acc, x) { return acc + x[1]; }, 0);
  const totalClassifiers = (s.classifier_trains_by_label || []).reduce(function(acc, x) { return acc + x[1]; }, 0);
  root.append(
    card('Days Observed', (s.days_observed || []).length),
    card('Unique Users', s.unique_users_overall || 0),
    card('Avg Users/Day', (s.avg_users_per_day || 0).toFixed ? (s.avg_users_per_day || 0).toFixed(2) : s.avg_users_per_day || 0),
    card('HTTP Requests', s.total_http_requests || 0),
    card('Total Searches', totalSearches),
    card('Classifiers Trained', totalClassifiers),
    card('Cache Hit Rate', ((((s.cache || {}).hit_rate || 0) * 100).toFixed ? (((s.cache || {}).hit_rate || 0) * 100).toFixed(1) : ((s.cache || {}).hit_rate || 0) * 100) + '%')
  );
}

// Simple state store for chart data by container id
var __charts = {};

// Stored after fetchSummary completes, used by generateDashboardPng
let adminSummary = null;

const SEARCH_TYPE_META = {
  classifier_search:    { label: 'Classifier Search',               color: '#7F77DD' },
  caption_search:       { label: 'Caption Search',                  color: '#378ADD' },
  semantic_video_to_video: { label: 'Video-to-Video Semantic Search', color: '#1D9E75' },
  semantic_text_to_video:  { label: 'Text-to-Video Semantic Search',  color: '#EF9F27' },
  trajectory_shape:     { label: 'Trajectory Shape Search',         color: '#D85A30' },
  cluster_search:       { label: 'Cluster Search',                  color: '#D4537E' },
  wm_search:            { label: 'WM Search',                       color: '#888780' },
  caption_embed_search: { label: 'Caption Search with Embeddings',  color: '#639922' },
  visual_search_text:   { label: 'Text-to-Video Visual Search',     color: '#BA7517' },
  visual_search_image:  { label: 'Image-to-Video Visual Search',    color: '#C4823A' },
  vlm_judge_validate_search: { label: 'VLM Judge Validate Search',  color: '#A855F7' },
  vlm_judge_caption_score:   { label: 'VLM Judge Caption Score',    color: '#7C3AED' },
};

function pad2(n) { return (n < 10 ? '0' : '') + n; }
function dateToStr(d) {
  const y = d.getFullYear();
  const m = pad2(d.getMonth() + 1);
  const dd = pad2(d.getDate());
  return `${y}-${m}-${dd}`;
}
function parseDateStr(s) {
  // Parse 'YYYY-MM-DD' safely without timezone drift
  const [y, m, d] = (s || '').split('-').map(Number);
  if (!y || !m || !d) return null;
  return new Date(Date.UTC(y, m - 1, d));
}
function getLastNDays(endStr, n) {
  const end = parseDateStr(endStr) || new Date();
  const arr = [];
  for (let i = n - 1; i >= 0; i--) {
    const d = new Date(end.getTime());
    d.setUTCDate(end.getUTCDate() - i);
    arr.push(dateToStr(d));
  }
  return arr;
}
function pickLatestDateStr(objA, objB) {
  const keys = [];
  if (objA) keys.push(...Object.keys(objA));
  if (objB) keys.push(...Object.keys(objB));
  if (keys.length === 0) return dateToStr(new Date());
  let best = keys[0];
  let bestT = Date.parse(keys[0]);
  for (let i = 1; i < keys.length; i++) {
    const t = Date.parse(keys[i]);
    if (isFinite(t) && t > bestT) { bestT = t; best = keys[i]; }
  }
  return best;
}

function renderBars(containerId, labels, values, valueLabels) {
  const root = document.getElementById(containerId);
  if (!root) { return; }
  root.innerHTML = '';
  if (!labels || !labels.length) { root.textContent = 'No data'; return; }
  const max = Math.max(1, ...values);
  const wrap = el('div', { class: 'bars' });
  labels.forEach(function(lab, i) {
    const v = values[i] || 0;
    const pct = Math.round((v / max) * 100);
    const fill = el('div', { class: 'fill' });
    fill.style.width = pct + '%';
    wrap.append(
      el('div', { class: 'bar' },
        el('div', { class: 'label', title: lab }, lab),
        el('div', { class: 'track' }, fill),
        el('div', { class: 'value' }, valueLabels ? String(valueLabels[i]) : String(v))
      )
    );
  });
  root.append(wrap);
}

function truncateText(ctx, text, maxWidth) {
  // Truncate text to fit in maxWidth with ellipsis
  if (ctx.measureText(text).width <= maxWidth) return text;
  let ell = '…';
  let lo = 0, hi = text.length;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    const candidate = text.slice(0, mid) + ell;
    if (ctx.measureText(candidate).width <= maxWidth) lo = mid + 1; else hi = mid;
  }
  return text.slice(0, Math.max(0, lo - 1)) + ell;
}

function drawBarsToCanvas(containerId) {
  const data = __charts[containerId];
  if (!data) return null;
  const labels = data.labels || [];
  const values = data.values || [];
  const title = data.title || '';
  const n = labels.length;
  if (!n) return null;

  const width = 1000;
  const margins = { top: 50, right: 80, bottom: 20, left: 220 };
  const barH = 18, gap = 8;
  const height = margins.top + margins.bottom + n * (barH + gap);
  const canvas = document.createElement('canvas');
  canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext('2d');

  // Background
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, width, height);

  // Title
  ctx.fillStyle = '#1f2937';
  ctx.font = '16px sans-serif';
  ctx.fillText(title, margins.left, 28);

  const maxVal = Math.max(1, ...values);
  const innerW = width - margins.left - margins.right;
  const labelW = margins.left - 30;
  ctx.font = '12px sans-serif';

  for (let i = 0; i < n; i++) {
    const y = margins.top + i * (barH + gap);
    const v = values[i] || 0;
    const lab = String(labels[i] || '');
    const w = Math.round((v / maxVal) * innerW);

    // Track
    ctx.fillStyle = '#f3f4f6';
    ctx.fillRect(margins.left, y, innerW, barH);
    // Fill
    ctx.fillStyle = '#3b82f6';
    ctx.fillRect(margins.left, y, w, barH);

    // Label
    ctx.fillStyle = '#111827';
    const txt = truncateText(ctx, lab, labelW);
    ctx.fillText(txt, 10, y + barH - 4);

    // Value
    ctx.fillStyle = '#6b7280';
    const valStr = String(v);
    const valW = ctx.measureText(valStr).width;
    ctx.fillText(valStr, margins.left + innerW - valW, y + barH - 4);
  }

  return canvas;
}

function drawLineToCanvas(containerId) {
  const data = __charts[containerId];
  if (!data) return null;
  const labels = data.labels || [];
  const values = data.values || [];
  const title = data.title || '';
  const n = values.length;
  if (!n) return null;

  const width = 1000;
  const margins = { top: 50, right: 30, bottom: 40, left: 60 };
  const height = 300;
  const canvas = document.createElement('canvas');
  canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext('2d');

  // Background
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, width, height);

  // Title
  ctx.fillStyle = '#1f2937';
  ctx.font = '16px sans-serif';
  ctx.fillText(title, margins.left, 28);

  const maxVal = Math.max(1, ...values);
  const x0 = margins.left, x1 = width - margins.right;
  const y0 = height - margins.bottom, y1 = margins.top;
  const innerW = x1 - x0;
  const innerH = y0 - y1;

  // Axes
  ctx.strokeStyle = '#e5e7eb';
  ctx.lineWidth = 1;
  // horizontal grid lines (4)
  for (let i = 0; i <= 4; i++) {
    const y = y1 + (innerH * i / 4);
    ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
  }
  // y-axis labels
  ctx.fillStyle = '#6b7280';
  ctx.font = '12px sans-serif';
  for (let i = 0; i <= 4; i++) {
    const val = Math.round(maxVal * (1 - i / 4));
    const y = y1 + (innerH * i / 4);
    ctx.fillText(String(val), 10, y + 4);
  }

  // Line
  ctx.strokeStyle = '#3b82f6';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = x0 + (n === 1 ? innerW / 2 : (innerW * i / (n - 1)));
    const y = y0 - (values[i] / maxVal) * innerH;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // X labels (sparse: up to 12)
  const steps = Math.max(1, Math.ceil(n / 12));
  ctx.fillStyle = '#6b7280';
  ctx.font = '11px sans-serif';
  for (let i = 0; i < n; i += steps) {
    const x = x0 + (n === 1 ? innerW / 2 : (innerW * i / (n - 1)));
    const txt = String(labels[i]);
    const tw = ctx.measureText(txt).width;
    ctx.fillText(txt, Math.max(x - tw / 2, 0), y0 + 16);
  }

  return canvas;
}

function downloadChart(containerId) {
  const data = __charts[containerId];
  if (!data) return;
  const canvas = (data.type === 'line') ? drawLineToCanvas(containerId) : drawBarsToCanvas(containerId);
  if (!canvas) return;
  const a = document.createElement('a');
  const safe = (data.title || containerId).replace(/[^a-z0-9]+/gi, '_').replace(/^_+|_+$/g, '').toLowerCase();
  a.download = safe + '.png';
  a.href = canvas.toDataURL('image/png');
  a.click();
}

function ensureDownloadButton(containerId, title) {
  const root = document.getElementById(containerId);
  if (!root) return;
  let titleEl = root.previousElementSibling;
  if (!titleEl || (titleEl.tagName !== 'H2' && titleEl.tagName !== 'H3')) {
    // Fallback: attach inside root at top
    titleEl = root;
  }
  // Avoid duplicate button per id
  if (titleEl.querySelector('button.download-btn[data-target="' + containerId + '"]')) return;
  const btn = document.createElement('button');
  btn.className = 'download-btn';
  btn.textContent = '⬇ Download PNG';
  btn.setAttribute('data-target', containerId);
  btn.style.marginLeft = '8px';
  btn.addEventListener('click', function() { downloadChart(containerId); });
  titleEl.appendChild(btn);
}

function renderChart(containerId, labels, values, title, type, valueLabels) {
  const chartType = type || 'bars';
  __charts[containerId] = { labels: labels || [], values: values || [], title: title || containerId, type: chartType };
  if (chartType === 'line') {
    renderLine(containerId, labels, values);
  } else {
    renderBars(containerId, labels, values, valueLabels);
  }
  ensureDownloadButton(containerId, title);
}

function renderLine(containerId, labels, values) {
  const root = document.getElementById(containerId);
  if (!root) return; root.innerHTML = '';
  const n = (values || []).length; if (!n) { root.textContent = 'No data'; return; }
  const svgNS = 'http://www.w3.org/2000/svg';
  const W = 800, H = 240, m = { top: 10, right: 10, bottom: 20, left: 40 };
  const x0 = m.left, y0 = H - m.bottom, x1 = W - m.right, y1 = m.top;
  const innerW = x1 - x0, innerH = y0 - y1;
  const maxVal = Math.max(1, ...values);
  const steps = Math.max(1, Math.ceil(n / 10));

  const svg = document.createElementNS(svgNS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('width', '100%');
  svg.setAttribute('height', '180');

  // grid lines
  for (let i = 0; i <= 4; i++) {
    const y = y1 + (innerH * i / 4);
    const line = document.createElementNS(svgNS, 'line');
    line.setAttribute('x1', x0);
    line.setAttribute('x2', x1);
    line.setAttribute('y1', y);
    line.setAttribute('y2', y);
    line.setAttribute('stroke', '#e5e7eb');
    line.setAttribute('stroke-width', '1');
    svg.appendChild(line);
  }

  // path
  const points = [];
  for (let i = 0; i < n; i++) {
    const x = x0 + (n === 1 ? innerW / 2 : (innerW * i / (n - 1)));
    const y = y0 - (values[i] / maxVal) * innerH;
    points.push(`${x},${y}`);
  }
  const poly = document.createElementNS(svgNS, 'polyline');
  poly.setAttribute('fill', 'none');
  poly.setAttribute('stroke', '#3b82f6');
  poly.setAttribute('stroke-width', '2');
  poly.setAttribute('points', points.join(' '));
  svg.appendChild(poly);

  // x labels (sparse)
  for (let i = 0; i < n; i += steps) {
    const x = x0 + (n === 1 ? innerW / 2 : (innerW * i / (n - 1)));
    const txt = document.createElementNS(svgNS, 'text');
    txt.textContent = String(labels[i]);
    txt.setAttribute('x', x);
    txt.setAttribute('y', y0 + 14);
    txt.setAttribute('text-anchor', 'middle');
    txt.setAttribute('font-size', '10');
    txt.setAttribute('fill', '#6b7280');
    svg.appendChild(txt);
  }

  root.appendChild(svg);
}

function renderAll(s) {
  adminSummary = s;
  renderSummary(s);
  // Daily unique users
  const du = s.daily_unique_users || {};
  const cacheDailyPreview = ((s.cache || {}).daily) || {};
  const endDateStr = pickLatestDateStr(du, cacheDailyPreview);
  const last30 = getLastNDays(endDateStr, 30);
  const duDaysInWindow = last30.filter(d => du[d] != null);
  const duVals = duDaysInWindow.map(d => du[d]);
  renderChart('daily-unique-users', duDaysInWindow, duVals, 'Daily Unique Users');
  // Requests by hour
  const byHour = s.requests_by_hour || [];
  renderChart('by-hour', byHour.map(x => `${x[0]}:00`), byHour.map(x => x[1]), 'Avg Requests by Hour (per day)');

  // Specific endpoint usage panel
  const ep = s.endpoint_usage || [];
  if (ep.length) {
    renderChart('endpoint-usage', ep.map(x => x[0]), ep.map(x => x[1]), 'Endpoint Usage');
  }
  // Status
  const status = s.http_status || [];
  renderChart('status', status.map(x => String(x[0])), status.map(x => x[1]), 'Status Codes');
  // Methods
  const methods = s.http_methods || [];
  renderChart('methods', methods.map(x => x[0]), methods.map(x => x[1]), 'Methods');
  // Row 1: Search usage and Avg search time
  const search = s.search_usage || [];
  renderChart('search-usage', search.map(x => x[0]), search.map(x => x[1]), 'Search Usage');
  const avgST = s.avg_search_s_by_type || [];
  renderChart(
    'avg-search-ms',
    avgST.map(x => x[0]),
    avgST.map(x => Math.round((x[1] || 0) * 100) / 100),
    'Avg Search Time (s)'
  );
  // Row 2: Top Data Sources and Top Labels
  const ds = s.top_data_sources || [];
  renderChart('top-ds', ds.map(x => x[0]), ds.map(x => x[1]), 'Top Data Sources');
  const topLabels = s.top_label_filters || [];
  renderChart('top-labels', topLabels.map(x => x[0]), topLabels.map(x => x[1]), 'Top Labels');
  // Row 3: Top Caption Searches and Top Caption Embed Searches
  const topCaptions = s.top_caption_searches || [];
  renderChart('top-caption-searches', topCaptions.map(x => x[0]), topCaptions.map(x => x[1]), 'Top Caption Searches');
  const topCaptionEmbed = s.top_caption_embed_search_texts || [];
  renderChart('top-caption-embed-searches', topCaptionEmbed.map(x => x[0]), topCaptionEmbed.map(x => x[1]), 'Top Caption Embed Searches');
  // Row 4: Top Visual Search Texts and Top Semantic Search Texts
  const topVST = s.top_visual_search_texts || [];
  renderChart('top-visual-search-texts', topVST.map(x => x[0]), topVST.map(x => x[1]), 'Top Visual Search Texts');
  // Row 4: Top Semantic Search Texts and Zero-Result Searches
  const topSST = s.top_semantic_search_texts || [];
  renderChart('top-semantic-search-texts', topSST.map(x => x[0]), topSST.map(x => x[1]), 'Top Text-to-Video Semantic Search Texts');
  const zrc = s.zero_result_count_by_type || [];
  const searchTotals = Object.fromEntries((s.searches_by_type || []).map(x => [x[0], x[1]]));
  renderChart(
    'zero-result-rate',
    zrc.map(x => x[0]),
    zrc.map(x => x[1] || 0),
    'Zero-Result Searches',
    null,
    zrc.map(x => {
      const total = searchTotals[x[0]];
      const pct = total ? ' (' + (Math.round((x[1] / total) * 1000) / 10) + '%)' : '';
      return x[1] + pct;
    })
  );
  // Row 5: Zero-result queries by type
  const zSem = s.zero_semantic_search_texts || [];
  if (zSem.length) {
    renderChart('zero-semantic-texts', zSem.map(x => x[0]), zSem.map(x => x[1]), 'Zero-Result Text-to-Video');
  }
  const zCap = s.zero_caption_search_texts || [];
  if (zCap.length) {
    renderChart('zero-caption-texts', zCap.map(x => x[0]), zCap.map(x => x[1]), 'Zero-Result Captions');
  }
  const zCapEmbed = s.zero_caption_embed_search_texts || [];
  if (zCapEmbed.length) {
    renderChart('zero-caption-embed-texts', zCapEmbed.map(x => x[0]), zCapEmbed.map(x => x[1]), 'Zero-Result Caption Embed Searches');
  }
  const zVis = s.zero_visual_search_texts || [];
  if (zVis.length) {
    renderChart('zero-visual-texts', zVis.map(x => x[0]), zVis.map(x => x[1]), 'Zero-Result Visual Searches');
  }
  // Requests per user (top N from analyzer)
  const topUsers = s.top_users || [];
  renderChart('requests-per-user', topUsers.map(x => x[0]), topUsers.map(x => x[1]), 'Requests per User');

  // Annotation activity (last 30 days, only days with data per series)
  const ann = s.annotations_daily || {};
  const annDays = Object.keys(ann);
  const endDateStr2 = pickLatestDateStr(ann, null);
  const last30_2 = getLastNDays(endDateStr2, 30);
  function seriesFor(key) {
    const days = last30_2.filter(d => ann[d] && (ann[d][key] || 0) > 0);
    const vals = days.map(d => (ann[d] && ann[d][key]) || 0);
    return { days, vals };
  }
  const sAdded = seriesFor('added');
  renderChart('ann-added', sAdded.days, sAdded.vals, 'Added per Day');
  const sDeleted = seriesFor('deleted');
  renderChart('ann-deleted', sDeleted.days, sDeleted.vals, 'Deleted per Day');
  const sAuto = seriesFor('autolabeled');
  renderChart('ann-autolabeled', sAuto.days, sAuto.vals, 'Autolabeled per Day');
  const sVer = seriesFor('verified');
  renderChart('ann-verified', sVer.days, sVer.vals, 'Verified per Day');
  // Classifier Training
  const ctLabel = s.classifier_trains_by_label || [];
  if (ctLabel.length) {
    renderChart('classifier-trains-by-label', ctLabel.map(x => x[0]), ctLabel.map(x => x[1]), 'Trainings by Label');
  }
  const ctEmbed = s.classifier_trains_by_embed_type || [];
  if (ctEmbed.length) {
    renderChart('classifier-trains-by-embed-type', ctEmbed.map(x => x[0]), ctEmbed.map(x => x[1]), 'Trainings by Embed Type');
  }
  const ctUser = s.classifier_trains_by_user || [];
  if (ctUser.length) {
    renderChart('classifier-trains-by-user', ctUser.map(x => x[0]), ctUser.map(x => x[1]), 'Trainings by User');
  }
  // Query Rewrite
  const rw = s.query_rewrite || {};
  if (rw.total_calls) {
    const rwSummaryRoot = document.getElementById('query-rewrite-summary');
    if (rwSummaryRoot) {
      rwSummaryRoot.innerHTML = '';
      const card = function(k, v) {
        return el('div', { class: 'stat panel' }, el('div', { class: 'value' }, String(v)), el('div', { class: 'label' }, k));
      };
      rwSummaryRoot.append(card('Total Calls', rw.total_calls || 0), card('Unique Users', rw.unique_users || 0));
    }
    const rwq = rw.top_queries || [];
    if (rwq.length) {
      renderChart('rewrite-top-queries', rwq.map(x => x[0]), rwq.map(x => x[1]), 'Top Rewritten Queries');
    }
    const rwu = rw.top_users || [];
    if (rwu.length) {
      renderChart('rewrite-top-users', rwu.map(x => x[0]), rwu.map(x => x[1]), 'Top Rewrite Users');
    }
    const rwByType = rw.by_type || {};
    const rwByTypeRoot = document.getElementById('rewrite-by-type');
    if (rwByTypeRoot) {
      rwByTypeRoot.innerHTML = '';
      ['caption', 'caption-embed', 'semantic', 'visual'].forEach(function(stype) {
        const td = rwByType[stype];
        if (!td) return;
        const chartId = 'rewrite-type-' + stype.replace('-', '_');
        const queries = td.top_queries || [];
        const wrapper = el('div', {},
          el('h3', {}, stype + ' (' + (td.searches_with_rewrites || 0) + ' searches with rewrites)'),
          el('div', { id: chartId, class: 'chart' })
        );
        rwByTypeRoot.append(wrapper);
        if (queries.length) {
          renderChart(chartId, queries.map(x => x[0]), queries.map(x => x[1]), stype + ' rewritten queries');
        }
      });
    }
  }
  // Enable dashboard export now that data is ready
  const pngBtn = document.getElementById('dashboard-png-btn');
  if (pngBtn) pngBtn.disabled = false;

  // Cache
  const cache = s.cache || {};
  const cacheLabels = ['hits', 'misses'];
  const cacheVals = [cache.hits || 0, cache.misses || 0];
  renderChart('cache', cacheLabels, cacheVals, 'Cache Statistics');
  // Daily Cache Hit Rate
  const cacheDaily = (cache.daily) || {};
  const cacheDaysInWindow = last30.filter(d => cacheDaily[d] && typeof cacheDaily[d].hit_rate === 'number');
  // Convert to percent with one decimal place
  const cacheRatesPct = cacheDaysInWindow.map(d => Math.round((cacheDaily[d].hit_rate * 100) * 10) / 10);
  renderChart('cache-hit-rate', cacheDaysInWindow, cacheRatesPct, 'Daily Cache Hit Rate');
}

// ── Dashboard PNG export ──────────────────────────────────────────────────────

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function dashRoundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function generateDashboardPng() {
  const btn = document.getElementById('dashboard-png-btn');
  if (!adminSummary) return;
  btn.disabled = true;
  btn.textContent = 'Generating…';
  // yield to the browser so the button state repaints before the heavy canvas work
  setTimeout(function() {
    try { doGenerateDashboardPng(); } finally {
      btn.disabled = false;
      btn.textContent = 'Download Dashboard PNG';
    }
  }, 16);
}

function doGenerateDashboardPng() {
  const s = adminSummary;
  const BG = '#FAFAF8', GRID = '#EBEBEB', TXT = '#2C2C2A', MUT = '#73726C';
  const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

  // ── Daily data (last 75 days) ──────────────────────────────────────────────
  const du = s.daily_unique_users || {};
  const endDateStr = pickLatestDateStr(du, ((s.cache || {}).daily) || {});
  const last75 = getLastNDays(endDateStr, 75);
  const dailyDates = last75.filter(d => du[d] != null);
  const dailyVals  = dailyDates.map(d => du[d]);

  // monthly tick marks for x-axis
  const monthTicks = [];
  const seenMonths = new Set();
  dailyDates.forEach(function(d, i) {
    const key = d.slice(0, 7);
    if (!seenMonths.has(key)) {
      seenMonths.add(key);
      const dt = parseDateStr(d);
      const lbl = dt ? MONTHS[dt.getUTCMonth()] + " '" + String(dt.getUTCFullYear()).slice(2) : key;
      monthTicks.push({ i, lbl });
    }
  });

  // ── Search type data ───────────────────────────────────────────────────────
  const searchByType = Object.fromEntries((s.searches_by_type || []).map(x => [x[0], x[1]]));
  const zeroByType   = Object.fromEntries((s.zero_result_count_by_type || []).map(x => [x[0], x[1]]));
  const KNOWN_ORDER  = ['classifier_search','caption_search','semantic_video_to_video','semantic_text_to_video',
                        'trajectory_shape','cluster_search','wm_search','caption_embed_search',
                        'visual_search_text','visual_search_image',
                        'vlm_judge_validate_search','vlm_judge_caption_score'];

  const searchTypes = KNOWN_ORDER
    .filter(k => searchByType[k] != null)
    .map(function(k) {
      const meta     = SEARCH_TYPE_META[k] || { label: k, color: '#888888' };
      const total    = searchByType[k] || 0;
      const zeros    = zeroByType[k]   || 0;
      const zeroRate = total > 0 ? Math.round(zeros / total * 100) : 0;
      return {
        label: meta.label, color: meta.color, total, zeros, zeroRate,
        zeroColor: zeroRate > 40 ? '#E24B4A' : zeroRate > 20 ? '#EF9F27' : '#1D9E75',
      };
    });
  // append any unknown types
  Object.keys(searchByType).forEach(function(k) {
    if (!KNOWN_ORDER.includes(k)) {
      const total = searchByType[k] || 0, zeros = zeroByType[k] || 0;
      const zeroRate = total > 0 ? Math.round(zeros / total * 100) : 0;
      searchTypes.push({ label: k, color: '#888888', total, zeros, zeroRate,
        zeroColor: zeroRate > 40 ? '#E24B4A' : zeroRate > 20 ? '#EF9F27' : '#1D9E75' });
    }
  });

  // ── Metric card values ─────────────────────────────────────────────────────
  const totalSearches = (s.search_usage || []).reduce((a, x) => a + x[1], 0);
  const peakDaily     = dailyVals.length > 0 ? Math.max(...dailyVals) : 0;
  const avgDay        = (+(s.avg_users_per_day || 0)).toFixed(1);
  const metrics = [
    ['Total searches',  totalSearches.toLocaleString()],
    ['Unique users',    String(s.unique_users_overall || 0)],
    ['Peak daily users', String(peakDaily)],
    ['Avg users/day',   avgDay],
  ];

  // ── Canvas dimensions ──────────────────────────────────────────────────────
  const W      = 1300;
  const SCALE  = 2;          // retina-quality output
  const barH   = 22, barGap = 9;
  const nBars  = searchTypes.length;
  const barsChartH = Math.max(0, nBars * (barH + barGap) - barGap);

  const LEFT_PAD = 80, RIGHT_PAD = 80, LABEL_W = 160;
  const barLeft  = LEFT_PAD + LABEL_W;
  const barRight = W - RIGHT_PAD;
  const barWidth = barRight - barLeft;

  // section y positions
  const yTitle      = 12;
  const yCards      = 48;
  const cardH       = 62;
  const yLine       = yCards + cardH + 22;
  const lineH       = 185;
  const ySearchVol  = yLine + lineH + 38;
  const searchSecH  = 20 + barsChartH + 22;   // title + bars + x-axis labels
  // const yZeroRate   = ySearchVol + searchSecH + 28;
  // const ZERO_LEGEND_H = 30;
  // const zeroSecH    = searchSecH + ZERO_LEGEND_H;  // legend on top + bars
  const totalH      = ySearchVol + searchSecH + 24;

  const canvas = document.createElement('canvas');
  canvas.width  = W * SCALE;
  canvas.height = totalH * SCALE;
  const ctx = canvas.getContext('2d');
  ctx.scale(SCALE, SCALE);

  // ── Background ─────────────────────────────────────────────────────────────
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, W, totalH);

  // ── Title ──────────────────────────────────────────────────────────────────
  ctx.fillStyle = TXT;
  ctx.font = 'bold 16px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Search Usage & Daily Unique Users', W / 2, yTitle + 20);
  ctx.textAlign = 'left';

  // ── Metric cards ───────────────────────────────────────────────────────────
  const cardW   = 275, cardGap = 18;
  const totalCW = metrics.length * cardW + (metrics.length - 1) * cardGap;
  const cardX0  = (W - totalCW) / 2;
  metrics.forEach(function([lbl, val], i) {
    const cx = cardX0 + i * (cardW + cardGap), cy = yCards;
    ctx.fillStyle = '#F1EFE8';
    dashRoundRect(ctx, cx, cy, cardW, cardH, 5);
    ctx.fill();
    ctx.fillStyle = MUT;  ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(lbl, cx + cardW / 2, cy + 20);
    ctx.fillStyle = TXT;  ctx.font = 'bold 22px sans-serif';
    ctx.fillText(val, cx + cardW / 2, cy + 48);
    ctx.textAlign = 'left';
  });

  // ── Line chart: daily unique users ─────────────────────────────────────────
  const lm  = { left: LEFT_PAD, right: RIGHT_PAD, top: 18, bottom: 28 };
  const lx  = lm.left, lw = W - lm.left - lm.right;
  const ly  = yLine + lm.top, lh = lineH - lm.top - lm.bottom;
  const n   = dailyVals.length;
  const maxV = Math.max(1, ...dailyVals);

  ctx.fillStyle = MUT; ctx.font = '10px sans-serif';
  ctx.fillText('Daily unique users', lx, yLine + 12);

  // y-axis grid + labels
  for (let i = 0; i <= 4; i++) {
    const gy = ly + lh * i / 4;
    ctx.strokeStyle = GRID; ctx.lineWidth = 0.6;
    ctx.beginPath(); ctx.moveTo(lx, gy); ctx.lineTo(lx + lw, gy); ctx.stroke();
    ctx.fillStyle = MUT; ctx.font = '9px sans-serif'; ctx.textAlign = 'right';
    ctx.fillText(String(Math.round(maxV * (1 - i / 4))), lx - 5, gy + 4);
    ctx.textAlign = 'left';
  }

  if (n > 0) {
    const px = function(i) { return lx + (n === 1 ? lw / 2 : lw * i / (n - 1)); };
    const py = function(v) { return ly + lh - (v / maxV) * lh; };

    // fill under line
    ctx.fillStyle = hexToRgba('#378ADD', 0.12);
    ctx.beginPath();
    ctx.moveTo(px(0), ly + lh);
    dailyVals.forEach(function(v, i) { ctx.lineTo(px(i), py(v)); });
    ctx.lineTo(px(n - 1), ly + lh);
    ctx.closePath(); ctx.fill();

    // line
    ctx.strokeStyle = '#378ADD'; ctx.lineWidth = 1.8;
    ctx.beginPath();
    dailyVals.forEach(function(v, i) {
      if (i === 0) ctx.moveTo(px(i), py(v)); else ctx.lineTo(px(i), py(v));
    });
    ctx.stroke();
  }

  // x-axis month labels
  ctx.fillStyle = MUT; ctx.font = '9px sans-serif'; ctx.textAlign = 'center';
  monthTicks.forEach(function({ i, lbl }) {
    const x = n <= 1 ? lx + lw / 2 : lx + lw * i / (n - 1);
    ctx.fillText(lbl, x, ly + lh + 18);
  });
  ctx.textAlign = 'left';

  // frame
  ctx.strokeStyle = GRID; ctx.lineWidth = 0.6;
  ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx, ly + lh); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(lx, ly + lh); ctx.lineTo(lx + lw, ly + lh); ctx.stroke();

  // ── Horizontal bar chart helper ────────────────────────────────────────────
  function drawHBars(yStart, title, labels, values, colors, valueLabels, maxVal, axisLabel, topLegend) {
    const mv      = maxVal != null ? maxVal : Math.max(1, ...values);
    const legendH = topLegend ? ZERO_LEGEND_H : 0;
    const barsY   = yStart + 20 + legendH;

    // title
    ctx.fillStyle = TXT; ctx.font = 'bold 13px sans-serif';
    ctx.fillText(title, barLeft, yStart + 14);

    // optional legend row between title and bars
    if (topLegend) {
      let lgX = barRight;
      topLegend.slice().reverse().forEach(function({ color, label }) {
        ctx.font = 'bold 11px sans-serif';
        const tw = ctx.measureText(label).width;
        lgX -= tw + 10;
        // swatch
        ctx.fillStyle = hexToRgba(color, 0.7);
        dashRoundRect(ctx, lgX - 20, yStart + 18, 15, 13, 2);
        ctx.fill();
        ctx.strokeStyle = color; ctx.lineWidth = 0.8; ctx.stroke();
        // label text
        ctx.fillStyle = MUT;
        ctx.fillText(label, lgX, yStart + 30);
        lgX -= 28;
      });
    }

    // vertical grid lines + x-axis tick labels
    ctx.font = '9px sans-serif';
    for (let gi = 0; gi <= 4; gi++) {
      const gx = barLeft + barWidth * gi / 4;
      ctx.strokeStyle = GRID; ctx.lineWidth = 0.6;
      ctx.beginPath(); ctx.moveTo(gx, barsY); ctx.lineTo(gx, barsY + barsChartH); ctx.stroke();
      const tickVal = Math.round(mv * gi / 4);
      const tickStr = axisLabel ? tickVal + '%' : (tickVal >= 1000 ? (tickVal / 1000).toFixed(tickVal % 1000 ? 1 : 0) + 'k' : String(tickVal));
      ctx.fillStyle = MUT; ctx.textAlign = 'center';
      ctx.fillText(tickStr, gx, barsY + barsChartH + 14);
      ctx.textAlign = 'left';
    }

    labels.forEach(function(label, i) {
      const by  = barsY + i * (barH + barGap);
      const v   = values[i] || 0;
      const bw  = Math.round((v / mv) * barWidth);
      const col = colors[i] || '#888888';

      // bar
      ctx.fillStyle = hexToRgba(col, 0.8);
      ctx.fillRect(barLeft, by, bw, barH);
      ctx.strokeStyle = col; ctx.lineWidth = 0.8;
      ctx.strokeRect(barLeft, by, bw, barH);

      // label (right-aligned into label column)
      ctx.fillStyle = TXT; ctx.font = '10px sans-serif'; ctx.textAlign = 'right';
      ctx.fillText(truncateText(ctx, label, LABEL_W - 10), barLeft - 8, by + barH - 5);
      ctx.textAlign = 'left';

      // value label after bar
      ctx.fillStyle = MUT; ctx.font = '9px sans-serif';
      ctx.fillText(valueLabels ? valueLabels[i] : String(v), barLeft + bw + 5, by + barH - 5);
    });
  }

  // ── Search volume ──────────────────────────────────────────────────────────
  drawHBars(
    ySearchVol,
    'Search volume by type',
    searchTypes.map(t => t.label),
    searchTypes.map(t => t.total),
    searchTypes.map(t => t.color),
    searchTypes.map(t => t.total.toLocaleString()),
    null, null
  );

  // ── Zero-result rate ───────────────────────────────────────────────────────
  // drawHBars(
  //   yZeroRate,
  //   'Zero-result rate by search type',
  //   searchTypes.map(t => t.label),
  //   searchTypes.map(t => t.zeroRate),
  //   searchTypes.map(t => t.zeroColor),
  //   searchTypes.map(t => t.zeroRate + '%'),
  //   Math.max(55, ...searchTypes.map(t => t.zeroRate)),
  //   '%',
  //   [
  //     { color: '#1D9E75', label: 'Healthy (<20%)' },
  //     { color: '#EF9F27', label: 'Moderate (20–40%)' },
  //     { color: '#E24B4A', label: 'Concerning (>40%)' },
  //   ]
  // );

  // ── Download ───────────────────────────────────────────────────────────────
  const a = document.createElement('a');
  a.download = 'dashboard.png';
  a.href = canvas.toDataURL('image/png');
  a.click();
}

document.getElementById('dashboard-png-btn').addEventListener('click', generateDashboardPng);

document.getElementById('logout-btn').addEventListener('click', function() {
  const btn = document.getElementById('logout-btn');
  const prev = btn.textContent; btn.disabled = true; btn.textContent = 'Logging out…';
  fetch('/', { method: 'POST', body: 'logout::' })
    .catch(function() {})
    .then(function() {
      btn.disabled = false; btn.textContent = prev;
      window.location.replace('/login');
    });
});

fetchSummary()
  .then(function(s) { renderAll(s); })
  .catch(function() {
    const msg = document.getElementById('stats-msg');
    msg.textContent = 'Failed to compute statistics';
    msg.style.color = '#b91c1c';
  });
