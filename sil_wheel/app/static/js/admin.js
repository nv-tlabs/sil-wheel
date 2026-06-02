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

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => {
    if (k === 'class') e.className = v; else if (k === 'html') e.innerHTML = v; else e.setAttribute(k, v);
  });
  for (const c of children) e.append(c);
  return e;
}

async function loadAdminData() {
  const data = await api('/admin_data');
  const setMsg = (t, isErr=false) => {
    const el = document.getElementById('admin-msg');
    if (!el) return;
    el.textContent = t;
    if (isErr) el.style.color = '#b91c1c'; else el.style.color = '';
    if (t) setTimeout(() => { if (el.textContent === t) el.textContent = ''; }, 4000);
  };
  const tbodyU = document.querySelector('#users-table tbody');
  tbodyU.innerHTML = '';
  // Update users count in header
  const usersCountEl = document.getElementById('users-count');
  if (usersCountEl) usersCountEl.textContent = `(${(data.users || []).length})`;
  const dsOptions = data.dataset_options || [];
  for (const u of data.users) {
    // Build datasources select with options
    const dsSelect = el(
      'select',
      { multiple: true, size: Math.min(6, dsOptions.length) || 3, 'data-ds-username': u.username },
      ...dsOptions.map(opt => {
        const o = el('option', { value: opt }, opt);
        if ((u.datasources || []).includes(opt)) o.selected = true;
        return o;
      })
    );
    const createdTs = u.created_at ? new Date((u.created_at || 0) * 1000) : null;
    const createdStr = createdTs && isFinite(createdTs) ? createdTs.toLocaleString() : '';
    const tr = el('tr', {},
      el('td', {}, u.username),
      el('td', {}, el('input', { type: 'email', value: u.email || '', 'data-username': u.username })),
      el('td', {},
        el('select', { 'data-username': u.username },
          ...['user','admin'].map(role => {
            const opt = el('option', { value: role }, role);
            if (u.role === role) opt.selected = true;
            return opt;
          })
        )
      ),
      el('td', {}, dsSelect),
      el('td', {}, createdStr),
      el('td', {}, el('input', { type: 'password', placeholder: 'new password', 'data-username': u.username })),
      el('td', {},
        el('button', { class: 'link danger', 'data-act': 'delete', 'data-username': u.username }, 'Delete'), ' ',
        el('button', { class: 'link', 'data-act': 'save', 'data-username': u.username }, 'Save')
      ),
    );
    tbodyU.appendChild(tr);
    // Enhance with select2 if available
    try {
      if (window.$ && $(dsSelect).select2) {
        $(dsSelect).select2({
          width: '100%',
          closeOnSelect: false,
          dropdownAutoWidth: true,
        });
      }
    } catch (_) { /* no-op if select2 not available */ }
  }

  // Row actions: save/delete
  tbodyU.querySelectorAll('button[data-act]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const act = btn.getAttribute('data-act');
      const username = btn.getAttribute('data-username');
      if (act === 'delete') {
        if (!confirm(`Delete user ${username}?`)) return;
        try {
          const prev = btn.textContent; btn.disabled = true; btn.textContent = 'Deleting…';
          const r = await fetch('/', { method: 'POST', body: `admin_delete_user::${username}` });
          if (!r.ok) throw new Error();
          setMsg(`Deleted user ${username}`);
          await loadAdminData();
        } catch (_) {
          setMsg(`Failed to delete ${username}`, true);
        } finally {
          btn.disabled = false; btn.textContent = 'Delete';
        }
        return;
      }
      if (act === 'save') {
        const row = btn.closest('tr');
        const email = row.querySelector('input[type="email"][data-username]')?.value || '';
        const role = row.querySelector('select[data-username]')?.value || '';
        const pw = row.querySelector('input[type="password"][data-username]')?.value || '';
        const dsSel = row.querySelector('select[multiple][data-ds-username]');
        const ds = Array.from(dsSel?.selectedOptions || []).map(o => o.value).join(', ');
        try {
          const prev = btn.textContent; btn.disabled = true; btn.textContent = 'Saving…';
          const r = await fetch('/', { method: 'POST', body: `admin_update_user::${username}::${email}::${role}::${pw}::${ds}` });
          if (!r.ok) throw new Error();
          setMsg(`Updated user ${username}`);
        } catch (_) {
          setMsg(`Failed to update ${username}`, true);
        } finally {
          const pwEl = row.querySelector('input[type="password"][data-username]');
          if (pwEl) pwEl.value = '';
          btn.disabled = false; btn.textContent = 'Save';
        }
      }
    });
  });

  // Access requests
  const tbodyR = document.querySelector('#requests-table tbody');
  tbodyR.innerHTML = '';
  for (const r of data.requests) {
    const ts = r.created_at ? new Date((r.created_at || 0) * 1000) : null;
    const when = ts && isFinite(ts) ? ts.toLocaleString() : '';
    const tr = el('tr', {},
      el('td', {}, r.id),
      el('td', {}, r.username),
      el('td', {}, r.email || ''),
      el('td', {}, r.reason || ''),
      el('td', {}, when),
      el('td', {}, r.status),
      el('td', {},
        el('button', { 'data-id': r.id, 'data-act': 'approved' }, 'Approve'), ' ',
        el('button', { 'data-id': r.id, 'data-act': 'rejected' }, 'Reject')
      ),
    );
    tbodyR.appendChild(tr);
  }

  tbodyR.querySelectorAll('button[data-id]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const id = btn.getAttribute('data-id');
      const act = btn.getAttribute('data-act');
      try {
        const prev = btn.textContent; btn.disabled = true; btn.textContent = 'Working…';
        const r = await fetch('/', { method: 'POST', body: `admin_set_request_status::${id}::${act}` });
        if (!r.ok) throw new Error();
        setMsg(`Request ${id} ${act}`);
        await loadAdminData();
      } catch (_) {
        setMsg(`Failed to ${act} request ${id}`, true);
      } finally {
        btn.disabled = false; btn.textContent = act === 'approved' ? 'Approve' : 'Reject';
      }
    });
  });
}

document.getElementById('create-user-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  if (!form.reportValidity()) return;
  const u = document.getElementById('cu-username').value.trim();
  const email = document.getElementById('cu-email').value.trim();
  const p = document.getElementById('cu-password').value;
  const role = document.getElementById('cu-role').value;
  const msg = document.getElementById('create-user-msg');
  msg.textContent = '';
  try {
    const btn = form.querySelector('button[type="submit"]');
    const prev = btn.textContent; btn.disabled = true; btn.textContent = 'Creating…';
    const r = await fetch('/', { method: 'POST', body: `admin_create_user::${u}::${p}::${email}::${role}` });
    if (!r.ok) throw new Error();
    msg.textContent = 'User created';
    e.target.reset();
    await loadAdminData();
  } catch (err) {
    msg.style.color = '#b91c1c';
    msg.textContent = 'Failed to create user';
    if (err && err.message && /400/.test(String(err.message))) {
      msg.textContent = 'Password is required to create a user.';
    }
  } finally {
    const btn = form.querySelector('button[type="submit"]');
    btn.disabled = false; btn.textContent = 'Create';
  }
});

document.getElementById('logout-btn').addEventListener('click', async () => {
  const btn = document.getElementById('logout-btn');
  const prev = btn.textContent; btn.disabled = true; btn.textContent = 'Logging out…';
  await fetch('/', { method: 'POST', body: 'logout::' }).catch(() => {});
  btn.disabled = false; btn.textContent = prev;
  window.location.replace('/login');
});

// Load on start (after confirming admin in admin.html inline script)
loadAdminData().catch(() => {
  // If /admin_data fails (e.g. not admin), go to login
  window.location.replace('/login');
});
