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

// Simple tab switcher
const tabLogin = document.getElementById('tab-login');
const tabRequest = document.getElementById('tab-request');
const loginPane = document.getElementById('login-pane');
const requestPane = document.getElementById('request-pane');

tabLogin.addEventListener('click', () => {
  tabLogin.classList.add('active');
  tabRequest.classList.remove('active');
  loginPane.classList.add('active');
  requestPane.classList.remove('active');
});

tabRequest.addEventListener('click', () => {
  tabRequest.classList.add('active');
  tabLogin.classList.remove('active');
  requestPane.classList.add('active');
  loginPane.classList.remove('active');
});

// Login handler
document.getElementById('login-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  if (!form.reportValidity()) return;
  const u = document.getElementById('login-username').value.trim();
  const p = document.getElementById('login-password').value;
  const msg = document.getElementById('login-message');
  const btn = form.querySelector('button[type="submit"]');
  const prevText = btn.textContent;
  msg.textContent = '';
  msg.style.color = '';
  btn.disabled = true;
  btn.textContent = 'Logging in…';

  try {
    const payload = `user_login::${u}::${p}`;
    const res = await fetch('/', {
      method: 'POST',
      body: payload,
    });
    if (res.ok) {
      window.location.replace('/');
    } else if (res.status === 403) {
      msg.style.color = '#b91c1c';
      msg.textContent = 'Invalid credentials or inactive account.';
    } else {
      msg.style.color = '#b91c1c';
      msg.textContent = 'Login failed. Please try again.';
    }
  } catch (err) {
    msg.style.color = '#b91c1c';
    msg.textContent = 'Network error. Please try again.';
  } finally {
    btn.disabled = false;
    btn.textContent = prevText;
  }
});

// Access request handler
document.getElementById('request-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  if (!form.reportValidity()) return;
  const u = document.getElementById('request-username').value.trim();
  const pw = document.getElementById('request-password').value;
  const email = document.getElementById('request-email').value.trim();
  const reason = document.getElementById('request-reason').value.trim();
  const msg = document.getElementById('request-message');
  const btn = form.querySelector('button[type="submit"]');
  const prevText = btn.textContent;
  msg.textContent = '';
  msg.style.color = '';
  btn.disabled = true;
  btn.textContent = 'Sending…';

  try {
    const payload = `request_access::${u}::${pw}::${email}::${reason}`;
    const res = await fetch('/', {
      method: 'POST',
      body: payload,
    });
    if (res.ok) {
      msg.textContent = 'Request submitted. Your request should be approved promptly — if not, please reach out to Despoina Paschalidou (dpaschalidou@nvidia.com).';
      form.reset();
    } else {
      msg.style.color = '#b91c1c';
      msg.textContent = 'Could not submit the request. Ensure password and email are valid.';
    }
  } catch (err) {
    msg.style.color = '#b91c1c';
    msg.textContent = 'Network error. Please try again.';
  } finally {
    btn.disabled = false;
    btn.textContent = prevText;
  }
});
