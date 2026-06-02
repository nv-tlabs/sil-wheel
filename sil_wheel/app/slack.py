# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import configparser
import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from urllib import error, request

DEFAULT_PROFILE = "sil-wheel"
DEFAULT_BASE_DIR = "~/.slack"


@dataclass(frozen=True)
class SlackConfig:
    profile: str
    base_dir: Path
    bot_token: str
    signing_secret: str | None
    app_token: str | None
    channel_id: str | None
    dm_user_ids: tuple[str, ...]

    @property
    def has_destination(self) -> bool:
        return bool(self.channel_id or self.dm_user_ids)


def _read_ini(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None)
    if path.exists():
        parser.read(path)
    return parser


def _profile_value(
    parser: configparser.ConfigParser, profile: str, key: str
) -> str | None:
    if not parser.has_section(profile) or not parser.has_option(profile, key):
        return None
    value = parser.get(profile, key).strip()
    return value if value else None


def _parse_csv_ids(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return tuple()
    out = []
    seen = set()
    for part in raw.split(","):
        value = part.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


def load_slack_config(
    profile: str = DEFAULT_PROFILE, base_dir: str = DEFAULT_BASE_DIR
) -> SlackConfig | None:
    """Loads outbound Slack config from ~/.slack/{config,credentials}.

    Returns None when Slack is not configured or required fields are missing.
    """
    root = Path(base_dir).expanduser()
    config_path = root / "config"
    credentials_path = root / "credentials"

    if not config_path.exists() and not credentials_path.exists():
        return None

    config = _read_ini(config_path)
    credentials = _read_ini(credentials_path)

    if not config.has_section(profile) and not credentials.has_section(profile):
        return None

    bot_token = _profile_value(credentials, profile, "bot_token")
    signing_secret = _profile_value(credentials, profile, "signing_secret")
    app_token = _profile_value(credentials, profile, "app_token")
    channel_id = _profile_value(config, profile, "channel_id")
    dm_user_ids = _parse_csv_ids(_profile_value(config, profile, "dm_user_ids"))

    if not bot_token:
        return None

    slack_config = SlackConfig(
        profile=profile,
        base_dir=root,
        bot_token=bot_token,
        signing_secret=signing_secret,
        app_token=app_token,
        channel_id=channel_id,
        dm_user_ids=dm_user_ids,
    )

    if not slack_config.has_destination:
        return None

    return slack_config


class SlackNotifier:
    def __init__(self, config: SlackConfig, timeout_s: float = 5.0):
        self.config = config
        self.timeout_s = float(timeout_s)

    def _api_post(self, method: str, payload: dict) -> dict:
        req = request.Request(
            url=f"https://slack.com/api/{method}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.bot_token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Slack API {method} HTTP {exc.code}: {body[:200]}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"Slack API {method} network error: {exc.reason}"
            ) from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Slack API {method} returned invalid JSON"
            ) from exc

        if not data.get("ok"):
            raise RuntimeError(
                f"Slack API {method} failed: {data.get('error', 'unknown_error')}"
            )
        return data

    def _post_message(self, channel: str, text: str):
        self._api_post("chat.postMessage", {"channel": channel, "text": text})

    def _open_dm_channel(self, user_id: str) -> str:
        resp = self._api_post("conversations.open", {"users": user_id})
        channel_id = resp.get("channel", {}).get("id")
        if not channel_id:
            raise RuntimeError("conversations.open returned no channel id")
        return channel_id

    def notify_access_request(
        self, username: str, email: str, reason: str, req_id: int
    ) -> dict:
        ts = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        text = (
            "New user access request submitted.\n"
            f"request_id: {req_id}\n"
            f"username: {username}\n"
            f"email: {email}\n"
            f"reason: {(reason or '').strip() or '(empty)'}\n"
            f"time_utc: {ts}"
        )

        sent_to = []
        errors = []

        if self.config.channel_id:
            try:
                self._post_message(self.config.channel_id, text)
                sent_to.append("channel")
            except Exception as exc:
                errors.append(f"channel:{exc}")

        for dm_user_id in self.config.dm_user_ids:
            try:
                dm_channel = self._open_dm_channel(dm_user_id)
                self._post_message(dm_channel, text)
                sent_to.append(f"dm:{dm_user_id}")
            except Exception as exc:
                errors.append(f"dm:{dm_user_id}:{exc}")

        return {"sent_to": sent_to, "errors": errors}
