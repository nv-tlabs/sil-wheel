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

import httplib2
from string import ascii_uppercase
from googleapiclient import discovery
from oauth2client.service_account import ServiceAccountCredentials


def get_credentials(scopes, credential_path=".credentials"):
    return ServiceAccountCredentials.from_json_keyfile_name(
        credential_path,
        scopes
    )


def get_spreadsheets(credential_path=".credentials"):
    credentials = get_credentials(
        ["https://www.googleapis.com/auth/spreadsheets"],
        credential_path
    )
    assert credentials, "No credentials found"
    assert not credentials.invalid, "The credentials are invalid"

    http = credentials.authorize(httplib2.Http())
    service = discovery.build(
        "sheets",
        "v4",
        http=http,
        discoveryServiceUrl=("https://sheets.googleapis.com/$discovery/rest?"
                             "version=v4")
    )

    return service


def append_to_spreadsheet(spreadsheet_id, sheet, data,
                          credential_path=".credentials"):
    if len(data) == 0:
        return

    sheet_range = "%s!A1:%s1" % (
        sheet,
        ascii_uppercase[max(map(len, data)) - 1]
    )

    sheets = get_spreadsheets(credential_path)
    return sheets.spreadsheets().values().append(
        spreadsheetId=spreadsheet_id,
        range=sheet_range,
        body={
            "values": data,
            "majorDimension": "ROWS"
        },
        valueInputOption="USER_ENTERED"
    ).execute()
