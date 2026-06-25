<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Bug Report Configuration

Users can submit bug reports from the annotation and leaderboard pages via a button in the footer. Reports are appended to a Google Spreadsheet using a service account.

**If it is not configured, the feature is silently disabled.** The server starts
normally and bug reports are only written to the server log.

1. Create a Google Cloud service account and download its JSON key file.
2. Share your target spreadsheet with the service account's email (Editor access).
3. Add the following to your server YAML config under `server`:

```yaml
server:
  bug_report:
    spreadsheet_id: "your-spreadsheet-id"
    credential_path: "/path/to/service-account.json"
```

The `spreadsheet_id` is the long ID in the spreadsheet URL:
`https://docs.google.com/spreadsheets/d/<spreadsheet_id>/edit`

Each submitted report appends a row to a sheet named `Bug Reports` with columns:
`Timestamp | Username | Title | Description | Browser/OS`