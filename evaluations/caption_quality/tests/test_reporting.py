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

from reporting import write_markdown


def test_write_markdown_table_and_metadata(tmp_path):
    out = tmp_path / "out.md"
    write_markdown(
        str(out),
        "Caption Quality — nlg",
        headers=["group", "n", "bleu4"],
        rows=[("all", "3", "0.300"), ("x", "2", "0.500")],
        metadata={"reference_model": "ref", "prediction_model": "pred"},
    )
    text = out.read_text()
    assert "# Caption Quality — nlg" in text
    assert "reference_model: ref" in text
    assert "prediction_model: pred" in text
    # Header row, separator row, and both data rows present.
    assert "| group" in text
    assert "| all" in text and "| x" in text
    assert "0.500" in text


def test_write_markdown_append_adds_section(tmp_path):
    out = tmp_path / "out.md"
    write_markdown(str(out), "First", ["a"], [("1",)])
    write_markdown(str(out), "Second", ["a"], [("2",)], append=True)
    text = out.read_text()
    assert "# First" in text
    assert "# Second" in text
    # Append must not truncate the first section.
    assert text.index("# First") < text.index("# Second")
