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

class LRUDict(dict):
    def __init__(self, *args, **kwargs):
        self.cache_size = kwargs.pop("size", 10)
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if key in self:
            self.pop(key)

        if len(self) >= self.cache_size:
            first_key = next(iter(self))
            self.pop(first_key)

        super().__setitem__(key, value)

    def __getitem__(self, key):
        value = self.pop(key)
        super().__setitem__(key, value)
        return value

    def keys(self):
        return list(super().keys())

    def values(self):
        return list(super().values())

    def items(self):
        return list(super().items())
