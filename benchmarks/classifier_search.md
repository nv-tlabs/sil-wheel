<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Classifier Search Benchmark

_commit `b9c102e` · date 2026-03-29 18:06:49 · classifier dir: /path/to/classifiers/ · size: 1.6 GB_

| embed_type | label                                         | expression    | cold (s) | warm (s) | count   |
| ---------- | --------------------------------------------- | ------------- | -------- | -------- | ------- |
| cosmos     | Homeless on the road                          | p > 0.5       | 0.0086   | 0.0006   | 4913    |
| cosmos     | Homeless on the road                          | p < 0.5       | 0.0100   | 0.0069   | 22703   |
| cosmos     | Homeless on the road                          | 0.3 < p < 0.7 | 0.0103   | 0.0029   | 27175   |
| cosmos     | Backing out of parking space                  | p > 0.5       | 0.3569   | 0.1204   | 599789  |
| cosmos     | Backing out of parking space                  | p < 0.5       | 0.2926   | 0.0533   | 264444  |
| cosmos     | Backing out of parking space                  | 0.3 < p < 0.7 | 0.3061   | 0.0817   | 422970  |
| cosmos     | Driving through a tunnel                      | p > 0.5       | 0.1562   | 0.0474   | 305619  |
| cosmos     | Driving through a tunnel                      | p < 0.5       | 0.1228   | 0.0176   | 130181  |
| cosmos     | Driving through a tunnel                      | 0.3 < p < 0.7 | 0.1245   | 0.0283   | 194728  |
| cosmos     | Bridge                                        | p > 0.5       | 0.0758   | 0.0137   | 118533  |
| cosmos     | Bridge                                        | p < 0.5       | 0.0698   | 0.0188   | 150028  |
| cosmos     | Bridge                                        | 0.3 < p < 0.7 | 0.0754   | 0.0270   | 207147  |
| cosmos     | Accelerate                                    | p > 0.5       | 2.9309   | 1.2121   | 7000000 |
| cosmos     | Accelerate                                    | p < 0.5       | 2.0051   | 0.0005   | 0       |
| cosmos     | Accelerate                                    | 0.3 < p < 0.7 | 1.8151   | 0.0004   | 0       |
| cosmos     | Barrier gate                                  | p > 0.5       | 0.6292   | 0.2216   | 1073475 |
| cosmos     | Barrier gate                                  | p < 0.5       | 0.5774   | 0.1129   | 558657  |
| cosmos     | Barrier gate                                  | 0.3 < p < 0.7 | 0.6026   | 0.1957   | 943033  |
| cosmos     | Being parked                                  | p > 0.5       | 0.1737   | 0.0374   | 289023  |
| cosmos     | Being parked                                  | p < 0.5       | 0.1478   | 0.0296   | 247109  |
| cosmos     | Being parked                                  | 0.3 < p < 0.7 | 0.1644   | 0.0534   | 390144  |
| cosmos     | interesting                                   | p > 0.5       | 2.8016   | 0.9815   | 5715308 |
| cosmos     | interesting                                   | p < 0.5       | 2.1501   | 0.2136   | 1284689 |
| cosmos     | interesting                                   | 0.3 < p < 0.7 | 2.3842   | 0.5444   | 3165470 |
| cosmos     | VRU crossing - pedestrian                     | p > 0.5       | 1.2800   | 0.4061   | 2258889 |
| cosmos     | VRU crossing - pedestrian                     | p < 0.5       | 1.0565   | 0.1778   | 995864  |
| cosmos     | VRU crossing - pedestrian                     | 0.3 < p < 0.7 | 1.1107   | 0.3094   | 1728257 |
| cosmos     | VRU crossing - animal                         | p > 0.5       | 0.8708   | 0.2557   | 1302195 |
| cosmos     | VRU crossing - animal                         | p < 0.5       | 0.7686   | 0.1845   | 958135  |
| cosmos     | VRU crossing - animal                         | 0.3 < p < 0.7 | 0.8443   | 0.2863   | 1516227 |
| cosmos     | Barrier gate&&Enter a tunnel                  | p > 0.5       | 0.8457   | 0.2455   | 1302214 |
| cosmos     | Barrier gate&&Enter a tunnel                  | p < 0.5       | 0.7166   | 0.1747   | 855324  |
| cosmos     | Barrier gate&&Enter a tunnel                  | 0.3 < p < 0.7 | 0.8078   | 0.2642   | 1422432 |
| caption    | VRU - moves with traffic - cyclist            | p > 0.5       | 0.0879   | 0.0154   | 116079  |
| caption    | VRU - moves with traffic - cyclist            | p < 0.5       | 0.0494   | 0.0113   | 82136   |
| caption    | VRU - moves with traffic - cyclist            | 0.3 < p < 0.7 | 0.0533   | 0.0170   | 130099  |
| caption    | Barrier gate                                  | p > 0.5       | 0.7331   | 0.2116   | 1004766 |
| caption    | Barrier gate                                  | p < 0.5       | 0.7034   | 0.1925   | 906110  |
| caption    | Barrier gate                                  | 0.3 < p < 0.7 | 0.7610   | 0.2796   | 1401453 |
| caption    | VRU crossing - animal&&VRU crossing - cyclist | p > 0.5       | 0.2169   | 0.0394   | 258849  |
| caption    | VRU crossing - animal&&VRU crossing - cyclist | p < 0.5       | 0.1755   | 0.0453   | 293091  |
| caption    | VRU crossing - animal&&VRU crossing - cyclist | 0.3 < p < 0.7 | 0.1815   | 0.0620   | 423163  |
| caption    | VRU crossing - animal                         | p > 0.5       | 0.2020   | 0.0347   | 222215  |
| caption    | VRU crossing - animal                         | p < 0.5       | 0.2170   | 0.0644   | 435159  |
| caption    | VRU crossing - animal                         | 0.3 < p < 0.7 | 0.2434   | 0.0879   | 564071  |