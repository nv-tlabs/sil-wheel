<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Caption Search Benchmark

_commit `e502a81` · date 2026-04-02 18:56:05 · db: /path/to/wheel-data/captions_schema_latest.db · db size: 226.7 GB_

| query                     | dataset                        | seconds | count | limit |
| ------------------------- | ------------------------------ | ------- | ----- | ----- |
| turn                      | <all>                          | 2.573   | 5000  | 5000  |
| turn                      | Nexar                          | 0.525   | 5000  | 5000  |
| turn                      | Physical AI                    | 0.236   | 5000  | 5000  |
| turn                      | AV V2 train + AV V2 validation | 1.881   | 5000  | 5000  |
| brake                     | <all>                          | 1.071   | 5000  | 5000  |
| brake                     | Nexar                          | 0.380   | 5000  | 5000  |
| brake                     | Physical AI                    | 0.115   | 5000  | 5000  |
| brake                     | AV V2 train + AV V2 validation | 0.916   | 5000  | 5000  |
| stop                      | <all>                          | 3.925   | 5000  | 5000  |
| stop                      | Nexar                          | 0.721   | 5000  | 5000  |
| stop                      | Physical AI                    | 0.261   | 5000  | 5000  |
| stop                      | AV V2 train + AV V2 validation | 2.784   | 5000  | 5000  |
| intersection              | <all>                          | 6.507   | 5000  | 5000  |
| intersection              | Nexar                          | 0.996   | 5000  | 5000  |
| intersection              | Physical AI                    | 0.360   | 5000  | 5000  |
| intersection              | AV V2 train + AV V2 validation | 4.709   | 5000  | 5000  |
| pedestrian                | <all>                          | 4.116   | 5000  | 5000  |
| pedestrian                | Nexar                          | 0.666   | 5000  | 5000  |
| pedestrian                | Physical AI                    | 0.265   | 5000  | 5000  |
| pedestrian                | AV V2 train + AV V2 validation | 3.117   | 5000  | 5000  |
| highway                   | <all>                          | 4.747   | 5000  | 5000  |
| highway                   | Nexar                          | 0.651   | 5000  | 5000  |
| highway                   | Physical AI                    | 0.250   | 5000  | 5000  |
| highway                   | AV V2 train + AV V2 validation | 3.792   | 5000  | 5000  |
| "red light"               | <all>                          | 3.023   | 5000  | 5000  |
| "red light"               | Nexar                          | 1.393   | 5000  | 5000  |
| "red light"               | Physical AI                    | 0.983   | 5000  | 5000  |
| "red light"               | AV V2 train + AV V2 validation | 2.409   | 5000  | 5000  |
| "left turn"               | <all>                          | 1.522   | 5000  | 5000  |
| "left turn"               | Nexar                          | 0.775   | 5000  | 5000  |
| "left turn"               | Physical AI                    | 0.588   | 5000  | 5000  |
| "left turn"               | AV V2 train + AV V2 validation | 1.435   | 5000  | 5000  |
| "4-way stop intersection" | <all>                          | 0.237   | 4     | 5000  |
| "4-way stop intersection" | Nexar                          | 0.109   | 0     | 5000  |
| "4-way stop intersection" | Physical AI                    | 0.176   | 2     | 5000  |
| "4-way stop intersection" | AV V2 train + AV V2 validation | 0.399   | 3     | 5000  |
| "nudge left"              | <all>                          | 0.001   | 0     | 5000  |
| "nudge left"              | Nexar                          | 0.000   | 0     | 5000  |
| "nudge left"              | Physical AI                    | 0.001   | 0     | 5000  |
| "nudge left"              | AV V2 train + AV V2 validation | 0.001   | 0     | 5000  |
| nudge                     | <all>                          | 0.001   | 36    | 5000  |
| nudge                     | Nexar                          | 0.023   | 2     | 5000  |
| nudge                     | Physical AI                    | 0.000   | 0     | 5000  |
| nudge                     | AV V2 train + AV V2 validation | 0.142   | 4     | 5000  |
| sway                      | <all>                          | 0.466   | 5000  | 5000  |
| sway                      | Nexar                          | 0.189   | 5000  | 5000  |
| sway                      | Physical AI                    | 0.060   | 5000  | 5000  |
| sway                      | AV V2 train + AV V2 validation | 0.512   | 5000  | 5000  |
| fishtail                  | <all>                          | 0.000   | 9     | 5000  |
| fishtail                  | Nexar                          | 0.023   | 1     | 5000  |
| fishtail                  | Physical AI                    | 0.000   | 0     | 5000  |
| fishtail                  | AV V2 train + AV V2 validation | 0.146   | 7     | 5000  |
| hydroplane                | <all>                          | 0.002   | 145   | 5000  |
| hydroplane                | Nexar                          | 0.022   | 15    | 5000  |
| hydroplane                | Physical AI                    | 0.007   | 9     | 5000  |
| hydroplane                | AV V2 train + AV V2 validation | 0.137   | 111   | 5000  |
| jackknife                 | <all>                          | 0.000   | 4     | 5000  |
| jackknife                 | Nexar                          | 0.021   | 2     | 5000  |
| jackknife                 | Physical AI                    | 0.000   | 0     | 5000  |
| jackknife                 | AV V2 train + AV V2 validation | 0.134   | 2     | 5000  |
| chicane                   | <all>                          | 0.000   | 7     | 5000  |
| chicane                   | Nexar                          | 0.000   | 0     | 5000  |
| chicane                   | Physical AI                    | 0.006   | 5     | 5000  |
| chicane                   | AV V2 train + AV V2 validation | 0.133   | 1     | 5000  |
| pothole                   | <all>                          | 0.019   | 2646  | 5000  |
| pothole                   | Nexar                          | 0.025   | 174   | 5000  |
| pothole                   | Physical AI                    | 0.007   | 43    | 5000  |
| pothole                   | AV V2 train + AV V2 validation | 0.153   | 1766  | 5000  |
| glare                     | <all>                          | 0.593   | 5000  | 5000  |
| glare                     | Nexar                          | 0.221   | 5000  | 5000  |
| glare                     | Physical AI                    | 0.092   | 5000  | 5000  |
| glare                     | AV V2 train + AV V2 validation | 0.660   | 5000  | 5000  |
| tailgating                | <all>                          | 0.053   | 5000  | 5000  |
| tailgating                | Nexar                          | 0.030   | 221   | 5000  |
| tailgating                | Physical AI                    | 0.015   | 877   | 5000  |
| tailgating                | AV V2 train + AV V2 validation | 0.217   | 5000  | 5000  |
| contraflow                | <all>                          | 0.000   | 15    | 5000  |
| contraflow                | Nexar                          | 0.023   | 2     | 5000  |
| contraflow                | Physical AI                    | 0.000   | 0     | 5000  |
| contraflow                | AV V2 train + AV V2 validation | 0.135   | 8     | 5000  |

# Caption Rewrite-Bundle Benchmark

_commit `e502a81` · date 2026-04-02 18:56:05 · db: /path/to/wheel-data/captions_schema_latest.db · db size: 226.7 GB · rewrites column: queries in each bundle separated by \|_

| label               | rewrites                                                                                           | n_rewrites | dataset                        | base_secs | base_count | rewrite_secs | rewrite_count |
| ------------------- | -------------------------------------------------------------------------------------------------- | ---------- | ------------------------------ | --------- | ---------- | ------------ | ------------- |
| pedestrian crossing | pedestrian crossing \| person crossing \| crossing pedestrian \| street crossing \| walking across | 5          | <all>                          | 2.728     | 5000       | 10.459       | 5000          |
| pedestrian crossing | pedestrian crossing \| person crossing \| crossing pedestrian \| street crossing \| walking across | 5          | Nexar                          | 1.805     | 5000       | 3.825        | 5000          |
| pedestrian crossing | pedestrian crossing \| person crossing \| crossing pedestrian \| street crossing \| walking across | 5          | Physical AI                    | 0.794     | 5000       | 2.468        | 5000          |
| pedestrian crossing | pedestrian crossing \| person crossing \| crossing pedestrian \| street crossing \| walking across | 5          | AV V2 train + AV V2 validation | 2.668     | 5000       | 5.137        | 5000          |
| sharp left turn     | sharp left turn \| left turn \| turning left \| hard left \| steering left                         | 5          | <all>                          | 0.798     | 1928       | 3.980        | 5000          |
| sharp left turn     | sharp left turn \| left turn \| turning left \| hard left \| steering left                         | 5          | Nexar                          | 0.194     | 84         | 2.464        | 5000          |
| sharp left turn     | sharp left turn \| left turn \| turning left \| hard left \| steering left                         | 5          | Physical AI                    | 0.172     | 316        | 1.604        | 5000          |
| sharp left turn     | sharp left turn \| left turn \| turning left \| hard left \| steering left                         | 5          | AV V2 train + AV V2 validation | 0.360     | 1202       | 2.597        | 5000          |
| highway driving     | highway driving \| motorway \| freeway \| high speed \| expressway                                 | 5          | <all>                          | 4.795     | 5000       | 6.278        | 5000          |
| highway driving     | highway driving \| motorway \| freeway \| high speed \| expressway                                 | 5          | Nexar                          | 2.282     | 5000       | 2.765        | 5000          |
| highway driving     | highway driving \| motorway \| freeway \| high speed \| expressway                                 | 5          | Physical AI                    | 1.465     | 5000       | 1.596        | 5000          |
| highway driving     | highway driving \| motorway \| freeway \| high speed \| expressway                                 | 5          | AV V2 train + AV V2 validation | 2.587     | 5000       | 2.730        | 5000          |
| emergency vehicle   | emergency vehicle \| ambulance \| fire truck \| police car \| siren                                | 5          | <all>                          | 4.691     | 5000       | 5.211        | 5000          |
| emergency vehicle   | emergency vehicle \| ambulance \| fire truck \| police car \| siren                                | 5          | Nexar                          | 1.294     | 5000       | 1.525        | 5000          |
| emergency vehicle   | emergency vehicle \| ambulance \| fire truck \| police car \| siren                                | 5          | Physical AI                    | 0.320     | 301        | 0.995        | 1952          |
| emergency vehicle   | emergency vehicle \| ambulance \| fire truck \| police car \| siren                                | 5          | AV V2 train + AV V2 validation | 0.898     | 5000       | 1.541        | 5000          |
| construction zone   | construction zone \| road works \| lane shift \| work zone \| road construction                    | 5          | <all>                          | 2.024     | 5000       | 10.086       | 5000          |
| construction zone   | construction zone \| road works \| lane shift \| work zone \| road construction                    | 5          | Nexar                          | 1.658     | 5000       | 2.542        | 5000          |
| construction zone   | construction zone \| road works \| lane shift \| work zone \| road construction                    | 5          | Physical AI                    | 0.909     | 5000       | 1.352        | 5000          |
| construction zone   | construction zone \| road works \| lane shift \| work zone \| road construction                    | 5          | AV V2 train + AV V2 validation | 1.529     | 5000       | 2.359        | 5000          |

# Caption Search Limit-Sweep Benchmark

_commit `e502a81` · date 2026-04-02 18:56:05 · db: /path/to/wheel-data/captions_schema_latest.db · db size: 226.7 GB_

| query                     | dataset                        | limit  | seconds | count  |
| ------------------------- | ------------------------------ | ------ | ------- | ------ |
| turn                      | <all>                          | 5000   | 2.727   | 5000   |
| turn                      | <all>                          | 10000  | 3.714   | 10000  |
| turn                      | <all>                          | 30000  | 7.890   | 30000  |
| turn                      | <all>                          | 50000  | 8.342   | 50000  |
| turn                      | <all>                          | 100000 | 17.013  | 100000 |
| turn                      | <all>                          | 200000 | 30.888  | 200000 |
| turn                      | <all>                          | 500000 | 72.308  | 500000 |
| turn                      | Nexar                          | 5000   | 0.579   | 5000   |
| turn                      | Nexar                          | 10000  | 1.390   | 10000  |
| turn                      | Nexar                          | 30000  | 4.270   | 30000  |
| turn                      | Nexar                          | 50000  | 5.106   | 50000  |
| turn                      | Nexar                          | 100000 | 13.048  | 100000 |
| turn                      | Nexar                          | 200000 | 25.026  | 200000 |
| turn                      | Nexar                          | 500000 | 69.079  | 453922 |
| turn                      | Physical AI                    | 5000   | 0.263   | 5000   |
| turn                      | Physical AI                    | 10000  | 0.313   | 10000  |
| turn                      | Physical AI                    | 30000  | 0.617   | 30000  |
| turn                      | Physical AI                    | 50000  | 1.819   | 50000  |
| turn                      | Physical AI                    | 100000 | 4.762   | 100000 |
| turn                      | Physical AI                    | 200000 | 3.647   | 111101 |
| turn                      | Physical AI                    | 500000 | 0.814   | 111101 |
| turn                      | AV V2 train + AV V2 validation | 5000   | 1.859   | 5000   |
| turn                      | AV V2 train + AV V2 validation | 10000  | 2.094   | 10000  |
| turn                      | AV V2 train + AV V2 validation | 30000  | 3.359   | 30000  |
| turn                      | AV V2 train + AV V2 validation | 50000  | 4.699   | 50000  |
| turn                      | AV V2 train + AV V2 validation | 100000 | 9.159   | 100000 |
| turn                      | AV V2 train + AV V2 validation | 200000 | 17.412  | 200000 |
| turn                      | AV V2 train + AV V2 validation | 500000 | 12.980  | 500000 |
| brake                     | <all>                          | 5000   | 1.103   | 5000   |
| brake                     | <all>                          | 10000  | 1.709   | 10000  |
| brake                     | <all>                          | 30000  | 3.809   | 30000  |
| brake                     | <all>                          | 50000  | 4.171   | 50000  |
| brake                     | <all>                          | 100000 | 7.914   | 100000 |
| brake                     | <all>                          | 200000 | 15.804  | 200000 |
| brake                     | <all>                          | 500000 | 39.229  | 500000 |
| brake                     | Nexar                          | 5000   | 0.412   | 5000   |
| brake                     | Nexar                          | 10000  | 0.414   | 10000  |
| brake                     | Nexar                          | 30000  | 0.504   | 30000  |
| brake                     | Nexar                          | 50000  | 0.602   | 50000  |
| brake                     | Nexar                          | 100000 | 1.940   | 100000 |
| brake                     | Nexar                          | 200000 | 11.561  | 200000 |
| brake                     | Nexar                          | 500000 | 10.477  | 286495 |
| brake                     | Physical AI                    | 5000   | 0.126   | 5000   |
| brake                     | Physical AI                    | 10000  | 0.283   | 10000  |
| brake                     | Physical AI                    | 30000  | 1.066   | 30000  |
| brake                     | Physical AI                    | 50000  | 0.281   | 30660  |
| brake                     | Physical AI                    | 100000 | 0.260   | 30660  |
| brake                     | Physical AI                    | 200000 | 0.256   | 30660  |
| brake                     | Physical AI                    | 500000 | 0.271   | 30660  |
| brake                     | AV V2 train + AV V2 validation | 5000   | 0.952   | 5000   |
| brake                     | AV V2 train + AV V2 validation | 10000  | 0.952   | 10000  |
| brake                     | AV V2 train + AV V2 validation | 30000  | 1.060   | 30000  |
| brake                     | AV V2 train + AV V2 validation | 50000  | 1.654   | 50000  |
| brake                     | AV V2 train + AV V2 validation | 100000 | 6.147   | 100000 |
| brake                     | AV V2 train + AV V2 validation | 200000 | 3.510   | 200000 |
| brake                     | AV V2 train + AV V2 validation | 500000 | 18.626  | 500000 |
| stop                      | <all>                          | 5000   | 3.956   | 5000   |
| stop                      | <all>                          | 10000  | 4.260   | 10000  |
| stop                      | <all>                          | 30000  | 5.230   | 30000  |
| stop                      | <all>                          | 50000  | 5.607   | 50000  |
| stop                      | <all>                          | 100000 | 8.725   | 100000 |
| stop                      | <all>                          | 200000 | 14.483  | 200000 |
| stop                      | <all>                          | 500000 | 35.565  | 500000 |
| stop                      | Nexar                          | 5000   | 0.772   | 5000   |
| stop                      | Nexar                          | 10000  | 0.851   | 10000  |
| stop                      | Nexar                          | 30000  | 1.645   | 30000  |
| stop                      | Nexar                          | 50000  | 1.945   | 50000  |
| stop                      | Nexar                          | 100000 | 3.252   | 100000 |
| stop                      | Nexar                          | 200000 | 6.885   | 200000 |
| stop                      | Nexar                          | 500000 | 18.572  | 500000 |
| stop                      | Physical AI                    | 5000   | 0.271   | 5000   |
| stop                      | Physical AI                    | 10000  | 0.321   | 10000  |
| stop                      | Physical AI                    | 30000  | 0.609   | 30000  |
| stop                      | Physical AI                    | 50000  | 1.072   | 50000  |
| stop                      | Physical AI                    | 100000 | 1.592   | 100000 |
| stop                      | Physical AI                    | 200000 | 0.785   | 103086 |
| stop                      | Physical AI                    | 500000 | 0.780   | 103086 |
| stop                      | AV V2 train + AV V2 validation | 5000   | 2.960   | 5000   |
| stop                      | AV V2 train + AV V2 validation | 10000  | 2.990   | 10000  |
| stop                      | AV V2 train + AV V2 validation | 30000  | 3.329   | 30000  |
| stop                      | AV V2 train + AV V2 validation | 50000  | 3.459   | 50000  |
| stop                      | AV V2 train + AV V2 validation | 100000 | 5.622   | 100000 |
| stop                      | AV V2 train + AV V2 validation | 200000 | 8.824   | 200000 |
| stop                      | AV V2 train + AV V2 validation | 500000 | 15.303  | 500000 |
| intersection              | <all>                          | 5000   | 6.464   | 5000   |
| intersection              | <all>                          | 10000  | 7.003   | 10000  |
| intersection              | <all>                          | 30000  | 7.934   | 30000  |
| intersection              | <all>                          | 50000  | 8.006   | 50000  |
| intersection              | <all>                          | 100000 | 10.078  | 100000 |
| intersection              | <all>                          | 200000 | 13.392  | 200000 |
| intersection              | <all>                          | 500000 | 25.171  | 500000 |
| intersection              | Nexar                          | 5000   | 1.050   | 5000   |
| intersection              | Nexar                          | 10000  | 1.093   | 10000  |
| intersection              | Nexar                          | 30000  | 1.295   | 30000  |
| intersection              | Nexar                          | 50000  | 1.443   | 50000  |
| intersection              | Nexar                          | 100000 | 1.961   | 100000 |
| intersection              | Nexar                          | 200000 | 3.113   | 200000 |
| intersection              | Nexar                          | 500000 | 9.132   | 500000 |
| intersection              | Physical AI                    | 5000   | 0.386   | 5000   |
| intersection              | Physical AI                    | 10000  | 0.445   | 10000  |
| intersection              | Physical AI                    | 30000  | 0.589   | 30000  |
| intersection              | Physical AI                    | 50000  | 0.678   | 50000  |
| intersection              | Physical AI                    | 100000 | 1.595   | 100000 |
| intersection              | Physical AI                    | 200000 | 1.055   | 136463 |
| intersection              | Physical AI                    | 500000 | 1.028   | 136463 |
| intersection              | AV V2 train + AV V2 validation | 5000   | 4.824   | 5000   |
| intersection              | AV V2 train + AV V2 validation | 10000  | 4.895   | 10000  |
| intersection              | AV V2 train + AV V2 validation | 30000  | 5.231   | 30000  |
| intersection              | AV V2 train + AV V2 validation | 50000  | 5.282   | 50000  |
| intersection              | AV V2 train + AV V2 validation | 100000 | 6.035   | 100000 |
| intersection              | AV V2 train + AV V2 validation | 200000 | 7.478   | 200000 |
| intersection              | AV V2 train + AV V2 validation | 500000 | 14.786  | 500000 |
| pedestrian                | <all>                          | 5000   | 4.150   | 5000   |
| pedestrian                | <all>                          | 10000  | 4.434   | 10000  |
| pedestrian                | <all>                          | 30000  | 5.216   | 30000  |
| pedestrian                | <all>                          | 50000  | 5.643   | 50000  |
| pedestrian                | <all>                          | 100000 | 7.217   | 100000 |
| pedestrian                | <all>                          | 200000 | 10.577  | 200000 |
| pedestrian                | <all>                          | 500000 | 24.053  | 500000 |
| pedestrian                | Nexar                          | 5000   | 0.724   | 5000   |
| pedestrian                | Nexar                          | 10000  | 0.726   | 10000  |
| pedestrian                | Nexar                          | 30000  | 0.828   | 30000  |
| pedestrian                | Nexar                          | 50000  | 0.945   | 50000  |
| pedestrian                | Nexar                          | 100000 | 1.291   | 100000 |
| pedestrian                | Nexar                          | 200000 | 2.380   | 200000 |
| pedestrian                | Nexar                          | 500000 | 9.094   | 500000 |
| pedestrian                | Physical AI                    | 5000   | 0.272   | 5000   |
| pedestrian                | Physical AI                    | 10000  | 0.315   | 10000  |
| pedestrian                | Physical AI                    | 30000  | 0.452   | 30000  |
| pedestrian                | Physical AI                    | 50000  | 0.608   | 50000  |
| pedestrian                | Physical AI                    | 100000 | 1.056   | 100000 |
| pedestrian                | Physical AI                    | 200000 | 0.775   | 108170 |
| pedestrian                | Physical AI                    | 500000 | 0.781   | 108170 |
| pedestrian                | AV V2 train + AV V2 validation | 5000   | 2.896   | 5000   |
| pedestrian                | AV V2 train + AV V2 validation | 10000  | 2.970   | 10000  |
| pedestrian                | AV V2 train + AV V2 validation | 30000  | 3.165   | 30000  |
| pedestrian                | AV V2 train + AV V2 validation | 50000  | 3.187   | 50000  |
| pedestrian                | AV V2 train + AV V2 validation | 100000 | 3.775   | 100000 |
| pedestrian                | AV V2 train + AV V2 validation | 200000 | 4.943   | 200000 |
| pedestrian                | AV V2 train + AV V2 validation | 500000 | 8.747   | 500000 |
| highway                   | <all>                          | 5000   | 4.744   | 5000   |
| highway                   | <all>                          | 10000  | 5.233   | 10000  |
| highway                   | <all>                          | 30000  | 6.365   | 30000  |
| highway                   | <all>                          | 50000  | 6.405   | 50000  |
| highway                   | <all>                          | 100000 | 8.915   | 100000 |
| highway                   | <all>                          | 200000 | 12.404  | 200000 |
| highway                   | <all>                          | 500000 | 25.546  | 500000 |
| highway                   | Nexar                          | 5000   | 0.709   | 5000   |
| highway                   | Nexar                          | 10000  | 0.755   | 10000  |
| highway                   | Nexar                          | 30000  | 1.114   | 30000  |
| highway                   | Nexar                          | 50000  | 1.277   | 50000  |
| highway                   | Nexar                          | 100000 | 1.870   | 100000 |
| highway                   | Nexar                          | 200000 | 3.711   | 200000 |
| highway                   | Nexar                          | 500000 | 8.302   | 484279 |
| highway                   | Physical AI                    | 5000   | 0.258   | 5000   |
| highway                   | Physical AI                    | 10000  | 0.351   | 10000  |
| highway                   | Physical AI                    | 30000  | 0.745   | 30000  |
| highway                   | Physical AI                    | 50000  | 0.911   | 50000  |
| highway                   | Physical AI                    | 100000 | 0.609   | 66344  |
| highway                   | Physical AI                    | 200000 | 0.548   | 66344  |
| highway                   | Physical AI                    | 500000 | 0.541   | 66344  |
| highway                   | AV V2 train + AV V2 validation | 5000   | 3.882   | 5000   |
| highway                   | AV V2 train + AV V2 validation | 10000  | 4.043   | 10000  |
| highway                   | AV V2 train + AV V2 validation | 30000  | 4.429   | 30000  |
| highway                   | AV V2 train + AV V2 validation | 50000  | 4.555   | 50000  |
| highway                   | AV V2 train + AV V2 validation | 100000 | 5.511   | 100000 |
| highway                   | AV V2 train + AV V2 validation | 200000 | 7.350   | 200000 |
| highway                   | AV V2 train + AV V2 validation | 500000 | 11.276  | 500000 |
| "red light"               | <all>                          | 5000   | 3.051   | 5000   |
| "red light"               | <all>                          | 10000  | 3.132   | 10000  |
| "red light"               | <all>                          | 30000  | 3.481   | 30000  |
| "red light"               | <all>                          | 50000  | 3.561   | 50000  |
| "red light"               | <all>                          | 100000 | 4.522   | 100000 |
| "red light"               | <all>                          | 200000 | 5.973   | 200000 |
| "red light"               | <all>                          | 500000 | 11.325  | 500000 |
| "red light"               | Nexar                          | 5000   | 1.426   | 5000   |
| "red light"               | Nexar                          | 10000  | 1.426   | 10000  |
| "red light"               | Nexar                          | 30000  | 1.535   | 30000  |
| "red light"               | Nexar                          | 50000  | 1.638   | 50000  |
| "red light"               | Nexar                          | 100000 | 1.958   | 100000 |
| "red light"               | Nexar                          | 200000 | 2.526   | 200000 |
| "red light"               | Nexar                          | 500000 | 3.526   | 327662 |
| "red light"               | Physical AI                    | 5000   | 0.988   | 5000   |
| "red light"               | Physical AI                    | 10000  | 1.013   | 10000  |
| "red light"               | Physical AI                    | 30000  | 1.138   | 30000  |
| "red light"               | Physical AI                    | 50000  | 1.182   | 40436  |
| "red light"               | Physical AI                    | 100000 | 1.179   | 40436  |
| "red light"               | Physical AI                    | 200000 | 1.205   | 40436  |
| "red light"               | Physical AI                    | 500000 | 1.178   | 40436  |
| "red light"               | AV V2 train + AV V2 validation | 5000   | 2.439   | 5000   |
| "red light"               | AV V2 train + AV V2 validation | 10000  | 2.452   | 10000  |
| "red light"               | AV V2 train + AV V2 validation | 30000  | 2.567   | 30000  |
| "red light"               | AV V2 train + AV V2 validation | 50000  | 2.672   | 50000  |
| "red light"               | AV V2 train + AV V2 validation | 100000 | 2.943   | 100000 |
| "red light"               | AV V2 train + AV V2 validation | 200000 | 3.475   | 200000 |
| "red light"               | AV V2 train + AV V2 validation | 500000 | 7.865   | 500000 |
| "left turn"               | <all>                          | 5000   | 1.585   | 5000   |
| "left turn"               | <all>                          | 10000  | 1.571   | 10000  |
| "left turn"               | <all>                          | 30000  | 1.715   | 30000  |
| "left turn"               | <all>                          | 50000  | 1.822   | 50000  |
| "left turn"               | <all>                          | 100000 | 2.110   | 100000 |
| "left turn"               | <all>                          | 200000 | 3.582   | 200000 |
| "left turn"               | <all>                          | 500000 | 13.026  | 500000 |
| "left turn"               | Nexar                          | 5000   | 0.815   | 5000   |
| "left turn"               | Nexar                          | 10000  | 0.804   | 10000  |
| "left turn"               | Nexar                          | 30000  | 0.914   | 30000  |
| "left turn"               | Nexar                          | 50000  | 1.015   | 50000  |
| "left turn"               | Nexar                          | 100000 | 1.306   | 100000 |
| "left turn"               | Nexar                          | 200000 | 1.527   | 133917 |
| "left turn"               | Nexar                          | 500000 | 1.509   | 133917 |
| "left turn"               | Physical AI                    | 5000   | 0.585   | 5000   |
| "left turn"               | Physical AI                    | 10000  | 0.602   | 10000  |
| "left turn"               | Physical AI                    | 30000  | 0.706   | 30000  |
| "left turn"               | Physical AI                    | 50000  | 0.832   | 50000  |
| "left turn"               | Physical AI                    | 100000 | 0.829   | 50395  |
| "left turn"               | Physical AI                    | 200000 | 0.829   | 50395  |
| "left turn"               | Physical AI                    | 500000 | 0.831   | 50395  |
| "left turn"               | AV V2 train + AV V2 validation | 5000   | 1.448   | 5000   |
| "left turn"               | AV V2 train + AV V2 validation | 10000  | 1.465   | 10000  |
| "left turn"               | AV V2 train + AV V2 validation | 30000  | 1.586   | 30000  |
| "left turn"               | AV V2 train + AV V2 validation | 50000  | 1.681   | 50000  |
| "left turn"               | AV V2 train + AV V2 validation | 100000 | 1.968   | 100000 |
| "left turn"               | AV V2 train + AV V2 validation | 200000 | 2.549   | 200000 |
| "left turn"               | AV V2 train + AV V2 validation | 500000 | 6.774   | 475039 |
| "4-way stop intersection" | <all>                          | 5000   | 0.259   | 4      |
| "4-way stop intersection" | <all>                          | 10000  | 0.211   | 4      |
| "4-way stop intersection" | <all>                          | 30000  | 0.212   | 4      |
| "4-way stop intersection" | <all>                          | 50000  | 0.213   | 4      |
| "4-way stop intersection" | <all>                          | 100000 | 0.210   | 4      |
| "4-way stop intersection" | <all>                          | 200000 | 0.210   | 4      |
| "4-way stop intersection" | <all>                          | 500000 | 0.209   | 4      |
| "4-way stop intersection" | Nexar                          | 5000   | 0.110   | 0      |
| "4-way stop intersection" | Nexar                          | 10000  | 0.095   | 0      |
| "4-way stop intersection" | Nexar                          | 30000  | 0.093   | 0      |
| "4-way stop intersection" | Nexar                          | 50000  | 0.092   | 0      |
| "4-way stop intersection" | Nexar                          | 100000 | 0.092   | 0      |
| "4-way stop intersection" | Nexar                          | 200000 | 0.092   | 0      |
| "4-way stop intersection" | Nexar                          | 500000 | 0.093   | 0      |
| "4-way stop intersection" | Physical AI                    | 5000   | 0.180   | 2      |
| "4-way stop intersection" | Physical AI                    | 10000  | 0.179   | 2      |
| "4-way stop intersection" | Physical AI                    | 30000  | 0.179   | 2      |
| "4-way stop intersection" | Physical AI                    | 50000  | 0.187   | 2      |
| "4-way stop intersection" | Physical AI                    | 100000 | 0.181   | 2      |
| "4-way stop intersection" | Physical AI                    | 200000 | 0.180   | 2      |
| "4-way stop intersection" | Physical AI                    | 500000 | 0.179   | 2      |
| "4-way stop intersection" | AV V2 train + AV V2 validation | 5000   | 0.406   | 3      |
| "4-way stop intersection" | AV V2 train + AV V2 validation | 10000  | 0.406   | 3      |
| "4-way stop intersection" | AV V2 train + AV V2 validation | 30000  | 0.409   | 3      |
| "4-way stop intersection" | AV V2 train + AV V2 validation | 50000  | 0.413   | 3      |
| "4-way stop intersection" | AV V2 train + AV V2 validation | 100000 | 0.408   | 3      |
| "4-way stop intersection" | AV V2 train + AV V2 validation | 200000 | 0.406   | 3      |
| "4-way stop intersection" | AV V2 train + AV V2 validation | 500000 | 0.408   | 3      |
| "nudge left"              | <all>                          | 5000   | 0.001   | 0      |
| "nudge left"              | <all>                          | 10000  | 0.000   | 0      |
| "nudge left"              | <all>                          | 30000  | 0.000   | 0      |
| "nudge left"              | <all>                          | 50000  | 0.000   | 0      |
| "nudge left"              | <all>                          | 100000 | 0.000   | 0      |
| "nudge left"              | <all>                          | 200000 | 0.000   | 0      |
| "nudge left"              | <all>                          | 500000 | 0.000   | 0      |
| "nudge left"              | Nexar                          | 5000   | 0.000   | 0      |
| "nudge left"              | Nexar                          | 10000  | 0.000   | 0      |
| "nudge left"              | Nexar                          | 30000  | 0.000   | 0      |
| "nudge left"              | Nexar                          | 50000  | 0.000   | 0      |
| "nudge left"              | Nexar                          | 100000 | 0.000   | 0      |
| "nudge left"              | Nexar                          | 200000 | 0.000   | 0      |
| "nudge left"              | Nexar                          | 500000 | 0.000   | 0      |
| "nudge left"              | Physical AI                    | 5000   | 0.001   | 0      |
| "nudge left"              | Physical AI                    | 10000  | 0.000   | 0      |
| "nudge left"              | Physical AI                    | 30000  | 0.000   | 0      |
| "nudge left"              | Physical AI                    | 50000  | 0.000   | 0      |
| "nudge left"              | Physical AI                    | 100000 | 0.000   | 0      |
| "nudge left"              | Physical AI                    | 200000 | 0.000   | 0      |
| "nudge left"              | Physical AI                    | 500000 | 0.000   | 0      |
| "nudge left"              | AV V2 train + AV V2 validation | 5000   | 0.001   | 0      |
| "nudge left"              | AV V2 train + AV V2 validation | 10000  | 0.001   | 0      |
| "nudge left"              | AV V2 train + AV V2 validation | 30000  | 0.001   | 0      |
| "nudge left"              | AV V2 train + AV V2 validation | 50000  | 0.001   | 0      |
| "nudge left"              | AV V2 train + AV V2 validation | 100000 | 0.001   | 0      |
| "nudge left"              | AV V2 train + AV V2 validation | 200000 | 0.001   | 0      |
| "nudge left"              | AV V2 train + AV V2 validation | 500000 | 0.001   | 0      |
| nudge                     | <all>                          | 5000   | 0.001   | 36     |
| nudge                     | <all>                          | 10000  | 0.000   | 36     |
| nudge                     | <all>                          | 30000  | 0.000   | 36     |
| nudge                     | <all>                          | 50000  | 0.000   | 36     |
| nudge                     | <all>                          | 100000 | 0.000   | 36     |
| nudge                     | <all>                          | 200000 | 0.000   | 36     |
| nudge                     | <all>                          | 500000 | 0.000   | 36     |
| nudge                     | Nexar                          | 5000   | 0.023   | 2      |
| nudge                     | Nexar                          | 10000  | 0.021   | 2      |
| nudge                     | Nexar                          | 30000  | 0.021   | 2      |
| nudge                     | Nexar                          | 50000  | 0.021   | 2      |
| nudge                     | Nexar                          | 100000 | 0.021   | 2      |
| nudge                     | Nexar                          | 200000 | 0.021   | 2      |
| nudge                     | Nexar                          | 500000 | 0.021   | 2      |
| nudge                     | Physical AI                    | 5000   | 0.000   | 0      |
| nudge                     | Physical AI                    | 10000  | 0.000   | 0      |
| nudge                     | Physical AI                    | 30000  | 0.000   | 0      |
| nudge                     | Physical AI                    | 50000  | 0.000   | 0      |
| nudge                     | Physical AI                    | 100000 | 0.000   | 0      |
| nudge                     | Physical AI                    | 200000 | 0.000   | 0      |
| nudge                     | Physical AI                    | 500000 | 0.000   | 0      |
| nudge                     | AV V2 train + AV V2 validation | 5000   | 0.144   | 4      |
| nudge                     | AV V2 train + AV V2 validation | 10000  | 0.136   | 4      |
| nudge                     | AV V2 train + AV V2 validation | 30000  | 0.136   | 4      |
| nudge                     | AV V2 train + AV V2 validation | 50000  | 0.137   | 4      |
| nudge                     | AV V2 train + AV V2 validation | 100000 | 0.137   | 4      |
| nudge                     | AV V2 train + AV V2 validation | 200000 | 0.135   | 4      |
| nudge                     | AV V2 train + AV V2 validation | 500000 | 0.135   | 4      |
| sway                      | <all>                          | 5000   | 0.481   | 5000   |
| sway                      | <all>                          | 10000  | 0.829   | 10000  |
| sway                      | <all>                          | 30000  | 1.336   | 30000  |
| sway                      | <all>                          | 50000  | 1.766   | 50000  |
| sway                      | <all>                          | 100000 | 3.229   | 100000 |
| sway                      | <all>                          | 200000 | 4.977   | 200000 |
| sway                      | <all>                          | 500000 | 15.579  | 500000 |
| sway                      | Nexar                          | 5000   | 0.209   | 5000   |
| sway                      | Nexar                          | 10000  | 0.227   | 10000  |
| sway                      | Nexar                          | 30000  | 0.354   | 30000  |
| sway                      | Nexar                          | 50000  | 0.586   | 50000  |
| sway                      | Nexar                          | 100000 | 0.747   | 75846  |
| sway                      | Nexar                          | 200000 | 0.600   | 75846  |
| sway                      | Nexar                          | 500000 | 0.597   | 75846  |
| sway                      | Physical AI                    | 5000   | 0.061   | 5000   |
| sway                      | Physical AI                    | 10000  | 0.060   | 6284   |
| sway                      | Physical AI                    | 30000  | 0.055   | 6284   |
| sway                      | Physical AI                    | 50000  | 0.056   | 6284   |
| sway                      | Physical AI                    | 100000 | 0.053   | 6284   |
| sway                      | Physical AI                    | 200000 | 0.052   | 6284   |
| sway                      | Physical AI                    | 500000 | 0.052   | 6284   |
| sway                      | AV V2 train + AV V2 validation | 5000   | 0.522   | 5000   |
| sway                      | AV V2 train + AV V2 validation | 10000  | 0.554   | 10000  |
| sway                      | AV V2 train + AV V2 validation | 30000  | 0.632   | 30000  |
| sway                      | AV V2 train + AV V2 validation | 50000  | 0.722   | 50000  |
| sway                      | AV V2 train + AV V2 validation | 100000 | 0.975   | 100000 |
| sway                      | AV V2 train + AV V2 validation | 200000 | 1.386   | 200000 |
| sway                      | AV V2 train + AV V2 validation | 500000 | 2.045   | 299671 |
| fishtail                  | <all>                          | 5000   | 0.000   | 9      |
| fishtail                  | <all>                          | 10000  | 0.000   | 9      |
| fishtail                  | <all>                          | 30000  | 0.000   | 9      |
| fishtail                  | <all>                          | 50000  | 0.000   | 9      |
| fishtail                  | <all>                          | 100000 | 0.000   | 9      |
| fishtail                  | <all>                          | 200000 | 0.000   | 9      |
| fishtail                  | <all>                          | 500000 | 0.000   | 9      |
| fishtail                  | Nexar                          | 5000   | 0.025   | 1      |
| fishtail                  | Nexar                          | 10000  | 0.021   | 1      |
| fishtail                  | Nexar                          | 30000  | 0.021   | 1      |
| fishtail                  | Nexar                          | 50000  | 0.021   | 1      |
| fishtail                  | Nexar                          | 100000 | 0.021   | 1      |
| fishtail                  | Nexar                          | 200000 | 0.021   | 1      |
| fishtail                  | Nexar                          | 500000 | 0.021   | 1      |
| fishtail                  | Physical AI                    | 5000   | 0.000   | 0      |
| fishtail                  | Physical AI                    | 10000  | 0.000   | 0      |
| fishtail                  | Physical AI                    | 30000  | 0.000   | 0      |
| fishtail                  | Physical AI                    | 50000  | 0.000   | 0      |
| fishtail                  | Physical AI                    | 100000 | 0.000   | 0      |
| fishtail                  | Physical AI                    | 200000 | 0.000   | 0      |
| fishtail                  | Physical AI                    | 500000 | 0.000   | 0      |
| fishtail                  | AV V2 train + AV V2 validation | 5000   | 0.145   | 7      |
| fishtail                  | AV V2 train + AV V2 validation | 10000  | 0.136   | 7      |
| fishtail                  | AV V2 train + AV V2 validation | 30000  | 0.137   | 7      |
| fishtail                  | AV V2 train + AV V2 validation | 50000  | 0.135   | 7      |
| fishtail                  | AV V2 train + AV V2 validation | 100000 | 0.135   | 7      |
| fishtail                  | AV V2 train + AV V2 validation | 200000 | 0.135   | 7      |
| fishtail                  | AV V2 train + AV V2 validation | 500000 | 0.136   | 7      |
| hydroplane                | <all>                          | 5000   | 0.002   | 145    |
| hydroplane                | <all>                          | 10000  | 0.001   | 145    |
| hydroplane                | <all>                          | 30000  | 0.001   | 145    |
| hydroplane                | <all>                          | 50000  | 0.001   | 145    |
| hydroplane                | <all>                          | 100000 | 0.001   | 145    |
| hydroplane                | <all>                          | 200000 | 0.001   | 145    |
| hydroplane                | <all>                          | 500000 | 0.001   | 145    |
| hydroplane                | Nexar                          | 5000   | 0.022   | 15     |
| hydroplane                | Nexar                          | 10000  | 0.021   | 15     |
| hydroplane                | Nexar                          | 30000  | 0.021   | 15     |
| hydroplane                | Nexar                          | 50000  | 0.021   | 15     |
| hydroplane                | Nexar                          | 100000 | 0.021   | 15     |
| hydroplane                | Nexar                          | 200000 | 0.021   | 15     |
| hydroplane                | Nexar                          | 500000 | 0.021   | 15     |
| hydroplane                | Physical AI                    | 5000   | 0.007   | 9      |
| hydroplane                | Physical AI                    | 10000  | 0.006   | 9      |
| hydroplane                | Physical AI                    | 30000  | 0.006   | 9      |
| hydroplane                | Physical AI                    | 50000  | 0.006   | 9      |
| hydroplane                | Physical AI                    | 100000 | 0.006   | 9      |
| hydroplane                | Physical AI                    | 200000 | 0.006   | 9      |
| hydroplane                | Physical AI                    | 500000 | 0.006   | 9      |
| hydroplane                | AV V2 train + AV V2 validation | 5000   | 0.142   | 111    |
| hydroplane                | AV V2 train + AV V2 validation | 10000  | 0.136   | 111    |
| hydroplane                | AV V2 train + AV V2 validation | 30000  | 0.136   | 111    |
| hydroplane                | AV V2 train + AV V2 validation | 50000  | 0.136   | 111    |
| hydroplane                | AV V2 train + AV V2 validation | 100000 | 0.136   | 111    |
| hydroplane                | AV V2 train + AV V2 validation | 200000 | 0.136   | 111    |
| hydroplane                | AV V2 train + AV V2 validation | 500000 | 0.136   | 111    |
| jackknife                 | <all>                          | 5000   | 0.000   | 4      |
| jackknife                 | <all>                          | 10000  | 0.000   | 4      |
| jackknife                 | <all>                          | 30000  | 0.000   | 4      |
| jackknife                 | <all>                          | 50000  | 0.000   | 4      |
| jackknife                 | <all>                          | 100000 | 0.000   | 4      |
| jackknife                 | <all>                          | 200000 | 0.000   | 4      |
| jackknife                 | <all>                          | 500000 | 0.000   | 4      |
| jackknife                 | Nexar                          | 5000   | 0.022   | 2      |
| jackknife                 | Nexar                          | 10000  | 0.021   | 2      |
| jackknife                 | Nexar                          | 30000  | 0.021   | 2      |
| jackknife                 | Nexar                          | 50000  | 0.022   | 2      |
| jackknife                 | Nexar                          | 100000 | 0.022   | 2      |
| jackknife                 | Nexar                          | 200000 | 0.021   | 2      |
| jackknife                 | Nexar                          | 500000 | 0.021   | 2      |
| jackknife                 | Physical AI                    | 5000   | 0.000   | 0      |
| jackknife                 | Physical AI                    | 10000  | 0.000   | 0      |
| jackknife                 | Physical AI                    | 30000  | 0.000   | 0      |
| jackknife                 | Physical AI                    | 50000  | 0.000   | 0      |
| jackknife                 | Physical AI                    | 100000 | 0.000   | 0      |
| jackknife                 | Physical AI                    | 200000 | 0.000   | 0      |
| jackknife                 | Physical AI                    | 500000 | 0.000   | 0      |
| jackknife                 | AV V2 train + AV V2 validation | 5000   | 0.137   | 2      |
| jackknife                 | AV V2 train + AV V2 validation | 10000  | 0.135   | 2      |
| jackknife                 | AV V2 train + AV V2 validation | 30000  | 0.135   | 2      |
| jackknife                 | AV V2 train + AV V2 validation | 50000  | 0.135   | 2      |
| jackknife                 | AV V2 train + AV V2 validation | 100000 | 0.134   | 2      |
| jackknife                 | AV V2 train + AV V2 validation | 200000 | 0.134   | 2      |
| jackknife                 | AV V2 train + AV V2 validation | 500000 | 0.136   | 2      |
| chicane                   | <all>                          | 5000   | 0.000   | 7      |
| chicane                   | <all>                          | 10000  | 0.000   | 7      |
| chicane                   | <all>                          | 30000  | 0.000   | 7      |
| chicane                   | <all>                          | 50000  | 0.000   | 7      |
| chicane                   | <all>                          | 100000 | 0.000   | 7      |
| chicane                   | <all>                          | 200000 | 0.000   | 7      |
| chicane                   | <all>                          | 500000 | 0.000   | 7      |
| chicane                   | Nexar                          | 5000   | 0.000   | 0      |
| chicane                   | Nexar                          | 10000  | 0.000   | 0      |
| chicane                   | Nexar                          | 30000  | 0.000   | 0      |
| chicane                   | Nexar                          | 50000  | 0.000   | 0      |
| chicane                   | Nexar                          | 100000 | 0.000   | 0      |
| chicane                   | Nexar                          | 200000 | 0.000   | 0      |
| chicane                   | Nexar                          | 500000 | 0.000   | 0      |
| chicane                   | Physical AI                    | 5000   | 0.006   | 5      |
| chicane                   | Physical AI                    | 10000  | 0.006   | 5      |
| chicane                   | Physical AI                    | 30000  | 0.006   | 5      |
| chicane                   | Physical AI                    | 50000  | 0.006   | 5      |
| chicane                   | Physical AI                    | 100000 | 0.006   | 5      |
| chicane                   | Physical AI                    | 200000 | 0.006   | 5      |
| chicane                   | Physical AI                    | 500000 | 0.006   | 5      |
| chicane                   | AV V2 train + AV V2 validation | 5000   | 0.135   | 1      |
| chicane                   | AV V2 train + AV V2 validation | 10000  | 0.134   | 1      |
| chicane                   | AV V2 train + AV V2 validation | 30000  | 0.134   | 1      |
| chicane                   | AV V2 train + AV V2 validation | 50000  | 0.135   | 1      |
| chicane                   | AV V2 train + AV V2 validation | 100000 | 0.134   | 1      |
| chicane                   | AV V2 train + AV V2 validation | 200000 | 0.134   | 1      |
| chicane                   | AV V2 train + AV V2 validation | 500000 | 0.135   | 1      |
| pothole                   | <all>                          | 5000   | 0.020   | 2646   |
| pothole                   | <all>                          | 10000  | 0.011   | 2646   |
| pothole                   | <all>                          | 30000  | 0.009   | 2646   |
| pothole                   | <all>                          | 50000  | 0.009   | 2646   |
| pothole                   | <all>                          | 100000 | 0.009   | 2646   |
| pothole                   | <all>                          | 200000 | 0.008   | 2646   |
| pothole                   | <all>                          | 500000 | 0.008   | 2646   |
| pothole                   | Nexar                          | 5000   | 0.024   | 174    |
| pothole                   | Nexar                          | 10000  | 0.024   | 174    |
| pothole                   | Nexar                          | 30000  | 0.023   | 174    |
| pothole                   | Nexar                          | 50000  | 0.023   | 174    |
| pothole                   | Nexar                          | 100000 | 0.025   | 174    |
| pothole                   | Nexar                          | 200000 | 0.026   | 174    |
| pothole                   | Nexar                          | 500000 | 0.026   | 174    |
| pothole                   | Physical AI                    | 5000   | 0.008   | 43     |
| pothole                   | Physical AI                    | 10000  | 0.007   | 43     |
| pothole                   | Physical AI                    | 30000  | 0.007   | 43     |
| pothole                   | Physical AI                    | 50000  | 0.007   | 43     |
| pothole                   | Physical AI                    | 100000 | 0.007   | 43     |
| pothole                   | Physical AI                    | 200000 | 0.007   | 43     |
| pothole                   | Physical AI                    | 500000 | 0.007   | 43     |
| pothole                   | AV V2 train + AV V2 validation | 5000   | 0.153   | 1766   |
| pothole                   | AV V2 train + AV V2 validation | 10000  | 0.153   | 1766   |
| pothole                   | AV V2 train + AV V2 validation | 30000  | 0.152   | 1766   |
| pothole                   | AV V2 train + AV V2 validation | 50000  | 0.150   | 1766   |
| pothole                   | AV V2 train + AV V2 validation | 100000 | 0.151   | 1766   |
| pothole                   | AV V2 train + AV V2 validation | 200000 | 0.156   | 1766   |
| pothole                   | AV V2 train + AV V2 validation | 500000 | 0.152   | 1766   |
| glare                     | <all>                          | 5000   | 0.596   | 5000   |
| glare                     | <all>                          | 10000  | 0.792   | 10000  |
| glare                     | <all>                          | 30000  | 1.211   | 30000  |
| glare                     | <all>                          | 50000  | 1.343   | 50000  |
| glare                     | <all>                          | 100000 | 2.247   | 100000 |
| glare                     | <all>                          | 200000 | 3.841   | 200000 |
| glare                     | <all>                          | 500000 | 10.409  | 500000 |
| glare                     | Nexar                          | 5000   | 0.243   | 5000   |
| glare                     | Nexar                          | 10000  | 0.254   | 10000  |
| glare                     | Nexar                          | 30000  | 0.345   | 30000  |
| glare                     | Nexar                          | 50000  | 0.498   | 50000  |
| glare                     | Nexar                          | 100000 | 0.860   | 100000 |
| glare                     | Nexar                          | 200000 | 0.782   | 104834 |
| glare                     | Nexar                          | 500000 | 0.781   | 104834 |
| glare                     | Physical AI                    | 5000   | 0.098   | 5000   |
| glare                     | Physical AI                    | 10000  | 0.088   | 10000  |
| glare                     | Physical AI                    | 30000  | 0.231   | 25085  |
| glare                     | Physical AI                    | 50000  | 0.205   | 25085  |
| glare                     | Physical AI                    | 100000 | 0.204   | 25085  |
| glare                     | Physical AI                    | 200000 | 0.206   | 25085  |
| glare                     | Physical AI                    | 500000 | 0.204   | 25085  |
| glare                     | AV V2 train + AV V2 validation | 5000   | 0.667   | 5000   |
| glare                     | AV V2 train + AV V2 validation | 10000  | 0.688   | 10000  |
| glare                     | AV V2 train + AV V2 validation | 30000  | 0.790   | 30000  |
| glare                     | AV V2 train + AV V2 validation | 50000  | 0.885   | 50000  |
| glare                     | AV V2 train + AV V2 validation | 100000 | 1.162   | 100000 |
| glare                     | AV V2 train + AV V2 validation | 200000 | 1.579   | 200000 |
| glare                     | AV V2 train + AV V2 validation | 500000 | 4.862   | 476307 |
| tailgating                | <all>                          | 5000   | 0.061   | 5000   |
| tailgating                | <all>                          | 10000  | 0.139   | 10000  |
| tailgating                | <all>                          | 30000  | 0.113   | 11910  |
| tailgating                | <all>                          | 50000  | 0.079   | 11910  |
| tailgating                | <all>                          | 100000 | 0.076   | 11910  |
| tailgating                | <all>                          | 200000 | 0.075   | 11910  |
| tailgating                | <all>                          | 500000 | 0.074   | 11910  |
| tailgating                | Nexar                          | 5000   | 0.029   | 221    |
| tailgating                | Nexar                          | 10000  | 0.027   | 221    |
| tailgating                | Nexar                          | 30000  | 0.026   | 221    |
| tailgating                | Nexar                          | 50000  | 0.025   | 221    |
| tailgating                | Nexar                          | 100000 | 0.025   | 221    |
| tailgating                | Nexar                          | 200000 | 0.027   | 221    |
| tailgating                | Nexar                          | 500000 | 0.026   | 221    |
| tailgating                | Physical AI                    | 5000   | 0.015   | 877    |
| tailgating                | Physical AI                    | 10000  | 0.013   | 877    |
| tailgating                | Physical AI                    | 30000  | 0.012   | 877    |
| tailgating                | Physical AI                    | 50000  | 0.012   | 877    |
| tailgating                | Physical AI                    | 100000 | 0.012   | 877    |
| tailgating                | Physical AI                    | 200000 | 0.012   | 877    |
| tailgating                | Physical AI                    | 500000 | 0.012   | 877    |
| tailgating                | AV V2 train + AV V2 validation | 5000   | 0.223   | 5000   |
| tailgating                | AV V2 train + AV V2 validation | 10000  | 0.232   | 10000  |
| tailgating                | AV V2 train + AV V2 validation | 30000  | 0.250   | 10334  |
| tailgating                | AV V2 train + AV V2 validation | 50000  | 0.254   | 10334  |
| tailgating                | AV V2 train + AV V2 validation | 100000 | 0.248   | 10334  |
| tailgating                | AV V2 train + AV V2 validation | 200000 | 0.247   | 10334  |
| tailgating                | AV V2 train + AV V2 validation | 500000 | 0.257   | 10334  |
| contraflow                | <all>                          | 5000   | 0.000   | 15     |
| contraflow                | <all>                          | 10000  | 0.000   | 15     |
| contraflow                | <all>                          | 30000  | 0.000   | 15     |
| contraflow                | <all>                          | 50000  | 0.000   | 15     |
| contraflow                | <all>                          | 100000 | 0.000   | 15     |
| contraflow                | <all>                          | 200000 | 0.000   | 15     |
| contraflow                | <all>                          | 500000 | 0.000   | 15     |
| contraflow                | Nexar                          | 5000   | 0.023   | 2      |
| contraflow                | Nexar                          | 10000  | 0.021   | 2      |
| contraflow                | Nexar                          | 30000  | 0.021   | 2      |
| contraflow                | Nexar                          | 50000  | 0.021   | 2      |
| contraflow                | Nexar                          | 100000 | 0.021   | 2      |
| contraflow                | Nexar                          | 200000 | 0.021   | 2      |
| contraflow                | Nexar                          | 500000 | 0.021   | 2      |
| contraflow                | Physical AI                    | 5000   | 0.000   | 0      |
| contraflow                | Physical AI                    | 10000  | 0.000   | 0      |
| contraflow                | Physical AI                    | 30000  | 0.000   | 0      |
| contraflow                | Physical AI                    | 50000  | 0.000   | 0      |
| contraflow                | Physical AI                    | 100000 | 0.000   | 0      |
| contraflow                | Physical AI                    | 200000 | 0.000   | 0      |
| contraflow                | Physical AI                    | 500000 | 0.000   | 0      |
| contraflow                | AV V2 train + AV V2 validation | 5000   | 0.143   | 8      |
| contraflow                | AV V2 train + AV V2 validation | 10000  | 0.135   | 8      |
| contraflow                | AV V2 train + AV V2 validation | 30000  | 0.134   | 8      |
| contraflow                | AV V2 train + AV V2 validation | 50000  | 0.134   | 8      |
| contraflow                | AV V2 train + AV V2 validation | 100000 | 0.136   | 8      |
| contraflow                | AV V2 train + AV V2 validation | 200000 | 0.133   | 8      |
| contraflow                | AV V2 train + AV V2 validation | 500000 | 0.134   | 8      |