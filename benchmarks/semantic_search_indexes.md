# Semantic Search Benchmark — Summary

_commit `68be5b2` · date 2026-03-22 20:29:03 · embeddings_dir: /path/to/wheel-data/benchmark · embeddings_size: 31.9 GB · k: 2048 · n_queries: 14 · n_indexes: 11 · annotations_db: /path/to/wheel-data/annotations_latest_schema.db_

| index                     | nprobe | slice | n_slice | mean_s | mean_count | mean_recall | valid_q |
| ------------------------- | ------ | ----- | ------- | ------ | ---------- | ----------- | ------- |
| flat                      | 256    | all   | 3000000 | 0.2163 | 4096       | n/a         | 0/14    |
| flat                      | 256    | mads  | 21459   | 0.2141 | 53         | n/a         | 0/14    |
| flat                      | 256    | night | 534     | 0.2144 | 12         | n/a         | 0/14    |
| flat                      | 256    | sign  | 325     | 0.2143 | 1          | n/a         | 0/14    |
| ivf4096_pq96x8            | 256    | all   | 3000000 | 0.0116 | 2048       | 0.397       | 14/14   |
| ivf4096_pq96x8            | 256    | mads  | 21459   | 0.0111 | 5          | 0.343       | 7/14    |
| ivf4096_pq96x8            | 256    | night | 534     | 0.0109 | 5          | 0.364       | 6/14    |
| ivf4096_pq96x8            | 256    | sign  | 325     | 0.0115 | 0          | 1.000       | 1/14    |
| opq96_768_ivf4096_pq96x8  | 256    | all   | 3000000 | 0.0045 | 819        | 0.063       | 14/14   |
| opq96_768_ivf4096_pq96x8  | 256    | mads  | 21459   | 0.0044 | 1          | 0.074       | 3/14    |
| opq96_768_ivf4096_pq96x8  | 256    | night | 534     | 0.0044 | 6          | 0.092       | 4/14    |
| opq96_768_ivf4096_pq96x8  | 256    | sign  | 325     | 0.0042 | 0          | 0.000       | 1/14    |
| ivf8192_pq96x8            | 512    | all   | 3000000 | 0.0132 | 2048       | 0.408       | 14/14   |
| ivf8192_pq96x8            | 512    | mads  | 21459   | 0.0118 | 7          | 0.279       | 7/14    |
| ivf8192_pq96x8            | 512    | night | 534     | 0.0114 | 6          | 0.415       | 6/14    |
| ivf8192_pq96x8            | 512    | sign  | 325     | 0.0114 | 0          | n/a         | 0/14    |
| ivf8192_pq96x8_np1024     | 1024   | all   | 3000000 | 0.0184 | 2048       | 0.409       | 14/14   |
| ivf8192_pq96x8_np1024     | 1024   | mads  | 21459   | 0.0167 | 8          | 0.290       | 7/14    |
| ivf8192_pq96x8_np1024     | 1024   | night | 534     | 0.0165 | 6          | 0.403       | 6/14    |
| ivf8192_pq96x8_np1024     | 1024   | sign  | 325     | 0.0168 | 0          | n/a         | 0/14    |
| ivf8192_pq96x8_np2048     | 2048   | all   | 3000000 | 0.0299 | 2048       | 0.409       | 14/14   |
| ivf8192_pq96x8_np2048     | 2048   | mads  | 21459   | 0.0286 | 8          | 0.291       | 7/14    |
| ivf8192_pq96x8_np2048     | 2048   | night | 534     | 0.0282 | 6          | 0.403       | 6/14    |
| ivf8192_pq96x8_np2048     | 2048   | sign  | 325     | 0.0288 | 0          | n/a         | 0/14    |
| opq96_768_ivf8192_pq96x8  | 512    | all   | 3000000 | 0.0047 | 308        | 0.039       | 14/14   |
| opq96_768_ivf8192_pq96x8  | 512    | mads  | 21459   | 0.0045 | 0          | 0.043       | 3/14    |
| opq96_768_ivf8192_pq96x8  | 512    | night | 534     | 0.0044 | 0          | 0.133       | 1/14    |
| opq96_768_ivf8192_pq96x8  | 512    | sign  | 325     | 0.0045 | 0          | n/a         | 0/14    |
| ivf16384_pq96x8           | 1024   | all   | 3000000 | 0.0136 | 2048       | 0.419       | 14/14   |
| ivf16384_pq96x8           | 1024   | mads  | 21459   | 0.0129 | 14         | 0.394       | 9/14    |
| ivf16384_pq96x8           | 1024   | night | 534     | 0.0121 | 4          | 0.384       | 5/14    |
| ivf16384_pq96x8           | 1024   | sign  | 325     | 0.0124 | 0          | 0.750       | 2/14    |
| opq96_768_ivf16384_pq96x8 | 1024   | all   | 3000000 | 0.0053 | 150        | 0.023       | 14/14   |
| opq96_768_ivf16384_pq96x8 | 1024   | mads  | 21459   | 0.0050 | 9          | 0.130       | 2/14    |
| opq96_768_ivf16384_pq96x8 | 1024   | night | 534     | 0.0051 | 0          | n/a         | 0/14    |
| opq96_768_ivf16384_pq96x8 | 1024   | sign  | 325     | 0.0050 | 0          | n/a         | 0/14    |
| hnsw32                    | 128    | all   | 3000000 | 0.0067 | 1996       | 0.317       | 14/14   |
| hnsw32                    | 128    | mads  | 21459   | 0.0064 | 15         | 0.273       | 9/14    |
| hnsw32                    | 128    | night | 534     | 0.0061 | 3          | 0.220       | 4/14    |
| hnsw32                    | 128    | sign  | 325     | 0.0060 | 0          | 1.000       | 1/14    |
| hnsw64                    | 256    | all   | 3000000 | 0.0080 | 2048       | 0.428       | 14/14   |
| hnsw64                    | 256    | mads  | 21459   | 0.0068 | 15         | 0.314       | 8/14    |
| hnsw64                    | 256    | night | 534     | 0.0069 | 4          | 0.360       | 4/14    |
| hnsw64                    | 256    | sign  | 325     | 0.0074 | 1          | 1.000       | 3/14    |

# Semantic Search Benchmark — Detail

_commit `68be5b2` · date 2026-03-22 20:29:03 · embeddings_dir: /path/to/wheel-data/benchmark · embeddings_size: 31.9 GB · k: 2048 · n_queries: 14 · n_indexes: 11 · annotations_db: /path/to/wheel-data/annotations_latest_schema.db_

| index                     | nprobe | slice | n_slice | query                                     | seconds | count | max_score | min_score | recall   |
| ------------------------- | ------ | ----- | ------- | ----------------------------------------- | ------- | ----- | --------- | --------- | -------- |
| flat                      | 256    | all   | 3000000 | pedestrian crossing the street            | 0.2151  | 4096  | 0.4149    | 0.2779    | baseline |
| flat                      | 256    | all   | 3000000 | wheelchair user crossing road             | 0.2164  | 4096  | 0.4993    | 0.2316    | baseline |
| flat                      | 256    | all   | 3000000 | bicyclist passing by parked cars          | 0.2231  | 4096  | 0.5239    | 0.3126    | baseline |
| flat                      | 256    | all   | 3000000 | child near a crosswalk                    | 0.2167  | 4096  | 0.3473    | 0.2163    | baseline |
| flat                      | 256    | all   | 3000000 | dog on the sidewalk                       | 0.2221  | 4096  | 0.4483    | 0.1868    | baseline |
| flat                      | 256    | all   | 3000000 | car turning left at an intersection       | 0.2184  | 4096  | 0.2663    | 0.1810    | baseline |
| flat                      | 256    | all   | 3000000 | traffic cones on the street               | 0.2151  | 4096  | 0.4472    | 0.2544    | baseline |
| flat                      | 256    | all   | 3000000 | construction zone with cones and barriers | 0.2146  | 4096  | 0.4569    | 0.2498    | baseline |
| flat                      | 256    | all   | 3000000 | nighttime street with headlights          | 0.2164  | 4096  | 0.4941    | 0.2780    | baseline |
| flat                      | 256    | all   | 3000000 | rainy day pedestrians with umbrellas      | 0.2139  | 4096  | 0.4544    | 0.2559    | baseline |
| flat                      | 256    | all   | 3000000 | night                                     | 0.2139  | 4096  | 0.3707    | 0.2876    | baseline |
| flat                      | 256    | all   | 3000000 | officer holding a sign                    | 0.2141  | 4096  | 0.3467    | 0.1567    | baseline |
| flat                      | 256    | all   | 3000000 | officer signaling to stop                 | 0.2139  | 4096  | 0.4154    | 0.2002    | baseline |
| flat                      | 256    | all   | 3000000 | officer waving to go                      | 0.2139  | 4096  | 0.3642    | 0.1440    | baseline |
| flat                      | 256    | mads  | 21459   | pedestrian crossing the street            | 0.2147  | 2     | 0.3233    | 0.3108    | baseline |
| flat                      | 256    | mads  | 21459   | dog on the sidewalk                       | 0.2144  | 2     | 0.1942    | 0.1872    | baseline |
| flat                      | 256    | mads  | 21459   | car turning left at an intersection       | 0.2146  | 416   | 0.2500    | 0.1810    | baseline |
| flat                      | 256    | mads  | 21459   | traffic cones on the street               | 0.2148  | 9     | 0.3093    | 0.2554    | baseline |
| flat                      | 256    | mads  | 21459   | construction zone with cones and barriers | 0.2132  | 10    | 0.3260    | 0.2514    | baseline |
| flat                      | 256    | mads  | 21459   | nighttime street with headlights          | 0.2125  | 12    | 0.3692    | 0.2794    | baseline |
| flat                      | 256    | mads  | 21459   | rainy day pedestrians with umbrellas      | 0.2144  | 2     | 0.3299    | 0.2578    | baseline |
| flat                      | 256    | mads  | 21459   | night                                     | 0.2138  | 2     | 0.2959    | 0.2876    | baseline |
| flat                      | 256    | mads  | 21459   | officer holding a sign                    | 0.2146  | 52    | 0.2670    | 0.1571    | baseline |
| flat                      | 256    | mads  | 21459   | officer signaling to stop                 | 0.2139  | 6     | 0.2459    | 0.2071    | baseline |
| flat                      | 256    | mads  | 21459   | officer waving to go                      | 0.2142  | 65    | 0.2314    | 0.1441    | baseline |
| flat                      | 256    | night | 534     | wheelchair user crossing road             | 0.2144  | 2     | 0.2419    | 0.2386    | baseline |
| flat                      | 256    | night | 534     | child near a crosswalk                    | 0.2144  | 4     | 0.2400    | 0.2175    | baseline |
| flat                      | 256    | night | 534     | nighttime street with headlights          | 0.2149  | 15    | 0.3473    | 0.2856    | baseline |
| flat                      | 256    | night | 534     | night                                     | 0.2145  | 26    | 0.3319    | 0.2881    | baseline |
| flat                      | 256    | night | 534     | officer holding a sign                    | 0.2135  | 13    | 0.1773    | 0.1572    | baseline |
| flat                      | 256    | night | 534     | officer signaling to stop                 | 0.2146  | 14    | 0.2800    | 0.2094    | baseline |
| flat                      | 256    | night | 534     | officer waving to go                      | 0.2146  | 9     | 0.1926    | 0.1453    | baseline |
| flat                      | 256    | sign  | 325     | pedestrian crossing the street            | 0.2144  | 1     | 0.3350    | 0.3350    | baseline |
| flat                      | 256    | sign  | 325     | child near a crosswalk                    | 0.2140  | 1     | 0.2216    | 0.2216    | baseline |
| flat                      | 256    | sign  | 325     | traffic cones on the street               | 0.2145  | 1     | 0.2827    | 0.2827    | baseline |
| flat                      | 256    | sign  | 325     | construction zone with cones and barriers | 0.2143  | 2     | 0.2697    | 0.2595    | baseline |
| ivf4096_pq96x8            | 256    | all   | 3000000 | pedestrian crossing the street            | 0.0126  | 2048  | 0.4148    | 0.2919    | 0.396    |
| ivf4096_pq96x8            | 256    | all   | 3000000 | wheelchair user crossing road             | 0.0119  | 2048  | 0.3637    | 0.2419    | 0.392    |
| ivf4096_pq96x8            | 256    | all   | 3000000 | bicyclist passing by parked cars          | 0.0123  | 2048  | 0.5141    | 0.3243    | 0.415    |
| ivf4096_pq96x8            | 256    | all   | 3000000 | child near a crosswalk                    | 0.0123  | 2048  | 0.3475    | 0.2258    | 0.404    |
| ivf4096_pq96x8            | 256    | all   | 3000000 | dog on the sidewalk                       | 0.0114  | 2048  | 0.4260    | 0.2135    | 0.366    |
| ivf4096_pq96x8            | 256    | all   | 3000000 | car turning left at an intersection       | 0.0123  | 2048  | 0.2530    | 0.1848    | 0.277    |
| ivf4096_pq96x8            | 256    | all   | 3000000 | traffic cones on the street               | 0.0117  | 2048  | 0.4028    | 0.2861    | 0.433    |
| ivf4096_pq96x8            | 256    | all   | 3000000 | construction zone with cones and barriers | 0.0112  | 2048  | 0.4150    | 0.2751    | 0.435    |
| ivf4096_pq96x8            | 256    | all   | 3000000 | nighttime street with headlights          | 0.0104  | 2048  | 0.4529    | 0.3072    | 0.470    |
| ivf4096_pq96x8            | 256    | all   | 3000000 | rainy day pedestrians with umbrellas      | 0.0107  | 2048  | 0.4544    | 0.2889    | 0.450    |
| ivf4096_pq96x8            | 256    | all   | 3000000 | night                                     | 0.0107  | 2048  | 0.3768    | 0.3100    | 0.365    |
| ivf4096_pq96x8            | 256    | all   | 3000000 | officer holding a sign                    | 0.0114  | 2048  | 0.2792    | 0.1489    | 0.361    |
| ivf4096_pq96x8            | 256    | all   | 3000000 | officer signaling to stop                 | 0.0118  | 2048  | 0.3661    | 0.2140    | 0.411    |
| ivf4096_pq96x8            | 256    | all   | 3000000 | officer waving to go                      | 0.0115  | 2048  | 0.3138    | 0.1548    | 0.379    |
| ivf4096_pq96x8            | 256    | mads  | 21459   | pedestrian crossing the street            | 0.0119  | 1     | 0.3567    | 0.3567    | 0.500    |
| ivf4096_pq96x8            | 256    | mads  | 21459   | dog on the sidewalk                       | 0.0112  | 0     | nan       | nan       | n/a      |
| ivf4096_pq96x8            | 256    | mads  | 21459   | car turning left at an intersection       | 0.0122  | 38    | 0.2345    | 0.1852    | 0.089    |
| ivf4096_pq96x8            | 256    | mads  | 21459   | traffic cones on the street               | 0.0118  | 7     | 0.3213    | 0.2884    | 0.667    |
| ivf4096_pq96x8            | 256    | mads  | 21459   | construction zone with cones and barriers | 0.0110  | 4     | 0.3325    | 0.2788    | 0.400    |
| ivf4096_pq96x8            | 256    | mads  | 21459   | nighttime street with headlights          | 0.0098  | 2     | 0.3229    | 0.3087    | 0.167    |
| ivf4096_pq96x8            | 256    | mads  | 21459   | rainy day pedestrians with umbrellas      | 0.0099  | 1     | 0.3516    | 0.3516    | 0.500    |
| ivf4096_pq96x8            | 256    | mads  | 21459   | night                                     | 0.0102  | 0     | nan       | nan       | n/a      |
| ivf4096_pq96x8            | 256    | mads  | 21459   | officer holding a sign                    | 0.0113  | 4     | 0.1974    | 0.1571    | 0.077    |
| ivf4096_pq96x8            | 256    | mads  | 21459   | officer signaling to stop                 | 0.0123  | 0     | nan       | nan       | n/a      |
| ivf4096_pq96x8            | 256    | mads  | 21459   | officer waving to go                      | 0.0106  | 0     | nan       | nan       | n/a      |
| ivf4096_pq96x8            | 256    | night | 534     | wheelchair user crossing road             | 0.0113  | 0     | nan       | nan       | n/a      |
| ivf4096_pq96x8            | 256    | night | 534     | child near a crosswalk                    | 0.0112  | 1     | 0.2290    | 0.2290    | 0.250    |
| ivf4096_pq96x8            | 256    | night | 534     | nighttime street with headlights          | 0.0099  | 8     | 0.3275    | 0.3074    | 0.533    |
| ivf4096_pq96x8            | 256    | night | 534     | night                                     | 0.0112  | 12    | 0.3441    | 0.3107    | 0.269    |
| ivf4096_pq96x8            | 256    | night | 534     | officer holding a sign                    | 0.0106  | 5     | 0.1804    | 0.1533    | 0.308    |
| ivf4096_pq96x8            | 256    | night | 534     | officer signaling to stop                 | 0.0111  | 10    | 0.2590    | 0.2149    | 0.714    |
| ivf4096_pq96x8            | 256    | night | 534     | officer waving to go                      | 0.0112  | 2     | 0.1634    | 0.1623    | 0.111    |
| ivf4096_pq96x8            | 256    | sign  | 325     | pedestrian crossing the street            | 0.0122  | 1     | 0.3874    | 0.3874    | 1.000    |
| ivf4096_pq96x8            | 256    | sign  | 325     | child near a crosswalk                    | 0.0116  | 0     | nan       | nan       | n/a      |
| ivf4096_pq96x8            | 256    | sign  | 325     | traffic cones on the street               | 0.0111  | 0     | nan       | nan       | n/a      |
| ivf4096_pq96x8            | 256    | sign  | 325     | construction zone with cones and barriers | 0.0113  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf4096_pq96x8  | 256    | all   | 3000000 | pedestrian crossing the street            | 0.0050  | 728   | 0.3941    | 0.0743    | 0.057    |
| opq96_768_ivf4096_pq96x8  | 256    | all   | 3000000 | wheelchair user crossing road             | 0.0046  | 1029  | 0.3412    | 0.0116    | 0.044    |
| opq96_768_ivf4096_pq96x8  | 256    | all   | 3000000 | bicyclist passing by parked cars          | 0.0044  | 653   | 0.5224    | 0.0694    | 0.068    |
| opq96_768_ivf4096_pq96x8  | 256    | all   | 3000000 | child near a crosswalk                    | 0.0045  | 947   | 0.3326    | 0.0137    | 0.062    |
| opq96_768_ivf4096_pq96x8  | 256    | all   | 3000000 | dog on the sidewalk                       | 0.0046  | 1318  | 0.3995    | 0.0020    | 0.083    |
| opq96_768_ivf4096_pq96x8  | 256    | all   | 3000000 | car turning left at an intersection       | 0.0048  | 1068  | 0.2384    | 0.0295    | 0.042    |
| opq96_768_ivf4096_pq96x8  | 256    | all   | 3000000 | traffic cones on the street               | 0.0044  | 561   | 0.4499    | 0.0147    | 0.076    |
| opq96_768_ivf4096_pq96x8  | 256    | all   | 3000000 | construction zone with cones and barriers | 0.0041  | 561   | 0.4538    | 0.0021    | 0.043    |
| opq96_768_ivf4096_pq96x8  | 256    | all   | 3000000 | nighttime street with headlights          | 0.0043  | 613   | 0.4828    | 0.1368    | 0.090    |
| opq96_768_ivf4096_pq96x8  | 256    | all   | 3000000 | rainy day pedestrians with umbrellas      | 0.0043  | 489   | 0.4540    | 0.1634    | 0.102    |
| opq96_768_ivf4096_pq96x8  | 256    | all   | 3000000 | night                                     | 0.0042  | 528   | 0.3606    | 0.1577    | 0.051    |
| opq96_768_ivf4096_pq96x8  | 256    | all   | 3000000 | officer holding a sign                    | 0.0044  | 784   | 0.2838    | 0.0016    | 0.020    |
| opq96_768_ivf4096_pq96x8  | 256    | all   | 3000000 | officer signaling to stop                 | 0.0046  | 1208  | 0.3156    | 0.0195    | 0.081    |
| opq96_768_ivf4096_pq96x8  | 256    | all   | 3000000 | officer waving to go                      | 0.0045  | 980   | 0.2692    | 0.0014    | 0.058    |
| opq96_768_ivf4096_pq96x8  | 256    | mads  | 21459   | pedestrian crossing the street            | 0.0045  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf4096_pq96x8  | 256    | mads  | 21459   | dog on the sidewalk                       | 0.0046  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf4096_pq96x8  | 256    | mads  | 21459   | car turning left at an intersection       | 0.0047  | 3     | 0.0787    | 0.0295    | 0.000    |
| opq96_768_ivf4096_pq96x8  | 256    | mads  | 21459   | traffic cones on the street               | 0.0043  | 5     | 0.3132    | 0.1983    | 0.222    |
| opq96_768_ivf4096_pq96x8  | 256    | mads  | 21459   | construction zone with cones and barriers | 0.0041  | 5     | 0.2295    | 0.1426    | 0.000    |
| opq96_768_ivf4096_pq96x8  | 256    | mads  | 21459   | nighttime street with headlights          | 0.0042  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf4096_pq96x8  | 256    | mads  | 21459   | rainy day pedestrians with umbrellas      | 0.0042  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf4096_pq96x8  | 256    | mads  | 21459   | night                                     | 0.0042  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf4096_pq96x8  | 256    | mads  | 21459   | officer holding a sign                    | 0.0043  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf4096_pq96x8  | 256    | mads  | 21459   | officer signaling to stop                 | 0.0045  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf4096_pq96x8  | 256    | mads  | 21459   | officer waving to go                      | 0.0043  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf4096_pq96x8  | 256    | night | 534     | wheelchair user crossing road             | 0.0044  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf4096_pq96x8  | 256    | night | 534     | child near a crosswalk                    | 0.0044  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf4096_pq96x8  | 256    | night | 534     | nighttime street with headlights          | 0.0042  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf4096_pq96x8  | 256    | night | 534     | night                                     | 0.0042  | 2     | 0.3222    | 0.2888    | 0.077    |
| opq96_768_ivf4096_pq96x8  | 256    | night | 534     | officer holding a sign                    | 0.0044  | 8     | 0.1697    | 0.0020    | 0.077    |
| opq96_768_ivf4096_pq96x8  | 256    | night | 534     | officer signaling to stop                 | 0.0045  | 22    | 0.2436    | 0.0981    | 0.214    |
| opq96_768_ivf4096_pq96x8  | 256    | night | 534     | officer waving to go                      | 0.0044  | 10    | 0.1435    | 0.0580    | 0.000    |
| opq96_768_ivf4096_pq96x8  | 256    | sign  | 325     | pedestrian crossing the street            | 0.0044  | 2     | 0.2130    | 0.1309    | 0.000    |
| opq96_768_ivf4096_pq96x8  | 256    | sign  | 325     | child near a crosswalk                    | 0.0044  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf4096_pq96x8  | 256    | sign  | 325     | traffic cones on the street               | 0.0042  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf4096_pq96x8  | 256    | sign  | 325     | construction zone with cones and barriers | 0.0039  | 0     | nan       | nan       | n/a      |
| ivf8192_pq96x8            | 512    | all   | 3000000 | pedestrian crossing the street            | 0.0131  | 2048  | 0.4145    | 0.2883    | 0.410    |
| ivf8192_pq96x8            | 512    | all   | 3000000 | wheelchair user crossing road             | 0.0144  | 2048  | 0.3547    | 0.2419    | 0.396    |
| ivf8192_pq96x8            | 512    | all   | 3000000 | bicyclist passing by parked cars          | 0.0135  | 2048  | 0.4966    | 0.3279    | 0.424    |
| ivf8192_pq96x8            | 512    | all   | 3000000 | child near a crosswalk                    | 0.0128  | 2048  | 0.3258    | 0.2275    | 0.409    |
| ivf8192_pq96x8            | 512    | all   | 3000000 | dog on the sidewalk                       | 0.0137  | 2048  | 0.4130    | 0.2122    | 0.384    |
| ivf8192_pq96x8            | 512    | all   | 3000000 | car turning left at an intersection       | 0.0134  | 2048  | 0.2577    | 0.1863    | 0.294    |
| ivf8192_pq96x8            | 512    | all   | 3000000 | traffic cones on the street               | 0.0120  | 2048  | 0.4323    | 0.2897    | 0.445    |
| ivf8192_pq96x8            | 512    | all   | 3000000 | construction zone with cones and barriers | 0.0139  | 2048  | 0.4138    | 0.2772    | 0.442    |
| ivf8192_pq96x8            | 512    | all   | 3000000 | nighttime street with headlights          | 0.0131  | 2048  | 0.4625    | 0.3060    | 0.477    |
| ivf8192_pq96x8            | 512    | all   | 3000000 | rainy day pedestrians with umbrellas      | 0.0123  | 2048  | 0.4730    | 0.2889    | 0.461    |
| ivf8192_pq96x8            | 512    | all   | 3000000 | night                                     | 0.0134  | 2048  | 0.3684    | 0.3085    | 0.376    |
| ivf8192_pq96x8            | 512    | all   | 3000000 | officer holding a sign                    | 0.0135  | 2048  | 0.3399    | 0.1549    | 0.380    |
| ivf8192_pq96x8            | 512    | all   | 3000000 | officer signaling to stop                 | 0.0131  | 2048  | 0.4022    | 0.2140    | 0.418    |
| ivf8192_pq96x8            | 512    | all   | 3000000 | officer waving to go                      | 0.0132  | 2048  | 0.3234    | 0.1555    | 0.396    |
| ivf8192_pq96x8            | 512    | mads  | 21459   | pedestrian crossing the street            | 0.0132  | 0     | nan       | nan       | n/a      |
| ivf8192_pq96x8            | 512    | mads  | 21459   | dog on the sidewalk                       | 0.0115  | 0     | nan       | nan       | n/a      |
| ivf8192_pq96x8            | 512    | mads  | 21459   | car turning left at an intersection       | 0.0124  | 33    | 0.2435    | 0.1874    | 0.079    |
| ivf8192_pq96x8            | 512    | mads  | 21459   | traffic cones on the street               | 0.0132  | 9     | 0.3545    | 0.2945    | 0.667    |
| ivf8192_pq96x8            | 512    | mads  | 21459   | construction zone with cones and barriers | 0.0120  | 5     | 0.3224    | 0.2778    | 0.400    |
| ivf8192_pq96x8            | 512    | mads  | 21459   | nighttime street with headlights          | 0.0110  | 5     | 0.3715    | 0.3085    | 0.417    |
| ivf8192_pq96x8            | 512    | mads  | 21459   | rainy day pedestrians with umbrellas      | 0.0116  | 0     | nan       | nan       | n/a      |
| ivf8192_pq96x8            | 512    | mads  | 21459   | night                                     | 0.0114  | 0     | nan       | nan       | n/a      |
| ivf8192_pq96x8            | 512    | mads  | 21459   | officer holding a sign                    | 0.0116  | 18    | 0.2399    | 0.1581    | 0.346    |
| ivf8192_pq96x8            | 512    | mads  | 21459   | officer signaling to stop                 | 0.0110  | 1     | 0.2454    | 0.2454    | 0.000    |
| ivf8192_pq96x8            | 512    | mads  | 21459   | officer waving to go                      | 0.0111  | 3     | 0.1795    | 0.1671    | 0.046    |
| ivf8192_pq96x8            | 512    | night | 534     | wheelchair user crossing road             | 0.0115  | 0     | nan       | nan       | n/a      |
| ivf8192_pq96x8            | 512    | night | 534     | child near a crosswalk                    | 0.0114  | 2     | 0.2409    | 0.2402    | 0.500    |
| ivf8192_pq96x8            | 512    | night | 534     | nighttime street with headlights          | 0.0118  | 6     | 0.3395    | 0.3088    | 0.400    |
| ivf8192_pq96x8            | 512    | night | 534     | night                                     | 0.0106  | 13    | 0.3297    | 0.3098    | 0.385    |
| ivf8192_pq96x8            | 512    | night | 534     | officer holding a sign                    | 0.0116  | 4     | 0.1764    | 0.1550    | 0.231    |
| ivf8192_pq96x8            | 512    | night | 534     | officer signaling to stop                 | 0.0118  | 12    | 0.2620    | 0.2150    | 0.643    |
| ivf8192_pq96x8            | 512    | night | 534     | officer waving to go                      | 0.0114  | 3     | 0.1630    | 0.1588    | 0.333    |
| ivf8192_pq96x8            | 512    | sign  | 325     | pedestrian crossing the street            | 0.0114  | 0     | nan       | nan       | n/a      |
| ivf8192_pq96x8            | 512    | sign  | 325     | child near a crosswalk                    | 0.0112  | 0     | nan       | nan       | n/a      |
| ivf8192_pq96x8            | 512    | sign  | 325     | traffic cones on the street               | 0.0120  | 0     | nan       | nan       | n/a      |
| ivf8192_pq96x8            | 512    | sign  | 325     | construction zone with cones and barriers | 0.0111  | 0     | nan       | nan       | n/a      |
| ivf8192_pq96x8_np1024     | 1024   | all   | 3000000 | pedestrian crossing the street            | 0.0194  | 2048  | 0.4145    | 0.2883    | 0.411    |
| ivf8192_pq96x8_np1024     | 1024   | all   | 3000000 | wheelchair user crossing road             | 0.0188  | 2048  | 0.3547    | 0.2419    | 0.396    |
| ivf8192_pq96x8_np1024     | 1024   | all   | 3000000 | bicyclist passing by parked cars          | 0.0197  | 2048  | 0.4966    | 0.3279    | 0.424    |
| ivf8192_pq96x8_np1024     | 1024   | all   | 3000000 | child near a crosswalk                    | 0.0189  | 2048  | 0.3258    | 0.2275    | 0.410    |
| ivf8192_pq96x8_np1024     | 1024   | all   | 3000000 | dog on the sidewalk                       | 0.0179  | 2048  | 0.4130    | 0.2122    | 0.384    |
| ivf8192_pq96x8_np1024     | 1024   | all   | 3000000 | car turning left at an intersection       | 0.0194  | 2048  | 0.2577    | 0.1866    | 0.298    |
| ivf8192_pq96x8_np1024     | 1024   | all   | 3000000 | traffic cones on the street               | 0.0183  | 2048  | 0.4323    | 0.2897    | 0.445    |
| ivf8192_pq96x8_np1024     | 1024   | all   | 3000000 | construction zone with cones and barriers | 0.0185  | 2048  | 0.4138    | 0.2772    | 0.442    |
| ivf8192_pq96x8_np1024     | 1024   | all   | 3000000 | nighttime street with headlights          | 0.0179  | 2048  | 0.4625    | 0.3060    | 0.477    |
| ivf8192_pq96x8_np1024     | 1024   | all   | 3000000 | rainy day pedestrians with umbrellas      | 0.0187  | 2048  | 0.4730    | 0.2889    | 0.461    |
| ivf8192_pq96x8_np1024     | 1024   | all   | 3000000 | night                                     | 0.0175  | 2048  | 0.3684    | 0.3085    | 0.376    |
| ivf8192_pq96x8_np1024     | 1024   | all   | 3000000 | officer holding a sign                    | 0.0179  | 2048  | 0.3399    | 0.1554    | 0.385    |
| ivf8192_pq96x8_np1024     | 1024   | all   | 3000000 | officer signaling to stop                 | 0.0179  | 2048  | 0.4022    | 0.2140    | 0.418    |
| ivf8192_pq96x8_np1024     | 1024   | all   | 3000000 | officer waving to go                      | 0.0169  | 2048  | 0.3234    | 0.1555    | 0.396    |
| ivf8192_pq96x8_np1024     | 1024   | mads  | 21459   | pedestrian crossing the street            | 0.0168  | 0     | nan       | nan       | n/a      |
| ivf8192_pq96x8_np1024     | 1024   | mads  | 21459   | dog on the sidewalk                       | 0.0162  | 0     | nan       | nan       | n/a      |
| ivf8192_pq96x8_np1024     | 1024   | mads  | 21459   | car turning left at an intersection       | 0.0175  | 41    | 0.2435    | 0.1874    | 0.099    |
| ivf8192_pq96x8_np1024     | 1024   | mads  | 21459   | traffic cones on the street               | 0.0168  | 9     | 0.3545    | 0.2945    | 0.667    |
| ivf8192_pq96x8_np1024     | 1024   | mads  | 21459   | construction zone with cones and barriers | 0.0171  | 5     | 0.3224    | 0.2778    | 0.400    |
| ivf8192_pq96x8_np1024     | 1024   | mads  | 21459   | nighttime street with headlights          | 0.0164  | 5     | 0.3715    | 0.3085    | 0.417    |
| ivf8192_pq96x8_np1024     | 1024   | mads  | 21459   | rainy day pedestrians with umbrellas      | 0.0172  | 0     | nan       | nan       | n/a      |
| ivf8192_pq96x8_np1024     | 1024   | mads  | 21459   | night                                     | 0.0161  | 0     | nan       | nan       | n/a      |
| ivf8192_pq96x8_np1024     | 1024   | mads  | 21459   | officer holding a sign                    | 0.0165  | 21    | 0.2399    | 0.1576    | 0.404    |
| ivf8192_pq96x8_np1024     | 1024   | mads  | 21459   | officer signaling to stop                 | 0.0166  | 1     | 0.2454    | 0.2454    | 0.000    |
| ivf8192_pq96x8_np1024     | 1024   | mads  | 21459   | officer waving to go                      | 0.0166  | 3     | 0.1795    | 0.1671    | 0.046    |
| ivf8192_pq96x8_np1024     | 1024   | night | 534     | wheelchair user crossing road             | 0.0164  | 0     | nan       | nan       | n/a      |
| ivf8192_pq96x8_np1024     | 1024   | night | 534     | child near a crosswalk                    | 0.0165  | 2     | 0.2409    | 0.2402    | 0.500    |
| ivf8192_pq96x8_np1024     | 1024   | night | 534     | nighttime street with headlights          | 0.0165  | 6     | 0.3395    | 0.3088    | 0.400    |
| ivf8192_pq96x8_np1024     | 1024   | night | 534     | night                                     | 0.0161  | 13    | 0.3297    | 0.3098    | 0.385    |
| ivf8192_pq96x8_np1024     | 1024   | night | 534     | officer holding a sign                    | 0.0166  | 3     | 0.1764    | 0.1615    | 0.154    |
| ivf8192_pq96x8_np1024     | 1024   | night | 534     | officer signaling to stop                 | 0.0166  | 12    | 0.2620    | 0.2150    | 0.643    |
| ivf8192_pq96x8_np1024     | 1024   | night | 534     | officer waving to go                      | 0.0167  | 3     | 0.1630    | 0.1588    | 0.333    |
| ivf8192_pq96x8_np1024     | 1024   | sign  | 325     | pedestrian crossing the street            | 0.0169  | 0     | nan       | nan       | n/a      |
| ivf8192_pq96x8_np1024     | 1024   | sign  | 325     | child near a crosswalk                    | 0.0165  | 0     | nan       | nan       | n/a      |
| ivf8192_pq96x8_np1024     | 1024   | sign  | 325     | traffic cones on the street               | 0.0169  | 0     | nan       | nan       | n/a      |
| ivf8192_pq96x8_np1024     | 1024   | sign  | 325     | construction zone with cones and barriers | 0.0170  | 0     | nan       | nan       | n/a      |
| ivf8192_pq96x8_np2048     | 2048   | all   | 3000000 | pedestrian crossing the street            | 0.0321  | 2048  | 0.4145    | 0.2883    | 0.411    |
| ivf8192_pq96x8_np2048     | 2048   | all   | 3000000 | wheelchair user crossing road             | 0.0306  | 2048  | 0.3547    | 0.2419    | 0.396    |
| ivf8192_pq96x8_np2048     | 2048   | all   | 3000000 | bicyclist passing by parked cars          | 0.0314  | 2048  | 0.4966    | 0.3279    | 0.424    |
| ivf8192_pq96x8_np2048     | 2048   | all   | 3000000 | child near a crosswalk                    | 0.0303  | 2048  | 0.3258    | 0.2275    | 0.410    |
| ivf8192_pq96x8_np2048     | 2048   | all   | 3000000 | dog on the sidewalk                       | 0.0303  | 2048  | 0.4130    | 0.2122    | 0.384    |
| ivf8192_pq96x8_np2048     | 2048   | all   | 3000000 | car turning left at an intersection       | 0.0322  | 2048  | 0.2577    | 0.1866    | 0.299    |
| ivf8192_pq96x8_np2048     | 2048   | all   | 3000000 | traffic cones on the street               | 0.0310  | 2048  | 0.4323    | 0.2897    | 0.445    |
| ivf8192_pq96x8_np2048     | 2048   | all   | 3000000 | construction zone with cones and barriers | 0.0309  | 2048  | 0.4138    | 0.2772    | 0.442    |
| ivf8192_pq96x8_np2048     | 2048   | all   | 3000000 | nighttime street with headlights          | 0.0280  | 2048  | 0.4625    | 0.3060    | 0.477    |
| ivf8192_pq96x8_np2048     | 2048   | all   | 3000000 | rainy day pedestrians with umbrellas      | 0.0288  | 2048  | 0.4730    | 0.2889    | 0.461    |
| ivf8192_pq96x8_np2048     | 2048   | all   | 3000000 | night                                     | 0.0284  | 2048  | 0.3684    | 0.3085    | 0.376    |
| ivf8192_pq96x8_np2048     | 2048   | all   | 3000000 | officer holding a sign                    | 0.0283  | 2048  | 0.3399    | 0.1555    | 0.385    |
| ivf8192_pq96x8_np2048     | 2048   | all   | 3000000 | officer signaling to stop                 | 0.0281  | 2048  | 0.4022    | 0.2140    | 0.418    |
| ivf8192_pq96x8_np2048     | 2048   | all   | 3000000 | officer waving to go                      | 0.0286  | 2048  | 0.3234    | 0.1555    | 0.396    |
| ivf8192_pq96x8_np2048     | 2048   | mads  | 21459   | pedestrian crossing the street            | 0.0288  | 0     | nan       | nan       | n/a      |
| ivf8192_pq96x8_np2048     | 2048   | mads  | 21459   | dog on the sidewalk                       | 0.0279  | 0     | nan       | nan       | n/a      |
| ivf8192_pq96x8_np2048     | 2048   | mads  | 21459   | car turning left at an intersection       | 0.0299  | 43    | 0.2435    | 0.1874    | 0.103    |
| ivf8192_pq96x8_np2048     | 2048   | mads  | 21459   | traffic cones on the street               | 0.0288  | 9     | 0.3545    | 0.2945    | 0.667    |
| ivf8192_pq96x8_np2048     | 2048   | mads  | 21459   | construction zone with cones and barriers | 0.0290  | 5     | 0.3224    | 0.2778    | 0.400    |
| ivf8192_pq96x8_np2048     | 2048   | mads  | 21459   | nighttime street with headlights          | 0.0283  | 5     | 0.3715    | 0.3085    | 0.417    |
| ivf8192_pq96x8_np2048     | 2048   | mads  | 21459   | rainy day pedestrians with umbrellas      | 0.0291  | 0     | nan       | nan       | n/a      |
| ivf8192_pq96x8_np2048     | 2048   | mads  | 21459   | night                                     | 0.0282  | 0     | nan       | nan       | n/a      |
| ivf8192_pq96x8_np2048     | 2048   | mads  | 21459   | officer holding a sign                    | 0.0280  | 21    | 0.2399    | 0.1576    | 0.404    |
| ivf8192_pq96x8_np2048     | 2048   | mads  | 21459   | officer signaling to stop                 | 0.0280  | 1     | 0.2454    | 0.2454    | 0.000    |
| ivf8192_pq96x8_np2048     | 2048   | mads  | 21459   | officer waving to go                      | 0.0284  | 3     | 0.1795    | 0.1671    | 0.046    |
| ivf8192_pq96x8_np2048     | 2048   | night | 534     | wheelchair user crossing road             | 0.0276  | 0     | nan       | nan       | n/a      |
| ivf8192_pq96x8_np2048     | 2048   | night | 534     | child near a crosswalk                    | 0.0283  | 2     | 0.2409    | 0.2402    | 0.500    |
| ivf8192_pq96x8_np2048     | 2048   | night | 534     | nighttime street with headlights          | 0.0282  | 6     | 0.3395    | 0.3088    | 0.400    |
| ivf8192_pq96x8_np2048     | 2048   | night | 534     | night                                     | 0.0282  | 13    | 0.3297    | 0.3098    | 0.385    |
| ivf8192_pq96x8_np2048     | 2048   | night | 534     | officer holding a sign                    | 0.0280  | 3     | 0.1764    | 0.1615    | 0.154    |
| ivf8192_pq96x8_np2048     | 2048   | night | 534     | officer signaling to stop                 | 0.0283  | 12    | 0.2620    | 0.2150    | 0.643    |
| ivf8192_pq96x8_np2048     | 2048   | night | 534     | officer waving to go                      | 0.0286  | 3     | 0.1630    | 0.1588    | 0.333    |
| ivf8192_pq96x8_np2048     | 2048   | sign  | 325     | pedestrian crossing the street            | 0.0285  | 0     | nan       | nan       | n/a      |
| ivf8192_pq96x8_np2048     | 2048   | sign  | 325     | child near a crosswalk                    | 0.0283  | 0     | nan       | nan       | n/a      |
| ivf8192_pq96x8_np2048     | 2048   | sign  | 325     | traffic cones on the street               | 0.0290  | 0     | nan       | nan       | n/a      |
| ivf8192_pq96x8_np2048     | 2048   | sign  | 325     | construction zone with cones and barriers | 0.0293  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf8192_pq96x8  | 512    | all   | 3000000 | pedestrian crossing the street            | 0.0052  | 362   | 0.4198    | 0.1423    | 0.045    |
| opq96_768_ivf8192_pq96x8  | 512    | all   | 3000000 | wheelchair user crossing road             | 0.0047  | 179   | 0.3495    | 0.0644    | 0.009    |
| opq96_768_ivf8192_pq96x8  | 512    | all   | 3000000 | bicyclist passing by parked cars          | 0.0048  | 346   | 0.5224    | 0.0919    | 0.036    |
| opq96_768_ivf8192_pq96x8  | 512    | all   | 3000000 | child near a crosswalk                    | 0.0046  | 257   | 0.2969    | 0.0395    | 0.014    |
| opq96_768_ivf8192_pq96x8  | 512    | all   | 3000000 | dog on the sidewalk                       | 0.0047  | 618   | 0.4397    | 0.0663    | 0.115    |
| opq96_768_ivf8192_pq96x8  | 512    | all   | 3000000 | car turning left at an intersection       | 0.0047  | 387   | 0.2412    | 0.0774    | 0.027    |
| opq96_768_ivf8192_pq96x8  | 512    | all   | 3000000 | traffic cones on the street               | 0.0046  | 222   | 0.4492    | 0.1308    | 0.030    |
| opq96_768_ivf8192_pq96x8  | 512    | all   | 3000000 | construction zone with cones and barriers | 0.0046  | 256   | 0.4384    | 0.0748    | 0.029    |
| opq96_768_ivf8192_pq96x8  | 512    | all   | 3000000 | nighttime street with headlights          | 0.0047  | 512   | 0.4488    | 0.1493    | 0.092    |
| opq96_768_ivf8192_pq96x8  | 512    | all   | 3000000 | rainy day pedestrians with umbrellas      | 0.0046  | 265   | 0.4482    | 0.0978    | 0.037    |
| opq96_768_ivf8192_pq96x8  | 512    | all   | 3000000 | night                                     | 0.0046  | 360   | 0.3494    | 0.2264    | 0.049    |
| opq96_768_ivf8192_pq96x8  | 512    | all   | 3000000 | officer holding a sign                    | 0.0044  | 126   | 0.3268    | 0.0418    | 0.017    |
| opq96_768_ivf8192_pq96x8  | 512    | all   | 3000000 | officer signaling to stop                 | 0.0044  | 126   | 0.4054    | 0.0611    | 0.016    |
| opq96_768_ivf8192_pq96x8  | 512    | all   | 3000000 | officer waving to go                      | 0.0045  | 301   | 0.3663    | 0.0022    | 0.033    |
| opq96_768_ivf8192_pq96x8  | 512    | mads  | 21459   | pedestrian crossing the street            | 0.0046  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf8192_pq96x8  | 512    | mads  | 21459   | dog on the sidewalk                       | 0.0045  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf8192_pq96x8  | 512    | mads  | 21459   | car turning left at an intersection       | 0.0046  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf8192_pq96x8  | 512    | mads  | 21459   | traffic cones on the street               | 0.0045  | 1     | 0.3033    | 0.3033    | 0.111    |
| opq96_768_ivf8192_pq96x8  | 512    | mads  | 21459   | construction zone with cones and barriers | 0.0045  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf8192_pq96x8  | 512    | mads  | 21459   | nighttime street with headlights          | 0.0047  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf8192_pq96x8  | 512    | mads  | 21459   | rainy day pedestrians with umbrellas      | 0.0046  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf8192_pq96x8  | 512    | mads  | 21459   | night                                     | 0.0045  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf8192_pq96x8  | 512    | mads  | 21459   | officer holding a sign                    | 0.0044  | 1     | 0.1980    | 0.1980    | 0.019    |
| opq96_768_ivf8192_pq96x8  | 512    | mads  | 21459   | officer signaling to stop                 | 0.0043  | 1     | 0.1780    | 0.1780    | 0.000    |
| opq96_768_ivf8192_pq96x8  | 512    | mads  | 21459   | officer waving to go                      | 0.0044  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf8192_pq96x8  | 512    | night | 534     | wheelchair user crossing road             | 0.0044  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf8192_pq96x8  | 512    | night | 534     | child near a crosswalk                    | 0.0045  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf8192_pq96x8  | 512    | night | 534     | nighttime street with headlights          | 0.0046  | 3     | 0.3085    | 0.2726    | 0.133    |
| opq96_768_ivf8192_pq96x8  | 512    | night | 534     | night                                     | 0.0044  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf8192_pq96x8  | 512    | night | 534     | officer holding a sign                    | 0.0043  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf8192_pq96x8  | 512    | night | 534     | officer signaling to stop                 | 0.0043  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf8192_pq96x8  | 512    | night | 534     | officer waving to go                      | 0.0045  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf8192_pq96x8  | 512    | sign  | 325     | pedestrian crossing the street            | 0.0046  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf8192_pq96x8  | 512    | sign  | 325     | child near a crosswalk                    | 0.0045  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf8192_pq96x8  | 512    | sign  | 325     | traffic cones on the street               | 0.0045  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf8192_pq96x8  | 512    | sign  | 325     | construction zone with cones and barriers | 0.0045  | 0     | nan       | nan       | n/a      |
| ivf16384_pq96x8           | 1024   | all   | 3000000 | pedestrian crossing the street            | 0.0151  | 2048  | 0.4250    | 0.2920    | 0.418    |
| ivf16384_pq96x8           | 1024   | all   | 3000000 | wheelchair user crossing road             | 0.0147  | 2048  | 0.3837    | 0.2434    | 0.417    |
| ivf16384_pq96x8           | 1024   | all   | 3000000 | bicyclist passing by parked cars          | 0.0145  | 2048  | 0.5021    | 0.3287    | 0.435    |
| ivf16384_pq96x8           | 1024   | all   | 3000000 | child near a crosswalk                    | 0.0137  | 2048  | 0.3396    | 0.2278    | 0.426    |
| ivf16384_pq96x8           | 1024   | all   | 3000000 | dog on the sidewalk                       | 0.0131  | 2048  | 0.4418    | 0.2103    | 0.401    |
| ivf16384_pq96x8           | 1024   | all   | 3000000 | car turning left at an intersection       | 0.0139  | 2048  | 0.2619    | 0.1863    | 0.314    |
| ivf16384_pq96x8           | 1024   | all   | 3000000 | traffic cones on the street               | 0.0133  | 2048  | 0.4142    | 0.2862    | 0.454    |
| ivf16384_pq96x8           | 1024   | all   | 3000000 | construction zone with cones and barriers | 0.0134  | 2048  | 0.4387    | 0.2761    | 0.456    |
| ivf16384_pq96x8           | 1024   | all   | 3000000 | nighttime street with headlights          | 0.0131  | 2048  | 0.4830    | 0.3064    | 0.480    |
| ivf16384_pq96x8           | 1024   | all   | 3000000 | rainy day pedestrians with umbrellas      | 0.0135  | 2048  | 0.4678    | 0.2896    | 0.468    |
| ivf16384_pq96x8           | 1024   | all   | 3000000 | night                                     | 0.0128  | 2048  | 0.3881    | 0.3074    | 0.386    |
| ivf16384_pq96x8           | 1024   | all   | 3000000 | officer holding a sign                    | 0.0132  | 2048  | 0.3127    | 0.1548    | 0.380    |
| ivf16384_pq96x8           | 1024   | all   | 3000000 | officer signaling to stop                 | 0.0132  | 2048  | 0.3968    | 0.2132    | 0.431    |
| ivf16384_pq96x8           | 1024   | all   | 3000000 | officer waving to go                      | 0.0131  | 2048  | 0.3391    | 0.1543    | 0.401    |
| ivf16384_pq96x8           | 1024   | mads  | 21459   | pedestrian crossing the street            | 0.0135  | 1     | 0.3224    | 0.3224    | 0.500    |
| ivf16384_pq96x8           | 1024   | mads  | 21459   | dog on the sidewalk                       | 0.0127  | 0     | nan       | nan       | n/a      |
| ivf16384_pq96x8           | 1024   | mads  | 21459   | car turning left at an intersection       | 0.0139  | 107   | 0.2529    | 0.1865    | 0.216    |
| ivf16384_pq96x8           | 1024   | mads  | 21459   | traffic cones on the street               | 0.0132  | 9     | 0.3468    | 0.2909    | 0.556    |
| ivf16384_pq96x8           | 1024   | mads  | 21459   | construction zone with cones and barriers | 0.0135  | 3     | 0.3255    | 0.2920    | 0.300    |
| ivf16384_pq96x8           | 1024   | mads  | 21459   | nighttime street with headlights          | 0.0132  | 6     | 0.3585    | 0.3239    | 0.500    |
| ivf16384_pq96x8           | 1024   | mads  | 21459   | rainy day pedestrians with umbrellas      | 0.0131  | 1     | 0.3338    | 0.3338    | 0.500    |
| ivf16384_pq96x8           | 1024   | mads  | 21459   | night                                     | 0.0120  | 1     | 0.3074    | 0.3074    | 0.500    |
| ivf16384_pq96x8           | 1024   | mads  | 21459   | officer holding a sign                    | 0.0123  | 22    | 0.2209    | 0.1568    | 0.385    |
| ivf16384_pq96x8           | 1024   | mads  | 21459   | officer signaling to stop                 | 0.0125  | 0     | nan       | nan       | n/a      |
| ivf16384_pq96x8           | 1024   | mads  | 21459   | officer waving to go                      | 0.0120  | 6     | 0.2022    | 0.1574    | 0.092    |
| ivf16384_pq96x8           | 1024   | night | 534     | wheelchair user crossing road             | 0.0121  | 0     | nan       | nan       | n/a      |
| ivf16384_pq96x8           | 1024   | night | 534     | child near a crosswalk                    | 0.0122  | 0     | nan       | nan       | n/a      |
| ivf16384_pq96x8           | 1024   | night | 534     | nighttime street with headlights          | 0.0120  | 6     | 0.3404    | 0.3094    | 0.400    |
| ivf16384_pq96x8           | 1024   | night | 534     | night                                     | 0.0118  | 5     | 0.3574    | 0.3123    | 0.192    |
| ivf16384_pq96x8           | 1024   | night | 534     | officer holding a sign                    | 0.0122  | 6     | 0.1732    | 0.1550    | 0.462    |
| ivf16384_pq96x8           | 1024   | night | 534     | officer signaling to stop                 | 0.0121  | 10    | 0.2857    | 0.2165    | 0.643    |
| ivf16384_pq96x8           | 1024   | night | 534     | officer waving to go                      | 0.0121  | 2     | 0.1782    | 0.1714    | 0.222    |
| ivf16384_pq96x8           | 1024   | sign  | 325     | pedestrian crossing the street            | 0.0125  | 1     | 0.3714    | 0.3714    | 1.000    |
| ivf16384_pq96x8           | 1024   | sign  | 325     | child near a crosswalk                    | 0.0124  | 0     | nan       | nan       | n/a      |
| ivf16384_pq96x8           | 1024   | sign  | 325     | traffic cones on the street               | 0.0123  | 0     | nan       | nan       | n/a      |
| ivf16384_pq96x8           | 1024   | sign  | 325     | construction zone with cones and barriers | 0.0125  | 1     | 0.2766    | 0.2766    | 0.500    |
| opq96_768_ivf16384_pq96x8 | 1024   | all   | 3000000 | pedestrian crossing the street            | 0.0061  | 143   | 0.3952    | 0.1947    | 0.025    |
| opq96_768_ivf16384_pq96x8 | 1024   | all   | 3000000 | wheelchair user crossing road             | 0.0056  | 71    | 0.3433    | 0.0864    | 0.005    |
| opq96_768_ivf16384_pq96x8 | 1024   | all   | 3000000 | bicyclist passing by parked cars          | 0.0055  | 201   | 0.5199    | 0.1903    | 0.033    |
| opq96_768_ivf16384_pq96x8 | 1024   | all   | 3000000 | child near a crosswalk                    | 0.0053  | 218   | 0.3387    | 0.0957    | 0.028    |
| opq96_768_ivf16384_pq96x8 | 1024   | all   | 3000000 | dog on the sidewalk                       | 0.0052  | 192   | 0.4396    | 0.1697    | 0.046    |
| opq96_768_ivf16384_pq96x8 | 1024   | all   | 3000000 | car turning left at an intersection       | 0.0051  | 107   | 0.2577    | 0.0957    | 0.016    |
| opq96_768_ivf16384_pq96x8 | 1024   | all   | 3000000 | traffic cones on the street               | 0.0051  | 98    | 0.3675    | 0.1239    | 0.013    |
| opq96_768_ivf16384_pq96x8 | 1024   | all   | 3000000 | construction zone with cones and barriers | 0.0051  | 113   | 0.4377    | 0.0819    | 0.017    |
| opq96_768_ivf16384_pq96x8 | 1024   | all   | 3000000 | nighttime street with headlights          | 0.0052  | 264   | 0.4841    | 0.1560    | 0.051    |
| opq96_768_ivf16384_pq96x8 | 1024   | all   | 3000000 | rainy day pedestrians with umbrellas      | 0.0051  | 87    | 0.4558    | 0.1851    | 0.018    |
| opq96_768_ivf16384_pq96x8 | 1024   | all   | 3000000 | night                                     | 0.0052  | 72    | 0.3542    | 0.2337    | 0.008    |
| opq96_768_ivf16384_pq96x8 | 1024   | all   | 3000000 | officer holding a sign                    | 0.0052  | 80    | 0.3040    | 0.0548    | 0.009    |
| opq96_768_ivf16384_pq96x8 | 1024   | all   | 3000000 | officer signaling to stop                 | 0.0052  | 226   | 0.4096    | 0.0041    | 0.029    |
| opq96_768_ivf16384_pq96x8 | 1024   | all   | 3000000 | officer waving to go                      | 0.0051  | 226   | 0.3521    | 0.0107    | 0.026    |
| opq96_768_ivf16384_pq96x8 | 1024   | mads  | 21459   | pedestrian crossing the street            | 0.0050  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf16384_pq96x8 | 1024   | mads  | 21459   | dog on the sidewalk                       | 0.0051  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf16384_pq96x8 | 1024   | mads  | 21459   | car turning left at an intersection       | 0.0050  | 101   | 0.2577    | 0.0957    | 0.149    |
| opq96_768_ivf16384_pq96x8 | 1024   | mads  | 21459   | traffic cones on the street               | 0.0050  | 2     | 0.2968    | 0.2444    | 0.111    |
| opq96_768_ivf16384_pq96x8 | 1024   | mads  | 21459   | construction zone with cones and barriers | 0.0050  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf16384_pq96x8 | 1024   | mads  | 21459   | nighttime street with headlights          | 0.0051  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf16384_pq96x8 | 1024   | mads  | 21459   | rainy day pedestrians with umbrellas      | 0.0050  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf16384_pq96x8 | 1024   | mads  | 21459   | night                                     | 0.0049  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf16384_pq96x8 | 1024   | mads  | 21459   | officer holding a sign                    | 0.0051  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf16384_pq96x8 | 1024   | mads  | 21459   | officer signaling to stop                 | 0.0051  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf16384_pq96x8 | 1024   | mads  | 21459   | officer waving to go                      | 0.0051  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf16384_pq96x8 | 1024   | night | 534     | wheelchair user crossing road             | 0.0050  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf16384_pq96x8 | 1024   | night | 534     | child near a crosswalk                    | 0.0051  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf16384_pq96x8 | 1024   | night | 534     | nighttime street with headlights          | 0.0051  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf16384_pq96x8 | 1024   | night | 534     | night                                     | 0.0050  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf16384_pq96x8 | 1024   | night | 534     | officer holding a sign                    | 0.0050  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf16384_pq96x8 | 1024   | night | 534     | officer signaling to stop                 | 0.0051  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf16384_pq96x8 | 1024   | night | 534     | officer waving to go                      | 0.0051  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf16384_pq96x8 | 1024   | sign  | 325     | pedestrian crossing the street            | 0.0050  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf16384_pq96x8 | 1024   | sign  | 325     | child near a crosswalk                    | 0.0051  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf16384_pq96x8 | 1024   | sign  | 325     | traffic cones on the street               | 0.0050  | 0     | nan       | nan       | n/a      |
| opq96_768_ivf16384_pq96x8 | 1024   | sign  | 325     | construction zone with cones and barriers | 0.0050  | 0     | nan       | nan       | n/a      |
| hnsw32                    | 128    | all   | 3000000 | pedestrian crossing the street            | 0.0082  | 2048  | 0.4149    | 0.2569    | 0.365    |
| hnsw32                    | 128    | all   | 3000000 | wheelchair user crossing road             | 0.0068  | 2048  | 0.4993    | 0.2042    | 0.292    |
| hnsw32                    | 128    | all   | 3000000 | bicyclist passing by parked cars          | 0.0067  | 2048  | 0.5239    | 0.2988    | 0.424    |
| hnsw32                    | 128    | all   | 3000000 | child near a crosswalk                    | 0.0070  | 2048  | 0.3473    | 0.1944    | 0.311    |
| hnsw32                    | 128    | all   | 3000000 | dog on the sidewalk                       | 0.0056  | 1525  | 0.4248    | 0.0024    | 0.210    |
| hnsw32                    | 128    | all   | 3000000 | car turning left at an intersection       | 0.0072  | 2048  | 0.2552    | 0.1543    | 0.254    |
| hnsw32                    | 128    | all   | 3000000 | traffic cones on the street               | 0.0069  | 2048  | 0.4472    | 0.2406    | 0.428    |
| hnsw32                    | 128    | all   | 3000000 | construction zone with cones and barriers | 0.0069  | 2048  | 0.4569    | 0.2343    | 0.413    |
| hnsw32                    | 128    | all   | 3000000 | nighttime street with headlights          | 0.0065  | 2048  | 0.4941    | 0.2764    | 0.490    |
| hnsw32                    | 128    | all   | 3000000 | rainy day pedestrians with umbrellas      | 0.0062  | 2048  | 0.4544    | 0.2295    | 0.415    |
| hnsw32                    | 128    | all   | 3000000 | night                                     | 0.0067  | 2048  | 0.3707    | 0.2852    | 0.464    |
| hnsw32                    | 128    | all   | 3000000 | officer holding a sign                    | 0.0078  | 2048  | 0.3467    | 0.0887    | 0.146    |
| hnsw32                    | 128    | all   | 3000000 | officer signaling to stop                 | 0.0056  | 1845  | 0.4036    | 0.0010    | 0.089    |
| hnsw32                    | 128    | all   | 3000000 | officer waving to go                      | 0.0062  | 2048  | 0.3314    | 0.0335    | 0.140    |
| hnsw32                    | 128    | mads  | 21459   | pedestrian crossing the street            | 0.0066  | 1     | 0.3233    | 0.3233    | 0.500    |
| hnsw32                    | 128    | mads  | 21459   | dog on the sidewalk                       | 0.0056  | 0     | nan       | nan       | n/a      |
| hnsw32                    | 128    | mads  | 21459   | car turning left at an intersection       | 0.0070  | 134   | 0.2500    | 0.1552    | 0.250    |
| hnsw32                    | 128    | mads  | 21459   | traffic cones on the street               | 0.0062  | 3     | 0.3006    | 0.2763    | 0.333    |
| hnsw32                    | 128    | mads  | 21459   | construction zone with cones and barriers | 0.0069  | 2     | 0.3260    | 0.2707    | 0.200    |
| hnsw32                    | 128    | mads  | 21459   | nighttime street with headlights          | 0.0066  | 7     | 0.3692    | 0.3013    | 0.583    |
| hnsw32                    | 128    | mads  | 21459   | rainy day pedestrians with umbrellas      | 0.0064  | 1     | 0.3299    | 0.3299    | 0.500    |
| hnsw32                    | 128    | mads  | 21459   | night                                     | 0.0065  | 0     | nan       | nan       | n/a      |
| hnsw32                    | 128    | mads  | 21459   | officer holding a sign                    | 0.0067  | 6     | 0.2188    | 0.0940    | 0.058    |
| hnsw32                    | 128    | mads  | 21459   | officer signaling to stop                 | 0.0060  | 2     | 0.1780    | 0.0017    | 0.000    |
| hnsw32                    | 128    | mads  | 21459   | officer waving to go                      | 0.0054  | 7     | 0.1726    | 0.0602    | 0.031    |
| hnsw32                    | 128    | night | 534     | wheelchair user crossing road             | 0.0062  | 0     | nan       | nan       | n/a      |
| hnsw32                    | 128    | night | 534     | child near a crosswalk                    | 0.0068  | 0     | nan       | nan       | n/a      |
| hnsw32                    | 128    | night | 534     | nighttime street with headlights          | 0.0065  | 8     | 0.3351    | 0.2856    | 0.533    |
| hnsw32                    | 128    | night | 534     | night                                     | 0.0064  | 9     | 0.3319    | 0.2894    | 0.346    |
| hnsw32                    | 128    | night | 534     | officer holding a sign                    | 0.0066  | 1     | 0.1504    | 0.1504    | 0.000    |
| hnsw32                    | 128    | night | 534     | officer signaling to stop                 | 0.0054  | 0     | nan       | nan       | n/a      |
| hnsw32                    | 128    | night | 534     | officer waving to go                      | 0.0051  | 1     | 0.1012    | 0.1012    | 0.000    |
| hnsw32                    | 128    | sign  | 325     | pedestrian crossing the street            | 0.0063  | 1     | 0.3350    | 0.3350    | 1.000    |
| hnsw32                    | 128    | sign  | 325     | child near a crosswalk                    | 0.0059  | 0     | nan       | nan       | n/a      |
| hnsw32                    | 128    | sign  | 325     | traffic cones on the street               | 0.0059  | 0     | nan       | nan       | n/a      |
| hnsw32                    | 128    | sign  | 325     | construction zone with cones and barriers | 0.0058  | 0     | nan       | nan       | n/a      |
| hnsw64                    | 256    | all   | 3000000 | pedestrian crossing the street            | 0.0097  | 2048  | 0.4149    | 0.2862    | 0.500    |
| hnsw64                    | 256    | all   | 3000000 | wheelchair user crossing road             | 0.0088  | 2048  | 0.4993    | 0.2262    | 0.438    |
| hnsw64                    | 256    | all   | 3000000 | bicyclist passing by parked cars          | 0.0097  | 2048  | 0.5239    | 0.3230    | 0.500    |
| hnsw64                    | 256    | all   | 3000000 | child near a crosswalk                    | 0.0101  | 2048  | 0.3470    | 0.1978    | 0.290    |
| hnsw64                    | 256    | all   | 3000000 | dog on the sidewalk                       | 0.0067  | 2048  | 0.4248    | 0.1223    | 0.265    |
| hnsw64                    | 256    | all   | 3000000 | car turning left at an intersection       | 0.0092  | 2048  | 0.2552    | 0.1740    | 0.389    |
| hnsw64                    | 256    | all   | 3000000 | traffic cones on the street               | 0.0080  | 2048  | 0.4472    | 0.2742    | 0.500    |
| hnsw64                    | 256    | all   | 3000000 | construction zone with cones and barriers | 0.0078  | 2048  | 0.4569    | 0.2671    | 0.500    |
| hnsw64                    | 256    | all   | 3000000 | nighttime street with headlights          | 0.0064  | 2048  | 0.4941    | 0.2919    | 0.500    |
| hnsw64                    | 256    | all   | 3000000 | rainy day pedestrians with umbrellas      | 0.0070  | 2048  | 0.4544    | 0.2759    | 0.500    |
| hnsw64                    | 256    | all   | 3000000 | night                                     | 0.0066  | 2048  | 0.3707    | 0.2942    | 0.500    |
| hnsw64                    | 256    | all   | 3000000 | officer holding a sign                    | 0.0074  | 2048  | 0.3467    | 0.1221    | 0.236    |
| hnsw64                    | 256    | all   | 3000000 | officer signaling to stop                 | 0.0072  | 2048  | 0.4154    | 0.1975    | 0.468    |
| hnsw64                    | 256    | all   | 3000000 | officer waving to go                      | 0.0069  | 2048  | 0.3642    | 0.1360    | 0.407    |
| hnsw64                    | 256    | mads  | 21459   | pedestrian crossing the street            | 0.0074  | 1     | 0.3233    | 0.3233    | 0.500    |
| hnsw64                    | 256    | mads  | 21459   | dog on the sidewalk                       | 0.0054  | 0     | nan       | nan       | n/a      |
| hnsw64                    | 256    | mads  | 21459   | car turning left at an intersection       | 0.0075  | 150   | 0.2500    | 0.1754    | 0.327    |
| hnsw64                    | 256    | mads  | 21459   | traffic cones on the street               | 0.0070  | 4     | 0.3006    | 0.2763    | 0.444    |
| hnsw64                    | 256    | mads  | 21459   | construction zone with cones and barriers | 0.0074  | 2     | 0.3260    | 0.2860    | 0.200    |
| hnsw64                    | 256    | mads  | 21459   | nighttime street with headlights          | 0.0062  | 6     | 0.3692    | 0.3013    | 0.500    |
| hnsw64                    | 256    | mads  | 21459   | rainy day pedestrians with umbrellas      | 0.0068  | 1     | 0.3299    | 0.3299    | 0.500    |
| hnsw64                    | 256    | mads  | 21459   | night                                     | 0.0063  | 0     | nan       | nan       | n/a      |
| hnsw64                    | 256    | mads  | 21459   | officer holding a sign                    | 0.0072  | 2     | 0.1866    | 0.1861    | 0.038    |
| hnsw64                    | 256    | mads  | 21459   | officer signaling to stop                 | 0.0070  | 0     | nan       | nan       | n/a      |
| hnsw64                    | 256    | mads  | 21459   | officer waving to go                      | 0.0068  | 1     | 0.1365    | 0.1365    | 0.000    |
| hnsw64                    | 256    | night | 534     | wheelchair user crossing road             | 0.0070  | 0     | nan       | nan       | n/a      |
| hnsw64                    | 256    | night | 534     | child near a crosswalk                    | 0.0079  | 0     | nan       | nan       | n/a      |
| hnsw64                    | 256    | night | 534     | nighttime street with headlights          | 0.0061  | 10    | 0.3473    | 0.2949    | 0.667    |
| hnsw64                    | 256    | night | 534     | night                                     | 0.0063  | 9     | 0.3319    | 0.2947    | 0.346    |
| hnsw64                    | 256    | night | 534     | officer holding a sign                    | 0.0071  | 0     | nan       | nan       | n/a      |
| hnsw64                    | 256    | night | 534     | officer signaling to stop                 | 0.0070  | 6     | 0.2800    | 0.2156    | 0.429    |
| hnsw64                    | 256    | night | 534     | officer waving to go                      | 0.0068  | 1     | 0.1376    | 0.1376    | 0.000    |
| hnsw64                    | 256    | sign  | 325     | pedestrian crossing the street            | 0.0073  | 1     | 0.3350    | 0.3350    | 1.000    |
| hnsw64                    | 256    | sign  | 325     | child near a crosswalk                    | 0.0079  | 1     | 0.2216    | 0.2216    | 1.000    |
| hnsw64                    | 256    | sign  | 325     | traffic cones on the street               | 0.0070  | 1     | 0.2827    | 0.2827    | 1.000    |
| hnsw64                    | 256    | sign  | 325     | construction zone with cones and barriers | 0.0072  | 0     | nan       | nan       | n/a      |

# Semantic Search Benchmark — PR Curve (all slice)

_commit `68be5b2` · date 2026-03-22 20:29:03 · embeddings_dir: /path/to/wheel-data/benchmark · embeddings_size: 31.9 GB · k: 2048 · n_queries: 14 · n_indexes: 11 · annotations_db: /path/to/wheel-data/annotations_latest_schema.db_

| index                     | nprobe | k    | precision | recall |
| ------------------------- | ------ | ---- | --------- | ------ |
| ivf4096_pq96x8            | 256    | 10   | 1.000     | 0.002  |
| ivf4096_pq96x8            | 256    | 50   | 0.989     | 0.012  |
| ivf4096_pq96x8            | 256    | 100  | 0.989     | 0.024  |
| ivf4096_pq96x8            | 256    | 200  | 0.980     | 0.048  |
| ivf4096_pq96x8            | 256    | 500  | 0.953     | 0.116  |
| ivf4096_pq96x8            | 256    | 1000 | 0.896     | 0.219  |
| ivf4096_pq96x8            | 256    | 2048 | 0.793     | 0.397  |
| opq96_768_ivf4096_pq96x8  | 256    | 10   | 1.000     | 0.002  |
| opq96_768_ivf4096_pq96x8  | 256    | 50   | 1.000     | 0.012  |
| opq96_768_ivf4096_pq96x8  | 256    | 100  | 0.988     | 0.024  |
| opq96_768_ivf4096_pq96x8  | 256    | 200  | 0.930     | 0.045  |
| opq96_768_ivf4096_pq96x8  | 256    | 500  | 0.514     | 0.063  |
| opq96_768_ivf4096_pq96x8  | 256    | 1000 | 0.257     | 0.063  |
| opq96_768_ivf4096_pq96x8  | 256    | 2048 | 0.126     | 0.063  |
| ivf8192_pq96x8            | 512    | 10   | 1.000     | 0.002  |
| ivf8192_pq96x8            | 512    | 50   | 0.997     | 0.012  |
| ivf8192_pq96x8            | 512    | 100  | 0.989     | 0.024  |
| ivf8192_pq96x8            | 512    | 200  | 0.977     | 0.048  |
| ivf8192_pq96x8            | 512    | 500  | 0.951     | 0.116  |
| ivf8192_pq96x8            | 512    | 1000 | 0.910     | 0.222  |
| ivf8192_pq96x8            | 512    | 2048 | 0.816     | 0.408  |
| ivf8192_pq96x8_np1024     | 1024   | 10   | 1.000     | 0.002  |
| ivf8192_pq96x8_np1024     | 1024   | 50   | 0.997     | 0.012  |
| ivf8192_pq96x8_np1024     | 1024   | 100  | 0.989     | 0.024  |
| ivf8192_pq96x8_np1024     | 1024   | 200  | 0.977     | 0.048  |
| ivf8192_pq96x8_np1024     | 1024   | 500  | 0.950     | 0.116  |
| ivf8192_pq96x8_np1024     | 1024   | 1000 | 0.910     | 0.222  |
| ivf8192_pq96x8_np1024     | 1024   | 2048 | 0.818     | 0.409  |
| ivf8192_pq96x8_np2048     | 2048   | 10   | 1.000     | 0.002  |
| ivf8192_pq96x8_np2048     | 2048   | 50   | 0.997     | 0.012  |
| ivf8192_pq96x8_np2048     | 2048   | 100  | 0.989     | 0.024  |
| ivf8192_pq96x8_np2048     | 2048   | 200  | 0.977     | 0.048  |
| ivf8192_pq96x8_np2048     | 2048   | 500  | 0.950     | 0.116  |
| ivf8192_pq96x8_np2048     | 2048   | 1000 | 0.910     | 0.222  |
| ivf8192_pq96x8_np2048     | 2048   | 2048 | 0.818     | 0.409  |
| opq96_768_ivf8192_pq96x8  | 512    | 10   | 1.000     | 0.002  |
| opq96_768_ivf8192_pq96x8  | 512    | 50   | 0.981     | 0.012  |
| opq96_768_ivf8192_pq96x8  | 512    | 100  | 0.876     | 0.021  |
| opq96_768_ivf8192_pq96x8  | 512    | 200  | 0.639     | 0.031  |
| opq96_768_ivf8192_pq96x8  | 512    | 500  | 0.321     | 0.039  |
| opq96_768_ivf8192_pq96x8  | 512    | 1000 | 0.160     | 0.039  |
| opq96_768_ivf8192_pq96x8  | 512    | 2048 | 0.078     | 0.039  |
| ivf16384_pq96x8           | 1024   | 10   | 1.000     | 0.002  |
| ivf16384_pq96x8           | 1024   | 50   | 0.997     | 0.012  |
| ivf16384_pq96x8           | 1024   | 100  | 0.996     | 0.024  |
| ivf16384_pq96x8           | 1024   | 200  | 0.987     | 0.048  |
| ivf16384_pq96x8           | 1024   | 500  | 0.966     | 0.118  |
| ivf16384_pq96x8           | 1024   | 1000 | 0.923     | 0.225  |
| ivf16384_pq96x8           | 1024   | 2048 | 0.838     | 0.419  |
| opq96_768_ivf16384_pq96x8 | 1024   | 10   | 1.000     | 0.002  |
| opq96_768_ivf16384_pq96x8 | 1024   | 50   | 0.910     | 0.011  |
| opq96_768_ivf16384_pq96x8 | 1024   | 100  | 0.746     | 0.018  |
| opq96_768_ivf16384_pq96x8 | 1024   | 200  | 0.468     | 0.023  |
| opq96_768_ivf16384_pq96x8 | 1024   | 500  | 0.189     | 0.023  |
| opq96_768_ivf16384_pq96x8 | 1024   | 1000 | 0.094     | 0.023  |
| opq96_768_ivf16384_pq96x8 | 1024   | 2048 | 0.046     | 0.023  |
| hnsw32                    | 128    | 10   | 1.000     | 0.002  |
| hnsw32                    | 128    | 50   | 1.000     | 0.012  |
| hnsw32                    | 128    | 100  | 1.000     | 0.024  |
| hnsw32                    | 128    | 200  | 1.000     | 0.049  |
| hnsw32                    | 128    | 500  | 0.981     | 0.120  |
| hnsw32                    | 128    | 1000 | 0.885     | 0.216  |
| hnsw32                    | 128    | 2048 | 0.634     | 0.317  |
| hnsw64                    | 256    | 10   | 1.000     | 0.002  |
| hnsw64                    | 256    | 50   | 1.000     | 0.012  |
| hnsw64                    | 256    | 100  | 1.000     | 0.024  |
| hnsw64                    | 256    | 200  | 1.000     | 0.049  |
| hnsw64                    | 256    | 500  | 1.000     | 0.122  |
| hnsw64                    | 256    | 1000 | 0.997     | 0.244  |
| hnsw64                    | 256    | 2048 | 0.856     | 0.428  |
