<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Semantic Search Benchmark

_commit `24cc8a6` · date 2026-04-02 20:54:05 · embeddings dir: /path/to/wheel-data/ · size: 1.1 TB · index spec: IVF4096,PQ96x8 · ks: 4096 8192 16384 327678 · index size: 19996448_

| type       | query / clip_id                      | k      | seconds | count  | max_score | min_score |
| ---------- | ------------------------------------ | ------ | ------- | ------ | --------- | --------- |
| text→video | pedestrian crossing                  | 4096   | 0.0544  | 4096   | 0.3763    | 0.2732    |
| text→video | sharp left turn                      | 4096   | 0.0536  | 4096   | 0.2615    | 0.1837    |
| text→video | sharp right turn                     | 4096   | 0.0537  | 4096   | 0.2446    | 0.1537    |
| text→video | lane change on highway               | 4096   | 0.0557  | 4096   | 0.3420    | 0.2870    |
| text→video | u-turn in residential area           | 4096   | 0.0523  | 4096   | 0.3841    | 0.3111    |
| text→video | parallel parking                     | 4096   | 0.0543  | 4096   | 0.4066    | 0.3050    |
| text→video | merging onto highway                 | 4096   | 0.0528  | 4096   | 0.4068    | 0.3300    |
| text→video | roundabout navigation                | 4096   | 0.0511  | 4096   | 0.3904    | 0.2831    |
| text→video | red traffic light                    | 4096   | 0.2171  | 4096   | 0.3718    | 0.2998    |
| text→video | stop sign                            | 4096   | 0.0549  | 4096   | 0.3498    | 0.2527    |
| text→video | construction zone with lane shifts   | 4096   | 0.0512  | 4096   | 0.4422    | 0.3276    |
| text→video | road works ahead                     | 4096   | 0.0466  | 4096   | 0.4062    | 0.2947    |
| text→video | highway driving                      | 4096   | 0.0552  | 4096   | 0.3802    | 0.2964    |
| text→video | rain                                 | 4096   | 0.0513  | 4096   | 0.4558    | 0.3477    |
| text→video | heavy rain reduces visibility        | 4096   | 0.0514  | 4096   | 0.5016    | 0.3767    |
| text→video | night driving                        | 4096   | 0.0512  | 4096   | 0.4827    | 0.4234    |
| text→video | fog                                  | 4096   | 0.0516  | 4096   | 0.4892    | 0.3601    |
| text→video | snow                                 | 4096   | 0.0517  | 4096   | 0.3980    | 0.3208    |
| text→video | cyclist on the road                  | 4096   | 0.0517  | 4096   | 0.4564    | 0.3170    |
| text→video | emergency vehicle with sirens        | 4096   | 0.0564  | 4096   | 0.4517    | 0.3173    |
| text→video | school zone with children            | 4096   | 0.0523  | 4096   | 0.3567    | 0.2225    |
| text→video | hard braking                         | 4096   | 0.0582  | 4096   | 0.2773    | 0.2032    |
| text→video | stop and go traffic                  | 4096   | 0.0553  | 4096   | 0.3908    | 0.2884    |
| text→video | vehicle driving the wrong way        | 4096   | 0.0543  | 4096   | 0.4037    | 0.3139    |
| text→video | animal crossing the road             | 4096   | 0.0533  | 4096   | 0.3226    | 0.2334    |
| text→video | debris on the road                   | 4096   | 0.0508  | 4096   | 0.3346    | 0.2250    |
| text→video | flooded road                         | 4096   | 0.0521  | 4096   | 0.4598    | 0.3274    |
| text→video | fallen tree blocking the road        | 4096   | 0.0479  | 4096   | 0.3638    | 0.2558    |
| text→video | car accident                         | 4096   | 0.0549  | 4096   | 0.3583    | 0.2734    |
| text→video | level crossing with train            | 4096   | 0.0504  | 4096   | 0.4214    | 0.2994    |
| text→video | tunnel entrance                      | 4096   | 0.0498  | 4096   | 0.4756    | 0.3479    |
| text→video | icy road                             | 4096   | 0.0513  | 4096   | 0.3911    | 0.3012    |
| text→video | glare from the sun                   | 4096   | 0.0495  | 4096   | 0.4870    | 0.3423    |
| clip→video | cdcfb35f-0031-4e41-8d43-8c729ccf6326 | 4096   | 0.0421  | 4096   | 0.9604    | 0.7482    |
| clip→video | ee76a44e-0087-4afd-be52-401eab2205ae | 4096   | 0.0414  | 4096   | 1.0068    | 0.7280    |
| text→video | pedestrian crossing                  | 8192   | 0.0670  | 8192   | 0.3763    | 0.2587    |
| text→video | sharp left turn                      | 8192   | 0.2316  | 8192   | 0.2615    | 0.1747    |
| text→video | sharp right turn                     | 8192   | 0.0691  | 8192   | 0.2446    | 0.1429    |
| text→video | lane change on highway               | 8192   | 0.0699  | 8192   | 0.3420    | 0.2791    |
| text→video | u-turn in residential area           | 8192   | 0.0654  | 8192   | 0.3841    | 0.3000    |
| text→video | parallel parking                     | 8192   | 0.0665  | 8192   | 0.4066    | 0.2924    |
| text→video | merging onto highway                 | 8192   | 0.0637  | 8192   | 0.4068    | 0.3183    |
| text→video | roundabout navigation                | 8192   | 0.0629  | 8192   | 0.3904    | 0.2670    |
| text→video | red traffic light                    | 8192   | 0.0638  | 8192   | 0.3718    | 0.2880    |
| text→video | stop sign                            | 8192   | 0.0645  | 8192   | 0.3498    | 0.2407    |
| text→video | construction zone with lane shifts   | 8192   | 0.0617  | 8192   | 0.4422    | 0.3118    |
| text→video | road works ahead                     | 8192   | 0.0590  | 8192   | 0.4062    | 0.2751    |
| text→video | highway driving                      | 8192   | 0.0670  | 8192   | 0.3802    | 0.2856    |
| text→video | rain                                 | 8192   | 0.0620  | 8192   | 0.4558    | 0.3313    |
| text→video | heavy rain reduces visibility        | 8192   | 0.0615  | 8192   | 0.5016    | 0.3535    |
| text→video | night driving                        | 8192   | 0.0621  | 8192   | 0.4827    | 0.4138    |
| text→video | fog                                  | 8192   | 0.2465  | 8192   | 0.4892    | 0.3431    |
| text→video | snow                                 | 8192   | 0.0676  | 8192   | 0.3980    | 0.3122    |
| text→video | cyclist on the road                  | 8192   | 0.0647  | 8192   | 0.4564    | 0.2875    |
| text→video | emergency vehicle with sirens        | 8192   | 0.0669  | 8192   | 0.4517    | 0.2975    |
| text→video | school zone with children            | 8192   | 0.0620  | 8192   | 0.3567    | 0.2027    |
| text→video | hard braking                         | 8192   | 0.0692  | 8192   | 0.2773    | 0.1930    |
| text→video | stop and go traffic                  | 8192   | 0.0670  | 8192   | 0.3908    | 0.2786    |
| text→video | vehicle driving the wrong way        | 8192   | 0.0653  | 8192   | 0.4037    | 0.3044    |
| text→video | animal crossing the road             | 8192   | 0.0659  | 8192   | 0.3226    | 0.2231    |
| text→video | debris on the road                   | 8192   | 0.0642  | 8192   | 0.3346    | 0.2123    |
| text→video | flooded road                         | 8192   | 0.0632  | 8192   | 0.4598    | 0.3060    |
| text→video | fallen tree blocking the road        | 8192   | 0.0606  | 8192   | 0.3638    | 0.2423    |
| text→video | car accident                         | 8192   | 0.0669  | 8192   | 0.3583    | 0.2617    |
| text→video | level crossing with train            | 8192   | 0.0621  | 8192   | 0.4214    | 0.2707    |
| text→video | tunnel entrance                      | 8192   | 0.0624  | 8192   | 0.4756    | 0.3330    |
| text→video | icy road                             | 8192   | 0.2314  | 8192   | 0.3911    | 0.2878    |
| text→video | glare from the sun                   | 8192   | 0.0642  | 8192   | 0.4870    | 0.3223    |
| clip→video | cdcfb35f-0031-4e41-8d43-8c729ccf6326 | 8192   | 0.0524  | 8192   | 0.9604    | 0.7077    |
| clip→video | ee76a44e-0087-4afd-be52-401eab2205ae | 8192   | 0.0520  | 8192   | 1.0068    | 0.6930    |
| text→video | pedestrian crossing                  | 16384  | 0.0909  | 16384  | 0.3763    | 0.2430    |
| text→video | sharp left turn                      | 16384  | 0.0895  | 16384  | 0.2615    | 0.1650    |
| text→video | sharp right turn                     | 16384  | 0.0963  | 16384  | 0.2446    | 0.1321    |
| text→video | lane change on highway               | 16384  | 0.0951  | 16384  | 0.3420    | 0.2702    |
| text→video | u-turn in residential area           | 16384  | 0.0897  | 16384  | 0.3841    | 0.2877    |
| text→video | parallel parking                     | 16384  | 0.0900  | 16384  | 0.4066    | 0.2770    |
| text→video | merging onto highway                 | 16384  | 0.2624  | 16384  | 0.4068    | 0.3039    |
| text→video | roundabout navigation                | 16384  | 0.0916  | 16384  | 0.3904    | 0.2483    |
| text→video | red traffic light                    | 16384  | 0.0929  | 16384  | 0.3718    | 0.2736    |
| text→video | stop sign                            | 16384  | 0.0902  | 16384  | 0.3498    | 0.2273    |
| text→video | construction zone with lane shifts   | 16384  | 0.0864  | 16384  | 0.4422    | 0.2932    |
| text→video | road works ahead                     | 16384  | 0.0838  | 16384  | 0.4062    | 0.2524    |
| text→video | highway driving                      | 16384  | 0.0897  | 16384  | 0.3802    | 0.2733    |
| text→video | rain                                 | 16384  | 0.2555  | 16384  | 0.4558    | 0.3102    |
| text→video | heavy rain reduces visibility        | 16384  | 0.0872  | 16384  | 0.5016    | 0.3225    |
| text→video | night driving                        | 16384  | 0.0865  | 16384  | 0.4827    | 0.4025    |
| text→video | fog                                  | 16384  | 0.0891  | 16384  | 0.4892    | 0.3209    |
| text→video | snow                                 | 16384  | 0.0895  | 16384  | 0.3980    | 0.3021    |
| text→video | cyclist on the road                  | 16384  | 0.0875  | 16384  | 0.4564    | 0.2527    |
| text→video | emergency vehicle with sirens        | 16384  | 0.0886  | 16384  | 0.4517    | 0.2761    |
| text→video | school zone with children            | 16384  | 0.0865  | 16384  | 0.3567    | 0.1811    |
| text→video | hard braking                         | 16384  | 0.2680  | 16384  | 0.2773    | 0.1822    |
| text→video | stop and go traffic                  | 16384  | 0.0939  | 16384  | 0.3908    | 0.2675    |
| text→video | vehicle driving the wrong way        | 16384  | 0.0897  | 16384  | 0.4037    | 0.2938    |
| text→video | animal crossing the road             | 16384  | 0.0906  | 16384  | 0.3226    | 0.2116    |
| text→video | debris on the road                   | 16384  | 0.0913  | 16384  | 0.3346    | 0.1990    |
| text→video | flooded road                         | 16384  | 0.0874  | 16384  | 0.4598    | 0.2806    |
| text→video | fallen tree blocking the road        | 16384  | 0.0850  | 16384  | 0.3638    | 0.2274    |
| text→video | car accident                         | 16384  | 0.0896  | 16384  | 0.3583    | 0.2487    |
| text→video | level crossing with train            | 16384  | 0.2564  | 16384  | 0.4214    | 0.2376    |
| text→video | tunnel entrance                      | 16384  | 0.0896  | 16384  | 0.4756    | 0.3144    |
| text→video | icy road                             | 16384  | 0.0882  | 16384  | 0.3911    | 0.2718    |
| text→video | glare from the sun                   | 16384  | 0.0866  | 16384  | 0.4870    | 0.3013    |
| clip→video | cdcfb35f-0031-4e41-8d43-8c729ccf6326 | 16384  | 0.0725  | 16384  | 0.9604    | 0.6223    |
| clip→video | ee76a44e-0087-4afd-be52-401eab2205ae | 16384  | 0.0728  | 16384  | 1.0068    | 0.6395    |
| text→video | pedestrian crossing                  | 327678 | 1.4347  | 327678 | 0.3763    | 0.1467    |
| text→video | sharp left turn                      | 327678 | 1.5329  | 327678 | 0.2615    | 0.1059    |
| text→video | sharp right turn                     | 327678 | 1.3392  | 327678 | 0.2446    | 0.0764    |
| text→video | lane change on highway               | 327678 | 1.3501  | 327678 | 0.3420    | 0.2094    |
| text→video | u-turn in residential area           | 327678 | 1.5129  | 327678 | 0.3841    | 0.2027    |
| text→video | parallel parking                     | 327678 | 1.3378  | 327678 | 0.4066    | 0.1769    |
| text→video | merging onto highway                 | 327678 | 1.5061  | 327678 | 0.4068    | 0.1930    |
| text→video | roundabout navigation                | 327678 | 1.3422  | 327678 | 0.3904    | 0.1325    |
| text→video | red traffic light                    | 327678 | 1.3467  | 327678 | 0.3718    | 0.1619    |
| text→video | stop sign                            | 327678 | 1.5667  | 327678 | 0.3498    | 0.1331    |
| text→video | construction zone with lane shifts   | 327678 | 1.3251  | 327678 | 0.4422    | 0.0783    |
| text→video | road works ahead                     | 327678 | 1.3531  | 327678 | 0.4062    | 0.1352    |
| text→video | highway driving                      | 327678 | 1.3224  | 327678 | 0.3802    | 0.2002    |
| text→video | rain                                 | 327678 | 1.5237  | 327678 | 0.4558    | 0.1971    |
| text→video | heavy rain reduces visibility        | 327678 | 1.3172  | 327678 | 0.5016    | 0.1744    |
| text→video | night driving                        | 327678 | 1.4872  | 327678 | 0.4827    | 0.1992    |
| text→video | fog                                  | 327678 | 1.3229  | 327678 | 0.4892    | 0.0969    |
| text→video | snow                                 | 327678 | 1.2986  | 327678 | 0.3980    | 0.1737    |
| text→video | cyclist on the road                  | 327678 | 1.3106  | 327678 | 0.4564    | 0.0961    |
| text→video | emergency vehicle with sirens        | 327678 | 1.4606  | 327678 | 0.4517    | 0.1464    |
| text→video | school zone with children            | 327678 | 1.3261  | 327678 | 0.3567    | 0.0805    |
| text→video | hard braking                         | 327678 | 1.3493  | 327678 | 0.2773    | 0.1208    |
| text→video | stop and go traffic                  | 327678 | 1.5020  | 327678 | 0.3908    | 0.1984    |
| text→video | vehicle driving the wrong way        | 327678 | 1.2923  | 327678 | 0.4037    | 0.2209    |
| text→video | animal crossing the road             | 327678 | 1.2559  | 327678 | 0.3226    | 0.1364    |
| text→video | debris on the road                   | 327678 | 1.3245  | 327678 | 0.3346    | 0.1242    |
| text→video | flooded road                         | 327678 | 1.5057  | 327678 | 0.4598    | 0.1429    |
| text→video | fallen tree blocking the road        | 327678 | 1.3356  | 327678 | 0.3638    | 0.1355    |
| text→video | car accident                         | 327678 | 1.4462  | 327678 | 0.3583    | 0.1495    |
| text→video | level crossing with train            | 327678 | 1.3234  | 327678 | 0.4214    | 0.0908    |
| text→video | tunnel entrance                      | 327678 | 1.3096  | 327678 | 0.4756    | 0.1398    |
| text→video | icy road                             | 327678 | 1.3170  | 327678 | 0.3911    | 0.1389    |
| text→video | glare from the sun                   | 327678 | 1.5333  | 327678 | 0.4870    | 0.1856    |
| clip→video | cdcfb35f-0031-4e41-8d43-8c729ccf6326 | 327678 | 1.2465  | 327678 | 0.9604    | 0.3767    |
| clip→video | ee76a44e-0087-4afd-be52-401eab2205ae | 327678 | 1.2105  | 327678 | 1.0068    | 0.3886    |