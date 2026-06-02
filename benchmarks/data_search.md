# Data Store Benchmark

_commit `207d715` · date 2026-03-01 21:57:34 · db: /path/to/wheel-data/annotations_latest_schema.db · db size: 21.1 GB_

| Query                            | Min (s)    | Mean (s)   | Max (s)    | Results  |
| -------------------------------- | ---------- | ---------- | ---------- | -------- |
| default_results                  | 0.000000   | 0.000000   | 0.000001   | 18011926 |
| get_clip_ids_for_data_sources    | 0.000000   | 0.000001   | 0.000004   | 5643911  |
| search_data_source_filter        | 3.359677   | 3.373866   | 3.392845   | 5636511  |
| search_annotation_filter         | 0.000001   | 0.000002   | 0.000004   | 3        |
| search_combined_filter           | 0.000002   | 0.000003   | 0.000005   | 1        |
| get_clips_dict_10                | 0.000017   | 0.033167   | 0.165735   | 10       |
| get_single_clip                  | 0.000005   | 0.000005   | 0.000008   | 1        |
| get_clip_ids_without_annotations | 30.777435  | 30.899952  | 31.073083  | 10120999 |
| search_annotation_and_filter     | 0.000002   | 0.000002   | 0.000004   | 0        |
| search_exclude_labels            | 7.486035   | 7.495326   | 7.498927   | 18011923 |
| search_without_annotations       | 139.955015 | 140.269142 | 140.547124 | 17506999 |
| get_clips_dict_100               | 0.000131   | 0.000136   | 0.000144   | 100      |
| options_lookup                   | 0.000012   | 0.000014   | 0.000019   | 210      |
| search_country_filter            | 2.455141   | 2.483282   | 2.519796   | 4268518  |
| search_times_with                | 0.036444   | 0.036699   | 0.037236   | 2014     |
| search_times_without             | 0.709736   | 0.712902   | 0.716750   | 502913   |
| search_label_type                | 0.261254   | 0.262676   | 0.265319   | 502097   |
| search_clipid                    | 0.000001   | 0.000001   | 0.000002   | 1        |
| search_annotation_multi_or       | 0.000004   | 0.000005   | 0.000006   | 53       |
| search_annotation_and_country    | 2.431651   | 2.439012   | 2.446885   | 2        |
| search_annotation_and_exclude    | 0.000002   | 0.000003   | 0.000004   | 3        |
