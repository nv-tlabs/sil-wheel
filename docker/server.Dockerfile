# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Interactive SIL-Wheel server: serves the UI + search over a prebuilt
# wheel-data dir. Long-lived; mount wheel-data + config.yaml and expose a port.
#
#   docker build -f docker/base.Dockerfile   -t silwheel:base .
#   docker build -f docker/server.Dockerfile -t silwheel:server .
#   docker run --gpus all --rm -p 8012:8012 \
#       -v /path/to/wheel-data-physical-ai:/data/wheel-data:ro \
#       silwheel:server /data/wheel-data/config.yaml
#
# GPU is optional: query-time text/image encoders (sentence-transformers,
# SigLIP) use it when present and fall back to CPU otherwise. With no embeddings
# the server still starts (stores degrade to an empty stand-in). The port comes
# from config.yaml's `bindto`; EXPOSE below is documentation only.

FROM silwheel:base

EXPOSE 8012

ENTRYPOINT ["python", "/opt/sil-wheel/scripts/launch_server.py"]
CMD ["--help"]
