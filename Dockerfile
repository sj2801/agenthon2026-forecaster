# Agenthon 2026 Track 2 — claude-final: regime-vol joint Student-t forecaster, numbers only.
#
# Build (from this folder):  docker build --platform linux/amd64 -t agenthon-t2:claude-final .
# Smoke  (offline):          docker run --rm --network=none -v <unit-dir>:/input:ro -v <out>:/output \
#                              agenthon-t2:claude-final forecast --panels /input/panels/ --text /input/text/ \
#                              --asof <YYYY-MM-DD> --out /output/forecast.parquet
FROM --platform=linux/amd64 python:3.13-slim-bookworm

LABEL qfbench2.interface_version="2.0" \
      qfbench2.track="forecasting" \
      qfbench2.verb="forecast"

# Numbers-only by default: the House model is never called (no MODEL_* variables needed).
ARG NO_LLM="1"
ENV T2_NO_LLM=${NO_LLM} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt

# Everything is installed at build time; the evaluation network cannot download packages.
RUN python -m pip install --no-cache-dir \
      "numpy==2.1.3" \
      "pandas==2.2.3" \
      "pyarrow==18.1.0" \
      "jsonschema==4.23.0" \
      "qfbench2-common @ https://github.com/Agenthon-2026/Agenthon2026-public/archive/refs/tags/v2.6.0.tar.gz#subdirectory=common"

# Participant code only: no task data, answers or development artifacts.
# qfbench2_track_forecasting = organizer's MIT reference package (track2-forecasting-public
# @60509df4), used only for the official monthly observation-period step mapping.
COPY t2agent/ /opt/t2agent/
COPY qfbench2_track_forecasting/ /opt/qfbench2_track_forecasting/

RUN printf '#!/bin/sh\nexec python3 -m t2agent.cli "$@"\n' > /usr/local/bin/forecast \
 && chmod 0755 /usr/local/bin/forecast \
 && useradd --create-home --uid 1000 runner

USER runner
CMD ["forecast", "--help"]
