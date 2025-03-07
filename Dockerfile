# FROM ubuntu:22.04 as base
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Ensure no installs try to launch interactive screen
ARG DEBIAN_FRONTEND=noninteractive

# Add git
RUN apt-get update && apt-get install -y git build-essential pkg-config libhdf5-dev

# Add uv and use the system python (no need to make venv)
USER root
COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /bin/uv
ENV UV_SYSTEM_PYTHON=1

WORKDIR /home/app/atlas

COPY . .

RUN uv pip install -e .

EXPOSE 6006