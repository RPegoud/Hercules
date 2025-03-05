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

# TODO: replace with project name
WORKDIR /home/app/template

COPY . .

RUN uv pip install -e .

ARG USE_CUDA=false
RUN if [ "$USE_CUDA" = true ] ; \
    then uv pip install jax[cuda12]==0.4.30 ; \
    fi

EXPOSE 6006