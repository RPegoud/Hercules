FROM python:3.13-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y git build-essential pkg-config libhdf5-dev curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

ADD . /app
WORKDIR /app
RUN uv sync --locked

RUN uv pip install -e .

EXPOSE 8888