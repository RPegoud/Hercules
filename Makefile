# Makefile for Dockerized Atlas Project

# --- Configuration ---
# Docker image name and tag
IMAGE_NAME := atlas-dev
IMAGE_TAG := latest
FULL_IMAGE_NAME := $(IMAGE_NAME):$(IMAGE_TAG)

# Local project directory (current directory)
PROJECT_DIR := $(shell pwd)

# Mount path inside the container
CONTAINER_PROJECT_PATH := /app/Atlas

# Jupyter port
JUPYTER_PORT := 8888

# --- Docker Commands ---

.PHONY: build
build:
	@echo "Building Docker image $(FULL_IMAGE_NAME)..."
	docker build -t $(FULL_IMAGE_NAME) .

.PHONY: bash
bash: build # Ensure image is built before running
	@echo "Starting interactive bash session in container..."
	@echo "Your local directory $(PROJECT_DIR) is mounted at $(CONTAINER_PROJECT_PATH)"
	docker run -it --rm \
		-v "$(PROJECT_DIR):$(CONTAINER_PROJECT_PATH)" \
		-w "$(CONTAINER_PROJECT_PATH)" \
		$(FULL_IMAGE_NAME) bash

.PHONY: jupyter
jupyter: build # Ensure image is built before running
	@echo "Starting JupyterLab server..."
	@echo "Access it at http://localhost:$(JUPYTER_PORT) (or http://127.0.0.1:$(JUPYTER_PORT))"
	@echo "Your local directory $(PROJECT_DIR) is mounted at $(CONTAINER_PROJECT_PATH)"
	@echo "JupyterLab will start in $(CONTAINER_PROJECT_PATH)"
	docker run -it --rm \
		-p "$(JUPYTER_PORT):8888" \
		-v "$(PROJECT_DIR):$(CONTAINER_PROJECT_PATH)" \
		-w "$(CONTAINER_PROJECT_PATH)" \
		$(FULL_IMAGE_NAME) sh -c "/app/.venv/bin/activate && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir='$(CONTAINER_PROJECT_PATH)'"

.PHONY: stop
stop:
	@echo "Stopping any running containers for $(FULL_IMAGE_NAME)..."
	docker ps -q --filter ancestor=$(FULL_IMAGE_NAME) | xargs -r docker stop

.PHONY: clean
clean:
	@echo "Removing Docker image $(FULL_IMAGE_NAME)..."
	docker rmi $(FULL_IMAGE_NAME)

.PHONY: prune
prune:
	@echo "Pruning unused Docker images, containers, and networks..."
	docker system prune -f

# --- Help ---
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make build       Build the Docker image."
	@echo "  make bash        Run an interactive bash terminal inside the container."
	@echo "                   Your local ./Atlas directory will be mounted."
	@echo "  make jupyter     Run a JupyterLab server inside the container."
	@echo "                   Your local ./Atlas directory will be mounted and used as the notebook directory."
	@echo "  make stop        Stop any running containers based on this image."
	@echo "  make clean       Remove the Docker image."
	@echo "  make prune       Remove unused Docker objects to free up space."

# Default target
.DEFAULT_GOAL := help
