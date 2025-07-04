PYTHON = .venv/bin/python3
SIF_NAME=vllm-aarch64-openai_v0.6.1.sif
DEF_FILE=container.def
HOST_PORT=8888
CONTAINER_PORT=8888
MOUNT_DIR=$(shell pwd)

# apptainer pull --tmpdir ~/apptainer_tmp docker://drikster80/vllm-aarch64-openai:v0.6.1
# apptainer build --sandbox vllm-hercules-aarch64 vllm-aarch64-openai_v0.6.1.sif
# apptainer shell --fakeroot --nv vllm-hercules-aarch64

build:
	apptainer build --sandbox $(SIF_NAME) $(DEF_FILE)

jupyter:
	apptainer exec --nv --bind $(MOUNT_DIR):/opt/project $(SIF_NAME) \
			jupyter lab --ip=0.0.0.0 --port=$(CONTAINER_PORT) --no-browser --allow-root --notebook-dir=/opt/project	

shell:
	apptainer shell --fakeroot --nv --bind /etc/pki/ca-trust/extracted/pem:/etc/pki/ca-trust/extracted/pem $(SIF_NAME)

gpu_pt:
	uv run accelerate launch --multi_gpu -m hercules.memory_pretraining