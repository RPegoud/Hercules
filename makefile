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

pt_bl: # pre-training on Babilong
	uv run accelerate launch --multi_gpu -m hercules.bl_memory_pretraining 

pt_bl_ls: # pre-training on babilong with wandb loging and model saving
	uv run accelerate launch --multi_gpu -m hercules.bl_memory_pretraining \
	experiment.save_model=True \
	experiment.log_experiment=True

pt_ew: # pre-training on Eduweb
	uv run accelerate launch --multi_gpu -m hercules.eduweb_memory_pretraining 

pt_ew_ls: # pre-training on babilong with wandb loging and model saving
	uv run accelerate launch --multi_gpu -m hercules.eduweb_memory_pretraining \
	experiment.save_model=True \
	experiment.log_experiment=True

control: # control experiment (babilong without neural memory)
	uv run accelerate launch --multi_gpu -m hercules.babilong_control_experiment