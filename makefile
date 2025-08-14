PYTHON = .venv/bin/python3
SIF_NAME=vllm-aarch64-openai_v0.6.1.sif
DEF_FILE=container.def
HOST_PORT=8888
CONTAINER_PORT=8888
MOUNT_DIR=$(shell pwd)

# singularity pull --tmpdir ~/singularity_tmp docker://drikster80/vllm-aarch64-openai:v0.6.1
# singularity build --sandbox vllm-hercules-aarch64 vllm-aarch64-openai_v0.6.1.sif
# singularity shell --fakeroot --nv vllm-hercules-aarch64

build:
	singularity build --sandbox --fakeroot $(SIF_NAME) $(DEF_FILE)

shell:
	singularity exec --fakeroot --nv --cleanenv \
	--bind /etc/pki/ca-trust/extracted/pem:/etc/pki/ca-trust/extracted/pem \
	$(SIF_NAME) bash -c "cd Hercules && exec bash"

pt_ew: # pre-training on Eduweb
	accelerate launch -m hercules.scripts.eduweb_memory_pretraining 

pt_ew_ls: # pre-training on babilong with wandb loging and model saving
	accelerate launch -m hercules.scripts.eduweb_memory_pretraining \
	experiment.save_model=True \
	experiment.log_experiment=True
	
baseline:
	accelerate launch -m hercules.scripts.llama_baseline \
	experiment.save_model=True \
	experiment.log_experiment=True


# pt_ew: # pre-training on Eduweb
# 	uv run --active accelerate launch -m hercules.scripts.eduweb_memory_pretraining 

# pt_ew_ls: # pre-training on babilong with wandb loging and model saving
# 	uv run --active accelerate launch -m hercules.scripts.eduweb_memory_pretraining \
# 	experiment.save_model=True \
# 	experiment.log_experiment=True

# baseline:
# 	uv run accelerate launch -m hercules.scripts.llama_baseline \
# 	experiment.save_model=True \
# 	experiment.log_experiment=True