PYTHON = .venv/bin/python3
SIF_NAME=vllm-aarch64-openai_v0.6.1.sif
DEF_FILE=container.def
HOST_PORT=8888
CONTAINER_PORT=8888
MOUNT_DIR=$(shell pwd)

build:
	singularity build --sandbox --fakeroot $(SIF_NAME) $(DEF_FILE)

shell:
	singularity exec --fakeroot --nv --cleanenv \
	--bind /etc/pki/ca-trust/extracted/pem:/etc/pki/ca-trust/extracted/pem \
	$(SIF_NAME) bash -c "cd Hercules && exec bash"

pt_ew: # pre-training on Eduweb
	accelerate launch --mixed_precision=bf16 -m hercules.scripts.eduweb_memory_pretraining 

pt_ew_ls: # pre-training on Eduweb with wandb loging and model saving
	accelerate launch --mixed_precision=bf16 -m hercules.scripts.eduweb_memory_pretraining \
	experiment.log_experiment=True
	
baseline: # raw llama baseline on babilong
	accelerate launch --mixed_precision=bf16 -m hercules.scripts.llama_baseline

ft_baseline: # finetune llama baseline on babilong
	accelerate launch --mixed_precision=bf16 -m hercules.scripts.llama_finetune_baseline

# memory llama without finetuning (validates that memory llama behaves like llama before training)
memory_llama_no_ft: 
	accelerate launch --mixed_precision=bf16 -m hercules.scripts.memory_llama_no_finetune