import json

from colorama import Fore, Style, init
from transformers import AutoModelForCausalLM


def log_config(config: dict):
    init(autoreset=True)
    print(f"{Fore.GREEN}{Style.BRIGHT}Config:")
    print(
        f"{Fore.GREEN}{Style.BRIGHT}Hyperparameters:"
        f"{Style.NORMAL}{json.dumps(config, sort_keys=True, indent=4)}{Style.RESET_ALL}"
    )


def log_memory_model(model: AutoModelForCausalLM):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(
        f"""{Fore.BLUE}{Style.BRIGHT}Memory Llama:"
{Style.NORMAL}Trainable parameters: {trainable:.3e}
Frozen parameters: {frozen:.3e}{Style.RESET_ALL}"""
    )
