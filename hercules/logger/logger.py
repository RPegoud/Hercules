import json

from accelerate import Accelerator
from colorama import Fore, Style, init
from transformers import AutoModelForCausalLM

COLORS_TO_FORE = {
    "GREEN": Fore.GREEN,
    "BLUE": Fore.BLUE,
    "RED": Fore.RED,
    "CYAN": Fore.CYAN,
    "YELLOW": Fore.YELLOW,
    "WHITE": Fore.WHITE,
}

STR_TO_STYLE = {
    "BRIGHT": Style.BRIGHT,
    "NORMAL": Style.NORMAL,
}


class Logger:
    def __init__(self, accelerator: Accelerator = None):
        self.accelerator = accelerator
        init(autoreset=True)

    def log(
        self,
        message: str,
        color: str = "white",
        style: str = "bright",
        main_process: bool = True,  # ensures the message is logged a single time when accelerate is used
    ):

        if self.accelerator.is_main_process:
            print(
                f"{COLORS_TO_FORE[color.upper()]}{STR_TO_STYLE[style.upper()]}{message}"
            )
        else:
            if not main_process:
                print(
                    f"{COLORS_TO_FORE[color.upper()]}{STR_TO_STYLE[style.upper()]}{message}"
                )

    def log_config(self, config: dict, **kwargs):
        self.log("Config:", "green", **kwargs)
        self.log(
            f"{json.dumps(config, sort_keys=True, indent=4)}",
            "green",
            "normal",
            **kwargs,
        )

    def log_memory_model(self, model: AutoModelForCausalLM, **kwargs):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        self.log("Memory Llama", "blue", **kwargs)
        self.log(f"Trainable parameters: {trainable:.3e}", "blue", "normal", **kwargs)
        self.log(f"Frozen parameters: {frozen:.3e}", "blue", "normal", **kwargs)
