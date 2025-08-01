import json
from datetime import datetime

from accelerate import Accelerator
from colorama import Fore, Style, init
from omegaconf import DictConfig
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
        trainable_memory = sum(
            p.numel() for p in model.neural_memory.parameters() if p.requires_grad
        )
        trainable_llama = sum(
            p.numel() for p in model.llama.parameters() if p.requires_grad
        )
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        self.log("Memory Llama", "blue", **kwargs)
        self.log(
            f"Trainable parameters:\nMemory module: {trainable_memory:.3e}\nLlama: {trainable_llama:.3e}",
            "blue",
            "normal",
            **kwargs,
        )
        self.log(
            f"Total trainable parameters: {trainable_llama+trainable_memory:.3e}",
            "blue",
            "normal",
            **kwargs,
        )
        self.log(f"Frozen parameters: {frozen:.3e}", "blue", "normal", **kwargs)

    def set_experiment_name(self, cfg, cfg_dict: DictConfig) -> None:
        if cfg.experiment.log_experiment:
            self.ts = datetime.now().strftime("%m-%d_%H-%M")
            if cfg.experiment.name == "babilong_pt":
                if cfg.experiment.use_global_split:
                    run_name = f"global_split_{cfg.experiment.global_split_test_size}__{self.ts}"
                    self.log(
                        f"Using global train/test split with test size: {cfg.experiment.global_split_test_size}",
                        "yellow",
                    )
                else:
                    run_name = f"train_{cfg.experiment.train_splits}__test_{cfg.experiment.test_splits}__{self.ts}"
                    self.log(
                        f"""Using specific train/test splits:
                Train: {cfg.experiment.train_splits}
                Test: {cfg.experiment.test_splits}""",
                        "yellow",
                    )
            elif cfg.experiment.name == "eduweb_pt":
                run_name = f"{cfg.experiment.name}_{cfg.experiment.num_train_samples}__{cfg.experiment.test_splits}__{self.ts}"
                self.log(
                    f"Eduweb pre-training: {cfg.experiment.num_train_samples} samples, Babilong test splits: {cfg.experiment.test_splits}",
                    "yellow",
                )
            self.accelerator.init_trackers(
                project_name="Hercules",
                config=cfg_dict,
                init_kwargs={
                    "wandb": {
                        "name": run_name,
                        "entity": "ryan_pgd",
                    }
                },
            )
