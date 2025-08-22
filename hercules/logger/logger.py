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
    "MAGENTA": Fore.MAGENTA,
}

STR_TO_STYLE = {
    "BRIGHT": Style.BRIGHT,
    "NORMAL": Style.NORMAL,
}


class Logger:
    def __init__(self, accelerator: Accelerator = None):
        self.accelerator = accelerator
        self.ts = datetime.now().strftime("%m-%d_%H-%M")
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
        self.log("Memory Llama", "blue", **kwargs)
        model.model.print_trainable_parameters()
        self.log(
            f"Memory Module size: {model.n_memory_params:.3e}",
            "blue",
            "normal",
            **kwargs,
        )

    def set_experiment_name(self, cfg, cfg_dict: DictConfig) -> None:
        if cfg.experiment.log_experiment:
            if cfg.experiment.run_name is None:
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

            else:
                run_name = f"{cfg.experiment.run_name}__{self.ts}"

            self.log(f"Run name: {run_name}", "magenta")

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
