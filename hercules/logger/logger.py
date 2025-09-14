import json
from datetime import datetime

from accelerate import Accelerator
from colorama import Fore, Style, init
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM
import os

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
        if model.use_lora:
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

    def log_metrics(
        self,
        model,
        train_causal_loss,
        state,
        cfg: DictConfig,
    ):
        # log training loss and learning rate
        if self.accelerator.is_main_process and cfg.experiment.log_experiment:
            logs = {}
            logs["eduweb_causal_loss"] = train_causal_loss
            logs["learning_rate"] = state.scheduler.get_lr()[0]
            logs["epoch"] = state.epoch
            logs["step"] = state.global_step

            if cfg.memory_llama.track_memory_statistics:
                logs = self.log_memory_metrics(model, logs)

            self.accelerator.log(logs)

    def log_memory_metrics(self, model, logs):
        memory_stats = []
        for layer_id, layer in enumerate(model.layers):
            layer = model.layers[layer_id]
            if hasattr(layer, "last_stats") and layer.last_stats is not None:
                memory_stats.append(layer.last_stats.__dict__)

        if memory_stats:
            metrics = {
                "gate_bias": "Gate Bias",
                "gate_mean": "Gate Mean",
                "gate_std": "Gate Std",
                "gate_min": "Gate Min",
                "gate_max": "Gate Max",
                "mac_proj_memory_w_norm": "MAC Proj Memory Weight Norm",
                "global_avg_memory_contrib_ratio": "Global Average Memory Contribution Ratio",
                "per_token_avg_memory_contrib_ratio": "Average Memory Contribution Ratio per Token",
            }

            for metric_key, metric_name in metrics.items():
                layer_values = {}

                for s in memory_stats:
                    layer_id = s["layer"]
                    layer_values[f"Layer {layer_id}"] = s[metric_key]

                if layer_values:
                    logs[f"Memory Stats/{metric_name}"] = layer_values

        return logs

    def save_regular_checkpoint(
        self,
        train_causal_loss,
        state,
        cfg: DictConfig,
    ):
        if (state.global_step) % cfg.experiment.save_every == 0:
            num_recent_to_keep = cfg.experiment.get("save_recent_n", 3)

            recent_ckpt_dir = os.path.join(cfg.experiment.ckpt_dir, "recent")
            os.makedirs(recent_ckpt_dir, exist_ok=True)

            save_path = os.path.join(
                recent_ckpt_dir, f"recent_checkpoint_{state.recent_ckpt_idx}.pt"
            )

            self.log(
                f"Saving recent checkpoint for step {state.global_step} to {save_path}",
                "green",
                main_process=True,
            )
            state.model.save(save_path)
            with open(os.path.join(save_path, "metadata.txt"), "w") as f:
                f.write(
                    f"Training loss: {train_causal_loss}\nGlobal step: {state.global_step}"
                )

            # update the index for the next save
            state.recent_ckpt_idx = (state.recent_ckpt_idx + 1) % num_recent_to_keep

    def log_val_loss_and_save(self, avg_val_loss, state, cfg):
        self.log(f"Average validation loss: {avg_val_loss:.3f}", "yellow")
        if state.lowest_val_loss is None or avg_val_loss < state.lowest_val_loss:
            state.lowest_val_loss = avg_val_loss
            state.model.save(cfg.experiment.ckpt_dir)
            with open(
                os.path.join(cfg.experiment.ckpt_dir, "loss_value.txt"), "w"
            ) as f:
                f.write(f"Best loss: {state.lowest_val_loss}")

            self.log(
                f"Saved Checkpoint {state.global_step} [BEST LOSS = {state.lowest_val_loss:.3f}] under {cfg.experiment.ckpt_dir}",
                "green",
                main_process=True,
            )
