import hydra
from omegaconf import DictConfig, OmegaConf
from utils import log_config


@hydra.main(
    config_path="config",
    config_name="base.yaml",
    version_base="1.3",
)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    log_config(cfg_dict)


if __name__ == "__main__":
    main()
