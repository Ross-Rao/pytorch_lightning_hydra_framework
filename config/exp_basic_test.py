import hydra
import os
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import logging

# 获取 logger
logger = logging.getLogger('hydra')


@hydra.main(
    version_base="1.2",
    config_path=os.getenv('CONFIGS_LOCATION', os.getcwd()),
    config_name="hydra_basic",
)
def main(cfg: DictConfig):
    if HydraConfig.get().mode == hydra.types.RunMode.RUN:
        work_dir = HydraConfig.get().run.dir
    else:
        work_dir = os.path.join(
            HydraConfig.get().sweep.dir, HydraConfig.get().sweep.subdir
        )
    logger.info(work_dir)
    logger.info(OmegaConf.to_yaml(cfg))
    return 42


if __name__ == "__main__":
    main()
