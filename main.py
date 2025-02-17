import hydra
import os
import sys
import logging
from datetime import datetime
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
# local import
from module.example_module import ExampleModule


# 获取 logger
logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.2",
    config_path=os.getenv('CONFIGS_LOCATION', 'config'),
    config_name="config",
)
def main(cfg: DictConfig):
    # print the config
    script = os.path.basename(sys.argv[0])
    script_name = os.path.splitext(script)[0]
    args = sys.argv[1:]
    conda_env = os.getenv('CONDA_DEFAULT_ENV', 'N/A')
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    logger.info(f"Script Name: {script_name}")
    logger.info(f"Arguments: {args}")
    logger.info(f"Conda Environment: {conda_env}")
    logger.info(f"Start Time: {start_time}")
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    # build trainer
    if HydraConfig.get().mode == hydra.types.RunMode.RUN:
        work_dir = HydraConfig.get().run.dir
    else:
        work_dir = os.path.join(HydraConfig.get().sweep.dir,
                                HydraConfig.get().sweep.subdir)

    cfg = OmegaConf.to_container(cfg, resolve=True)
    tb_logger = TensorBoardLogger(save_dir=work_dir)

    callbacks = []  # if you want to use your own callbacks, you can add them here
    for key in cfg.get("callbacks").keys():
        callback = getattr(pl.callbacks, key)(**cfg.get("callbacks")[key])
        callbacks.append(callback)

    trainer = pl.Trainer(
        **cfg.get("trainer"),
        logger=tb_logger,
        callbacks=callbacks,
    )

    # build model
    model = ExampleModule(**cfg.get("model"), **cfg.get("optimizer"), **cfg.get("lr_scheduler"),
                          criterion=cfg.get("criterion"))
    return 42


if __name__ == "__main__":
    main()
