# local import
import os
import sys
import logging
from datetime import datetime
# package import
import hydra
import pandas as pd
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
# local import
from module.mnist_data_module import MNISTDataModule
from module.example_module import ExampleModule
from module.monai_data_module import MonaiDataModule
from utils import callbacks
from utils.util import get_multi_attr

# 获取 logger
logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.2",
    config_path=os.getenv('CONFIGS_LOCATION', 'config'),
    config_name="config",
)
def main(cfg: DictConfig):
    """
    Main function to run the training and testing pipeline using Hydra for configuration management.

    Args:
        cfg (DictConfig): Configuration dictionary provided by Hydra.
    """
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

    # if you want to use your own callbacks, you can add them to utils/callbacks.py
    # they will be imported by get_multi_attr
    callback_lt = get_multi_attr([pl.callbacks, callbacks], cfg.get("callbacks"))

    trainer = pl.Trainer(
        **cfg.get("trainer"),
        logger=tb_logger,
        callbacks=callback_lt,
    )
    logger.info("trainer built.")

    # set seed
    seed = cfg.get("dataset", {}).get("split", {}).get("seed", 42)
    pl.seed_everything(seed, workers=True)

    # build data Module
    dataset_config = cfg.get("dataset")
    datamodule = MonaiDataModule(**dataset_config) if dataset_config is not None else MNISTDataModule()
    logger.info("data module built.")

    # build model
    model = ExampleModule(**cfg.get("model"), **cfg.get("optimizer"), **cfg.get("lr_scheduler", {}),
                          **cfg.get("criterion"))
    logger.info("model built.")

    # train & test model
    trainer.fit(model, datamodule, ckpt_path=cfg.get("ckpt_path", None))
    logger.info("training finished.")

    result = trainer.test(model, datamodule)
    logger.info(f"test result: {result}")
    pd.DataFrame(result).to_csv(os.path.join(work_dir, 'test_result.csv'))

    result = trainer.test(model, datamodule, ckpt_path="best")
    logger.info(f"test result: {result}")
    pd.DataFrame(result).to_csv(os.path.join(work_dir, 'best_result.csv'))
    # trainer.reset_train_dataloader()  # hydra will not reset dataloader automatically
    # trainer.reset_val_dataloader()  # if you use multirun, you need to reset dataloader manually


if __name__ == "__main__":
    main()
