# local import
import os
import sys
import logging
from datetime import datetime
# package import
import torch
import hydra
import pandas as pd
import lightning.pytorch as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import TensorBoardLogger
# local import
from module.mnist_data_module import MNISTDataModule
from module.only_cls_module import ExampleModule
from module.monai_data_module import MonaiDataModule
from utils import callbacks
from utils.util import get_multi_attr
from utils.log import log_exception

# 获取 logger
logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.2",
    config_path=os.getenv('CONFIGS_LOCATION', 'config'),
    config_name="vit",
)
@log_exception(logger=logger)
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

    if HydraConfig.get().mode == hydra.types.RunMode.RUN:
        work_dir = HydraConfig.get().run.dir
    else:
        work_dir = os.path.join(HydraConfig.get().sweep.dir,
                                HydraConfig.get().sweep.subdir)

    cfg = OmegaConf.to_container(cfg, resolve=True)

    # set seed
    seed = cfg.get("dataset", {}).get("manual_seed", 42)
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # build trainer
    tb_logger = TensorBoardLogger(save_dir=work_dir)
    trainer_config = cfg.get("trainer")
    
    # if you want to use your own callbacks, you can add them to utils/callbacks.py
    # they will be imported by get_multi_attr
    callback_lt = get_multi_attr([pl.callbacks, callbacks], cfg.get("callbacks"))

    trainer = pl.Trainer(
        **trainer_config,
        logger=tb_logger,
        callbacks=callback_lt,
    )
    logger.info("trainer built.")

    # build data Module
    if cfg.get('mnist', False):  # +mnist=True in command line
        datamodule = MNISTDataModule()
    else:
        dataset_config = cfg.get("dataset")
        metadata_cfg, split_cfg = dataset_config.get("metadata"), dataset_config.get("split")
        load_cfg, loader_cfg = dataset_config.get("load"), dataset_config.get("loader")
        datamodule = MonaiDataModule(metadata_cfg, split_cfg, load_cfg, loader_cfg,
                                     num_classes=cfg.get("model").get("model_params").get("num_classes"))
    logger.info("data module built.")

    # build model
    model_config, criterion_config = cfg.get("model"), cfg.get("criterion")
    optimizer_config, lr_scheduler_config = cfg.get("optimizer"), cfg.get("lr_scheduler", {})
    model = ExampleModule(**model_config, **criterion_config, **optimizer_config, **lr_scheduler_config)
    logger.info("model built.")

    # train & test model
    if cfg.get('test_mode', False):
        import glob
        # Find all directories starting with 'version_'
        trial_dir = cfg.get('ckpt_path')
        if os.path.isdir(trial_dir):
            version_dirs = glob.glob(os.path.join(trial_dir, '**', 'version_*'), recursive=True)
            if not version_dirs:
                version_dirs = [trial_dir]
            # Initialize an empty DataFrame to store all results
            all_results = pd.DataFrame()

            # Iterate over each version directory
            for count, version_dir in enumerate(version_dirs):
                ckpt_dir = os.path.join(version_dir, 'checkpoints')
                if os.path.exists(ckpt_dir):
                    # Find all .ckpt files in the directory
                    ckpt_files = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))
                    for ct, ckpt_file in enumerate(ckpt_files):
                        # Perform testing
                        version_number = os.path.basename(version_dir).split('_')[-1]
                        # Create a new TensorBoardLogger with a specific version
                        tb_logger = TensorBoardLogger(save_dir=work_dir,
                                                      version=f'{count}_{ct}_version_{version_number}')
                        # Update the trainer with the new logger
                        test_trainer = pl.Trainer(
                            **trainer_config,
                            logger=tb_logger,
                            callbacks=callback_lt,
                        )
                        result = test_trainer.test(model, datamodule, ckpt_path=ckpt_file)
                        # Convert result to DataFrame and add experiment details
                        df = pd.DataFrame(result)
                        df['experiment'] = os.path.basename(version_dir)
                        df['ckpt'] = ckpt_file
                        # Append to all_results DataFrame
                        all_results = pd.concat([all_results, df], ignore_index=True)
                        logger.info(f"Testing finished for {ckpt_file}.")

            # Save all results to a single CSV file in trial_dir
            all_results.to_csv(os.path.join(work_dir, 'all_test_results.csv'))
        else:
            result = trainer.test(model, datamodule, ckpt_path=cfg.get("ckpt_path"))
            pd.DataFrame(result).to_csv(os.path.join(trainer.logger.log_dir, 'test_result.csv'))
            logger.info("testing finished.")
    else:
        trainer.fit(model, datamodule, ckpt_path=cfg.get("ckpt_path", None))
        logger.info("training finished.")

        ckpt_path = os.path.join(trainer.logger.log_dir, 'checkpoints')
        version_number = os.path.basename(trainer.logger.log_dir).split('_')[-1]
        for v, ckpt_file in enumerate(os.listdir(ckpt_path)):
            # Create a new TensorBoardLogger with a specific version
            tb_logger = TensorBoardLogger(save_dir=work_dir, version=f'version_{version_number}_test_{v}')
            test_trainer_cfg = trainer_config.copy()
            test_trainer_cfg['max_epochs'] = 30000  # if test needs train simple model
            # Update the trainer with the new logger
            test_trainer = pl.Trainer(
                **test_trainer_cfg,
                logger=tb_logger,
                callbacks=callback_lt,
            )

            result = test_trainer.test(model, datamodule, ckpt_path=os.path.join(ckpt_path, ckpt_file))
            base_name = os.path.basename(ckpt_file)
            pd.DataFrame(result).to_csv(os.path.join(test_trainer.logger.log_dir, f'{base_name}_result.csv'))
            logger.info(f"testing finished for {base_name}.")
    # trainer.reset_train_dataloader()  # hydra will not reset dataloader automatically
    # trainer.reset_val_dataloader()  # if you use multirun, you need to reset dataloader manually


if __name__ == "__main__":
    main()
