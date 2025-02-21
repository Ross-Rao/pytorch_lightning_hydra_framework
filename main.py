import hydra
import os
import sys
import logging
from datetime import datetime
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from torch.utils.data import DataLoader
# local import
from module.example_module import ExampleModule
from module.dataModule.processed_dataset import ProcessedDataset
from utils import callbacks
from utils.util import get_multi_attr
from utils.file_io import dir_to_df, preprocess_data

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

    # if you want to use your own callbacks, you can add them to utils/callbacks.py
    # they will be imported by get_multi_attr
    callback_lt = get_multi_attr([pl.callbacks, callbacks], cfg.get("callbacks"))

    trainer = pl.Trainer(
        **cfg.get("trainer"),
        logger=tb_logger,
        callbacks=callback_lt,
    )
    logger.info("trainer built.")

    # preprocess data (this part needs specific implementation)
    dataset_cfg = cfg.get("dataset")
    processed_data_root = dataset_cfg.get('preprocess_save_path')
    inline_files = ['train.pt', 'val.pt', 'test.pt', 'train_targets.pt', 'val_targets.pt', 'test_targets.pt']

    if not all([os.path.exists(os.path.join(processed_data_root, file)) for file in inline_files]):
        logger.info("Missing processed data, Preprocessing...")

        metadata = dir_to_df(**dataset_cfg.get('metadata'))  # build metadata csv for checking
        metadata['label'] = metadata['label'].astype(float)

        if not os.path.exists(processed_data_root):
            os.makedirs(processed_data_root)
        metadata.to_csv(os.path.join(processed_data_root, 'metadata.csv'), index=False)
        preprocess_data(metadata, save_path=processed_data_root, **dataset_cfg.get('preprocess'))

        logger.info("Data preprocessing finished.")

    # build dataloader
    train_dataset = ProcessedDataset(processed_data_root, 'train')
    val_dataset = ProcessedDataset(processed_data_root, 'val')
    test_dataset = ProcessedDataset(processed_data_root, 'test')
    train_loader = DataLoader(train_dataset, **cfg.get("train_loader"))
    val_loader = DataLoader(val_dataset, **cfg.get("val_loader"))
    test_loader = DataLoader(test_dataset, **cfg.get("test_loader"))
    logger.info("dataloader built.")

    # build model
    model = ExampleModule(**cfg.get("model"), **cfg.get("optimizer"), **cfg.get("lr_scheduler"),
                          criterion=cfg.get("criterion"), train_loader=train_loader,
                          val_loader=val_loader, test_loader=test_loader)
    logger.info("model built.")

    # train model
    trainer.fit(model)
    logger.info("training finished.")
    # trainer.reset_train_dataloader()  # hydra will not reset dataloader automatically
    # trainer.reset_val_dataloader()  # if you use multirun, you need to reset dataloader manually


if __name__ == "__main__":
    main()
