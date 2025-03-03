import hydra
import os
import sys
import logging
import pandas as pd
from multiprocessing import Pool

from functools import partial
from datetime import datetime
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
# local import
from module.data_modules import LoadedDataModule
from radiomic_features.radiomic_features import extract_features
from radiomic_features.make_dataset import get_raw_dataset, split_dataset
from radiomic_features.lasso_regression import train_logistic_regression, validate_logistic_regression

# 获取 logger
logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.2",
    config_path=os.getenv('CONFIGS_LOCATION', 'config'),
    config_name="pyradiomics",
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

    cfg = OmegaConf.to_container(cfg, resolve=True)

    # build trainer
    if HydraConfig.get().mode == hydra.types.RunMode.RUN:
        work_dir = HydraConfig.get().run.dir
    else:
        work_dir = os.path.join(HydraConfig.get().sweep.dir,
                                HydraConfig.get().sweep.subdir)

    # build data Module
    dataset_config = cfg.get("dataset")
    eval_name = ['is_valid_label', 'is_valid_file', 'grouped_attribute']
    for name in eval_name:
        if dataset_config.get(name, None) is not None:
            # str to lambda function
            dataset_config[name] = eval(dataset_config[name])

    # don't use preprocess
    dataset_config.update({'use_preprocessed': False})
    datamodule = LoadedDataModule(**dataset_config)
    datamodule.prepare_data()
    logger.info("dataloader built.")

    # 1. extract features
    metadata_path = './preprocessed/metadata.csv'
    metadata = pd.read_csv(metadata_path)
    file_paths = metadata["path"].tolist()
    settings = cfg.get("settings")
    image_types = cfg.get("image_types")
    threshold = cfg.get("threshold")
    partial_extract_features = partial(extract_features, settings=settings,
                                       image_types=image_types, threshold=threshold, logger=logger)

    # Use multiprocessing to extract features in parallel
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(partial_extract_features, file_paths)

    # Combine all DataFrames into one
    features_df = pd.concat(results, ignore_index=True)

    # Save features to CSV file
    features_save_path = os.path.join(work_dir, "radiomics_features.csv")
    features_df.to_csv(features_save_path, index=False)
    logger.info(f"Radiomics features saved to {features_save_path}")

    # 2. split features and metadata
    alpha = cfg.get('alpha')
    seed = cfg.get('seed')
    features_df = get_raw_dataset(metadata_path, features_save_path)
    train, test = split_dataset(features_df, seed=seed)

    # 3. lasso regression
    train_df, test_df = train.copy(), test.copy()
    output_file = os.path.join(work_dir, "validation_results.txt")
    model_info = train_logistic_regression(train_df, output_file, alpha=alpha, seed=seed)
    validate_logistic_regression(test_df, output_file, model_info)


if __name__ == "__main__":
    main()
