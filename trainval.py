import hydra
import os
import sys
import logging
import pandas as pd
from datetime import datetime
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
# local import
from module.data_modules import LoadedDataModule
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

    # 2. split features and metadata
    metadata_path = './preprocessed/metadata.csv'
    features_save_path = 'exp/2025-03-01_22-46-38_job/0/radiomics_features.csv'
    features_df = get_raw_dataset(metadata_path, features_save_path)
    train, test = split_dataset(features_df)

    # 3. lasso regression
    alpha = cfg.get('alpha')
    seed = cfg.get('seed')

    train_df, test_df = train.copy(), test.copy()
    model_info = train_logistic_regression(train_df, alpha=alpha, seed=seed)
    validation_results = validate_logistic_regression(test_df, model_info)

    output_file = os.path.join(work_dir, "validation_results.txt")
    with open(output_file, "w") as f:
        f.write("Validation Results:\n")
        f.write(f"Accuracy: {validation_results['accuracy']:.4f}\n")
        f.write(f"Precision: {validation_results['precision']:.4f}\n")
        f.write(f"Recall: {validation_results['recall']:.4f}\n")
        f.write(f"F1 Score: {validation_results['f1']:.4f}\n")
        f.write(f"AUC: {validation_results['auc']:.4f}\n")
        f.write("\nSignificance Summary (Test set):\n")
        f.write(validation_results['significance_summary'] + "\n")
        f.write(
            f"\nPearson Correlation between new feature and label (Test set): "
            f"{validation_results['pearson_correlation']:.4f}\n")
        f.write(f"Pearson p-value: {validation_results['pearson_p_value']:.4f}\n")


if __name__ == "__main__":
    main()
