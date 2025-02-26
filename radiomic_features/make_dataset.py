# python import

# package import
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# local import
from utils import logger


def get_grouped_columns(df: pd.DataFrame, col_fields: list):
    grouped_columns = {field: [] for field in col_fields}
    grouped_columns['other'] = []  # 用于存储不匹配任何字段的列名

    # 遍历所有列名，根据字段分组
    for col in df.columns:
        matched = False
        for field in col_fields:
            if field in col:  # 判断列名是否包含字段
                grouped_columns[field].append(col)
                matched = True
                break
        if not matched:  # 如果列名不匹配任何字段，则归为 "other"
            grouped_columns['other'].append(col)

    max_length = max(len(cols) for cols in grouped_columns.values())
    for field in grouped_columns:
        grouped_columns[field].extend([np.nan] * (max_length - len(grouped_columns[field])))

    # 将分组结果转换为 DataFrame
    result = pd.DataFrame(grouped_columns)
    return result


def get_raw_dataset(metadata_path: str, features_path: str):
    features = pd.read_csv(features_path).rename(columns={'image_path': 'path'})
    metadata = pd.read_csv(metadata_path)
    df = pd.merge(features, metadata, on='path', validate='1:1')

    fields = ['ngtdm', 'glszm', 'glrlm', 'gldm', 'glcm', 'firstorder', 'shape', 'label']

    grouped_df = get_grouped_columns(df, fields)
    logger.info(f"features:\n{grouped_df.to_string()}")

    # filter other columns
    filter_str = '|'.join(fields)
    result = df[df.columns[df.columns.str.contains(filter_str)]]
    return result


def split_dataset(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=seed)
    return train_df, test_df


if __name__ == "__main__":
    # image_types = ['Original']
    image_types = ['LoG', 'Wavelet']
    features_save_path = f"/home/user2/data/HCC-WCH/preprocessed/{image_types}_radiomics_features.csv"
    metadata_save_path = f"/home/user2/data/HCC-WCH/preprocessed/metadata.csv"

    features_df = get_raw_dataset(metadata_save_path, features_save_path)
    train, test = split_dataset(features_df)




