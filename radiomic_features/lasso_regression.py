# python import
# package import
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# local import
from utils import logger
from make_dataset import get_raw_dataset, split_dataset


def logistic_regression(train_df, test_df, alpha=0.01, seed=42):
    x_train = train_df.drop(columns=['label'])
    y_train = train_df['label']
    x_test = test_df.drop(columns=['label'])
    y_test = test_df['label']

    # 1. Z 分数归一化（标准化）
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    # 2. 使用 Lasso 回归筛选特征
    lasso = Lasso(alpha=alpha, random_state=seed)
    lasso.fit(x_train_scaled, y_train)
    selected_features = x_train.columns[abs(lasso.coef_) > 0]
    logger.info("Lasso selected features:\n" + selected_features.to_series().to_string())

    # 3. 筛选训练集和测试集中的特征
    x_train_selected = x_train[selected_features]
    x_test_selected = x_test[selected_features]

    # 4. 计算新特征（Lasso 系数加权的特征组合）
    new_feature_train = x_train_selected.dot(lasso.coef_[abs(lasso.coef_) > 0])
    new_feature_test = x_test_selected.dot(lasso.coef_[abs(lasso.coef_) > 0])

    # 5. 使用 Logistic Regression 进行分类
    log_reg = LogisticRegression(random_state=seed)
    log_reg.fit(new_feature_train.values.reshape(-1, 1), y_train)

    # 6. 预测测试集
    y_pred = log_reg.predict(new_feature_test.values.reshape(-1, 1))
    y_pred_proba = log_reg.predict_proba(new_feature_test.values.reshape(-1, 1))[:, 1]

    # 7. 计算性能指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1: {f1:.4f}")
    logger.info(f"AUC: {auc:.4f}")

    # 8. 显著性检验（使用 Statsmodels Logit 模型）
    X_new_train = sm.add_constant(new_feature_train)  # 添加截距项
    logit_model = sm.Logit(y_train, X_new_train).fit(disp=0)
    logger.info("Significance test for the new feature:")
    logger.info(logit_model.summary())

    # 9. 计算新特征与目标变量的 Pearson 相关性
    correlation, p_value = pearsonr(new_feature_train, y_train)
    logger.info(f"Pearson Correlation between new feature and label: {correlation:.4f}, p-value: {p_value:.4f}")


def feature_selection(train_df, threshold):
    x = train_df.drop(columns=['label'])
    y = train_df['label']
    # 1. Z 分数归一化（标准化）
    scaler = StandardScaler()
    scaled = scaler.fit_transform(x)

    # 将标准化后的特征转换回 DataFrame
    scaled_df = pd.DataFrame(scaled, columns=x.columns)

    # 2. 使用皮尔逊相关系数（PCC）筛选特征
    correlations = scaled_df.corrwith(y)
    logger.info(f"特征与目标变量的皮尔逊相关系数：\n{correlations.sort_values().to_string()}")

    selected_features = correlations[abs(correlations) > threshold].index

    return selected_features.to_series()


if __name__ == '__main__':
    # image_types = ['Original']
    image_types = ['LoG', 'Wavelet']
    features_save_path = f"./{image_types}_radiomics_features.csv"
    metadata_save_path = f"/home/user2/data/HCC-WCH/preprocessed/metadata.csv"

    features_df = get_raw_dataset(metadata_save_path, features_save_path)
    train, test = split_dataset(features_df)

    selected = feature_selection(train, 0.2)
    logger.info(f"selected features:\n{selected.to_string()}")

    logistic_regression(train[selected.tolist() + ['label']], test[selected.tolist() + ['label']])
