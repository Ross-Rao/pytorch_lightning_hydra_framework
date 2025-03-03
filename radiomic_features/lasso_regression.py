# python import
import logging
# package import
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# local import


logger = logging.getLogger(__name__)
__all__ = ['train_logistic_regression', 'validate_logistic_regression']


def train_logistic_regression(train_df, output, alpha=0.01, seed=42):
    # 1. 数据准备
    x_train = train_df.drop(columns=['label'])
    y_train = train_df['label']

    # 2. Z 分数归一化（标准化）
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    # 3. 使用 Lasso 回归筛选特征
    lasso = Lasso(alpha=alpha, random_state=seed)
    lasso.fit(x_train_scaled, y_train)
    selected_features = x_train.columns[abs(lasso.coef_) > 0]

    # 4. 筛选训练集中的特征
    x_train_selected = x_train[selected_features]

    # 5. 计算新特征（Lasso 系数加权的特征组合）
    new_feature_train = x_train_selected.dot(lasso.coef_[abs(lasso.coef_) > 0])

    # 6. 使用 Logistic Regression 进行分类
    log_reg = LogisticRegression(random_state=seed)
    log_reg.fit(new_feature_train.values.reshape(-1, 1), y_train)

    # 7. write results
    with open(output, 'w') as f:
        f.write("Lasso selected features:\n" + selected_features.to_series().to_string() + '\n')

    # 返回训练好的模型参数
    return {
        'scaler': scaler,
        'lasso': lasso,
        'selected_features': selected_features,
        'log_reg': log_reg
    }


def validate_logistic_regression(test_df, output, model_info):
    # 1. 数据准备
    x_test = test_df.drop(columns=['label'])
    y_test = test_df['label']

    # 2. 使用训练阶段的标准化器
    scaler = model_info['scaler']
    x_test_scaled = scaler.transform(x_test)

    # 3. 使用 Lasso 筛选的特征
    selected_features = model_info['selected_features']
    x_test_selected = x_test[selected_features]

    # 4. 计算新特征
    lasso = model_info['lasso']
    new_feature_test = x_test_selected.dot(lasso.coef_[abs(lasso.coef_) > 0])

    # 5. 使用训练好的 Logistic Regression 模型进行预测
    log_reg = model_info['log_reg']
    y_pred = log_reg.predict(new_feature_test.values.reshape(-1, 1))
    y_pred_proba = log_reg.predict_proba(new_feature_test.values.reshape(-1, 1))[:, 1]

    # 6. 计算性能指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 7. 显著性检验（使用 Statsmodels Logit 模型）
    x_new_test = sm.add_constant(new_feature_test)  # 添加截距项
    logit_model = sm.Logit(y_test, x_new_test).fit(disp=0)
    significance_summary = logit_model.summary().as_text()  # 将显著性检验结果保存为字符串

    # 8. 计算新特征与目标变量的 Pearson 相关性
    correlation, p_value = pearsonr(new_feature_test, y_test)

    # 9. 计算混淆矩阵
    confusion_matrix_result = confusion_matrix(y_test, y_pred)

    with open(output, 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write("\n")
        f.write("Significance test:\n")
        f.write(significance_summary)
        f.write("\n")
        f.write(f"Pearson correlation: {correlation:.4f}, p-value: {p_value:.4f}\n")
        f.write("\n")
        f.write("Confusion matrix:\n")
        f.write(pd.DataFrame(confusion_matrix_result).to_string())
        f.write("\n")
