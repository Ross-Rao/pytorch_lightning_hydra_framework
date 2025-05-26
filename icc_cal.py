import os
import pandas as pd
import pingouin as pg
from scipy.stats import chi2_contingency

# 指定包含五个文件夹的主文件夹路径
main_folder = './exp_test'  # 替换为你的主文件夹路径

# 初始化一个空的 DataFrame 用于存储所有数据
all_data = pd.DataFrame()

# 遍历主文件夹中的每个子文件夹
for folder_name in os.listdir(main_folder):
    folder_path = os.path.join(main_folder, folder_name)
    if os.path.isdir(folder_path):
        # 构造 cluster_with_risk.csv 文件的路径
        csv_path = os.path.join(folder_path, 'lightning_logs/version_0/cluster/cluster_with_risk.csv')

        # 检查文件是否存在
        if os.path.exists(csv_path):
            # 读取 CSV 文件
            df = pd.read_csv(csv_path).reset_index()

            # 将风险等级转换为数值
            df['risk_group_code'] = df['risk_group'].map({'Low Risk': 1, 'Medium Risk': 2, 'High Risk': 3})

            # 添加文件夹名称作为标识
            df['folder'] = folder_name

            # 将当前文件夹的数据追加到总数据中
            all_data = pd.concat([all_data, df])

            # # 构建列联表
            # contingency_table = pd.crosstab(all_data['label'], all_data['risk_group'])
            #
            # # 进行卡方检验
            # chi2, p, dof, expected = chi2_contingency(contingency_table)
            #
            # # 输出结果
            # print("Chi-Square Test Result:")
            # print(f"Chi2 Statistic: {chi2}")
            # print(f"P-value: {p}")
            # print(f"Degrees of Freedom: {dof}")
            # print("Expected Frequencies:")
            # print(expected)

        else:
            print(f"File not found: {csv_path}")

# 确保数据中有 cluster 和 risk_group_code 列
if 'cluster' in all_data.columns and 'risk_group_code' in all_data.columns:
    # 计算 ICC 指标
    risk_result = pg.intraclass_corr(data=all_data, targets='index', raters='folder', ratings='risk_group_code')
    pred_result = pg.intraclass_corr(data=all_data, targets='index', raters='folder', ratings='pred')

    # 输出结果
    print("ICC Calculation Result:")
    print(risk_result.to_string())
    print(pred_result.to_string())
else:
    print("Required columns not found in the data.")


