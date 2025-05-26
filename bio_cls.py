# python import
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
from module.contrastive_cluster_no_unet import ExampleModule
from module.monai_data_module import MonaiDataModule
from utils import callbacks
from utils.util import get_multi_attr
from utils.log import log_exception

# 获取 logger
logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.2",
    config_path=os.getenv('CONFIGS_LOCATION', 'config'),
    config_name="contrastive_cluster_no_unet",
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
    # dataset_config = None
    if cfg.get('mnist', False):  # +mnist=True in command line
        datamodule = MNISTDataModule()
    else:
        dataset_config = cfg.get("dataset")
        metadata_cfg, split_cfg = dataset_config.get("metadata"), dataset_config.get("split")
        load_cfg, loader_cfg = dataset_config.get("load"), dataset_config.get("loader")
        datamodule = MonaiDataModule(metadata_cfg, split_cfg, load_cfg, loader_cfg,
                                     num_classes=cfg.get("model").get("model_params")[1].get("num_classes"))
    length_train = len(datamodule.train_dataset)
    logger.info("data module built.")

    # build model
    model_config, criterion_config = cfg.get("model"), cfg.get("criterion")
    optimizer_config, lr_scheduler_config = cfg.get("optimizer"), cfg.get("lr_scheduler", {})
    model_config['model_params'][0]['n_samples'] = length_train * model_config['model_params'][1]['seq_len']
    model = ExampleModule(**model_config, **criterion_config, **optimizer_config, **lr_scheduler_config)
    model.stage_change_epoch = cfg.get('stage_change_epoch')
    model.max_search_ratio = cfg.get('max_search_ratio')
    model.anchor_update_frequency = cfg.get('anchor_update_frequency')
    logger.info("model built.")

    def visualize_anchor(save_dir):
        import matplotlib.pyplot as plt

        save_path = os.path.join(save_dir, 'anchor')
        os.makedirs(save_path, exist_ok=True)

        m = model.model[0].to('cuda:0')
        flag_tensor, neighbors_tensor = m.mc.flag.to('cuda:0'), m.mc.neighbors.to('cuda:0')
        cluster_tensor = m.get_all_cluster()

        label = torch.cat([dataset[i]['label'].unsqueeze(0)
                           for dataset in [datamodule.train_dataset]
                           for i in range(len(dataset))],
                          dim=0).to('cuda:0')
        label = label.repeat_interleave(cluster_tensor.size(0) // label.size(0), dim=0)
        label = torch.argmax(label, dim=1)

        cluster_df = pd.DataFrame({
            'cluster': cluster_tensor.cpu().numpy(),
            'label': label.cpu().numpy(),
        })
        for i in range(neighbors_tensor.size(1)):
            cluster_df[f'neighbor_{i}'] = neighbors_tensor[:, i].cpu().numpy()

        # 定义三组 cluster
        high_risk_clusters = {0, 2}
        medium_risk_clusters = {1, 4}
        low_risk_clusters = {3, 5, 6, 7}

        # 初始化统计字典
        stats = {
            'high_risk_positive': {'same_category': 0, 'total': 0},
            'high_risk_negative': {'same_category': 0, 'total': 0},
            'medium_risk_positive': {'same_category': 0, 'total': 0},
            'medium_risk_negative': {'same_category': 0, 'total': 0},
            'low_risk_positive': {'same_category': 0, 'total': 0},
            'low_risk_negative': {'same_category': 0, 'total': 0},
        }

        # 遍历每个样本
        for idx, row in cluster_df.iterrows():
            current_cluster = row['cluster']
            current_label = row['label']

            # 确定当前样本的类别
            if current_cluster in high_risk_clusters:
                category = 'high_risk'
            elif current_cluster in medium_risk_clusters:
                category = 'medium_risk'
            elif current_cluster in low_risk_clusters:
                category = 'low_risk'
            else:
                continue  # 跳过未知类别

            # 确定当前样本的 MVI 状态
            mvi_status = 'positive' if current_label == 1 else 'negative'

            # 检查每个邻居
            for neighbor_col in [f'neighbor_{i}' for i in range(neighbors_tensor.size(1))]:
                neighbor_idx = row[neighbor_col]
                if neighbor_idx >= 0:  # 有效的邻居索引
                    neighbor_cluster = cluster_df.loc[neighbor_idx, 'cluster']
                    neighbor_label = cluster_df.loc[neighbor_idx, 'label']

                    # 确定邻居的类别
                    if neighbor_cluster in high_risk_clusters:
                        neighbor_category = 'high_risk'
                    elif neighbor_cluster in medium_risk_clusters:
                        neighbor_category = 'medium_risk'
                    elif neighbor_cluster in low_risk_clusters:
                        neighbor_category = 'low_risk'
                    else:
                        continue  # 跳过未知类别

                    # 确定邻居的 MVI 状态
                    neighbor_mvi_status = 'positive' if neighbor_label == 1 else 'negative'

                    # 检查是否属于同一个类别
                    if (category == neighbor_category) or (mvi_status == neighbor_mvi_status):
                        stats[f'{category}_{mvi_status}']['same_category'] += 1
                    stats[f'{category}_{mvi_status}']['total'] += 1

        fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 2 行 3 列

        categories = [
            'high_risk_positive', 'medium_risk_positive', 'low_risk_positive',
            'high_risk_negative', 'medium_risk_negative', 'low_risk_negative'
        ]

        for idx, category in enumerate(categories):
            row = idx // 3
            col = idx % 3
            ax = axs[row, col]

            same_category = stats[category]['same_category']
            total = stats[category]['total']
            different_category = total - same_category

            labels = ['Same Category', 'Different Category']
            sizes = [same_category, different_category]
            colors = ['#ff9999', '#66b3ff']

            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # 等轴比例
            ax.set_title(category.replace('_', ' ').title())

        plt.tight_layout()

        plot_save_path = os.path.join(save_path, 'neighbor_category_pie_charts.png')
        plt.savefig(plot_save_path)
        plt.close()

        result_df = pd.DataFrame.from_dict(stats, orient='index')
        result_df.to_csv(os.path.join(save_path, 'neighbor_category_ratio.csv'), index_label='category')

    def visualize_cluster(save_dir):
        import matplotlib.pyplot as plt

        m = model.model[0].to('cuda:0')
        image = torch.cat([dataset[i]['image'].unsqueeze(0)
                           for dataset in [datamodule.original_train_dataset,
                                           datamodule.val_dataset, datamodule.test_dataset]
                           for i in range(len(dataset))],
                          dim=0).to('cuda:0')
        b, c, h, w = image.size()
        image = image.reshape(b * c, 1, h, w)
        label = torch.cat([torch.tensor([dataset[i]['label']], dtype=torch.long).unsqueeze(0)
                           for dataset in [datamodule.original_train_dataset,
                                           datamodule.val_dataset, datamodule.test_dataset]
                           for i in range(len(dataset))],
                          dim=0).to('cuda:0')
        label = label.repeat_interleave(c, dim=0)
        if cfg.get("model").get("model_params")[1].get("num_classes") == 2:
            label = torch.where(label >= 1, 1, 0)

        save_path = os.path.join(save_dir, 'cluster')
        os.makedirs(save_path, exist_ok=True)
        cluster = m.visualize_cluster(image, label, save_path=save_path)

        cluster_df = pd.DataFrame({
            'cluster': cluster,
            'label': label.reshape(-1).cpu().numpy(),  # 将 label 转换为 NumPy 数组并保存
        })

        cluster_df.to_csv(os.path.join(save_path, 'cluster.csv'), index=True)

        cluster_label_counts = cluster_df.groupby(['cluster', 'label']).size().unstack(fill_value=0)
        num_clusters = len(cluster_label_counts.index)
        num_columns = 4
        num_rows = (num_clusters + num_columns - 1) // num_columns  # 计算需要的行数
        fig, axs = plt.subplots(num_rows, num_columns, figsize=(5 * num_columns, 5 * num_rows))  # 每个饼图的宽度为 5
        if num_clusters == 1:
            axs = [axs]
        axs = axs.flatten()
        for idx, cluster_id in enumerate(cluster_label_counts.index):
            label_counts = cluster_label_counts.loc[cluster_id]
            labels = label_counts.index
            sizes = label_counts.values

            axs[idx].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            axs[idx].axis('equal')  # 等轴比例
            axs[idx].set_title(f'Cluster {cluster_id}')

        for idx in range(num_clusters, num_rows * num_columns):
            fig.delaxes(axs[idx])

        plt.tight_layout()
        plot_save_path = os.path.join(save_path, 'all_clusters_label_distribution.png')
        plt.savefig(plot_save_path)
        plt.close()

        # 统计每个 cluster 中 label=1 的数量
        label_1_counts = cluster_df[cluster_df['label'] == 1].groupby('cluster').size().reset_index(name='count')
        label_1_counts = label_1_counts.sort_values(by='count', ascending=False)
        all_clusters = cluster_df['cluster'].unique()

        # 定义三组 cluster
        high_risk_clusters = []
        medium_risk_clusters = []
        low_risk_clusters = []

        for idx, row in label_1_counts.iterrows():
            cluster_id = row['cluster']
            if idx < 2:  # 前两个数量最多的 cluster 为高风险
                high_risk_clusters.append(cluster_id)
            elif idx < 4:  # 接下来的两个数量次多的 cluster 为中风险
                medium_risk_clusters.append(cluster_id)
            else:  # 剩下的为低风险
                low_risk_clusters.append(cluster_id)

        for cluster_id in all_clusters:
            if cluster_id not in high_risk_clusters and cluster_id not in medium_risk_clusters:
                low_risk_clusters.append(cluster_id)

        # 创建画布和子图
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 三列，每列一个饼图

        # 高风险组
        high_risk_df = cluster_df[cluster_df['cluster'].isin(high_risk_clusters)]
        high_risk_label_counts = high_risk_df['label'].value_counts()
        labels = high_risk_label_counts.index
        sizes = high_risk_label_counts.values
        axs[0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['orange', 'blue'])
        axs[0].axis('equal')
        axs[0].set_title(f'High Risk (Clusters {", ".join(map(str, high_risk_clusters))})')

        # 中等风险组
        medium_risk_df = cluster_df[cluster_df['cluster'].isin(medium_risk_clusters)]
        medium_risk_label_counts = medium_risk_df['label'].value_counts()
        labels = medium_risk_label_counts.index
        sizes = medium_risk_label_counts.values
        axs[1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['orange', 'blue'])
        axs[1].axis('equal')
        axs[1].set_title(f'Medium Risk (Clusters {", ".join(map(str, medium_risk_clusters))})')

        # 低风险组
        low_risk_df = cluster_df[cluster_df['cluster'].isin(low_risk_clusters)]
        low_risk_label_counts = low_risk_df['label'].value_counts()
        labels = low_risk_label_counts.index
        sizes = low_risk_label_counts.values
        axs[2].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['orange', 'blue'])
        axs[2].axis('equal')
        axs[2].set_title(f'Low Risk (Clusters {", ".join(map(str, low_risk_clusters))})')

        # 调整布局
        plt.tight_layout()

        # 保存整个画布
        plot_save_path = os.path.join(save_path, 'risk_groups_label_distribution.png')
        plt.savefig(plot_save_path)
        plt.close()

    def grad_cam(grad_model, target_layer, input_tensor, target_class, save_path):
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        cam = GradCAM(model=grad_model, target_layers=[target_layer])
        targets = [ClassifierOutputTarget(target_class[i].item()) for i in range(input_tensor.shape[0])]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

        with torch.no_grad():
            pred = grad_model(input_tensor)
        pred_class = torch.argmax(pred, dim=1)

        # 检查预测结果与目标类别是否一致
        correct_indices = (pred_class == target_class).nonzero(as_tuple=True)[0]
        incorrect_indices = (pred_class != target_class).nonzero(as_tuple=True)[0]

        def _create_grid(visualizations):
            num_images = len(visualizations)
            grid_size = int(np.ceil(np.sqrt(num_images)))
            num_fill = grid_size * grid_size - num_images
            fill_image = np.zeros_like(visualizations[0])
            visualizations += [fill_image] * num_fill
            grid_visualization = np.vstack([np.hstack(visualizations[i * grid_size:(i + 1) * grid_size])
                                            for i in range(grid_size)])
            return grid_visualization

        # 保存原始图像的网格
        if len(correct_indices) > 0:
            correct_original_images = [input_tensor[i, 0, :, :] for i in correct_indices]  # 选择单通道
            grid_visualization_correct_original = _create_grid(correct_original_images)
            plt.imsave(os.path.join(save_path, f'grid_original_correct_{target_class[0].item()}.png'),
                       grid_visualization_correct_original, cmap='gray')  # 使用灰度图

        if len(incorrect_indices) > 0:
            incorrect_original_images = [input_tensor[i, 0, :, :] for i in incorrect_indices]  # 选择单通道
            grid_visualization_incorrect_original = _create_grid(incorrect_original_images)
            plt.imsave(os.path.join(save_path, f'grid_original_incorrect_{target_class[0].item()}.png'),
                       grid_visualization_incorrect_original, cmap='gray')  # 使用灰度图

        # 准备输入张量以便可视化
        input_tensor = input_tensor.permute(0, 2, 3, 1).cpu().detach().numpy()  # 调整通道顺序为 HxWxC

        # 保存 Grad-CAM 热力图的网格
        if len(correct_indices) > 0:
            correct_visualizations = [show_cam_on_image(input_tensor[i], grayscale_cam[i], use_rgb=True)
                                      for i in correct_indices]
            grid_visualization_correct = _create_grid(correct_visualizations)
            plt.imsave(os.path.join(save_path, f'grid_visualization_correct_{target_class[0].item()}.png'),
                       grid_visualization_correct)

        if len(incorrect_indices) > 0:
            incorrect_visualizations = [show_cam_on_image(input_tensor[i], grayscale_cam[i], use_rgb=True)
                                        for i in incorrect_indices]
            grid_visualization_incorrect = _create_grid(incorrect_visualizations)
            plt.imsave(os.path.join(save_path, f'grid_visualization_incorrect_{target_class[0].item()}.png'),
                       grid_visualization_incorrect)

    def visualize_grad(save_dir):
        import torch.nn as nn
        image = torch.cat([dataset[i]['image'].unsqueeze(0)
                           for dataset in [datamodule.test_dataset]
                           for i in range(len(dataset))],
                          dim=0).to('cuda:0')
        b, c, h, w = image.size()
        image = image.reshape(b * c, 1, h, w)
        label = torch.cat([torch.tensor([dataset[i]['label']], dtype=torch.long).unsqueeze(0)
                           for dataset in [datamodule.test_dataset]
                           for i in range(len(dataset))],
                          dim=0).to('cuda:0')
        label = label.repeat_interleave(c, dim=0).reshape(-1)
        if cfg.get("model").get("model_params")[1].get("num_classes") == 2:
            label = torch.where(label >= 1, 1, 0)

        class M(nn.Module):
            def __init__(self, model0, model1):
                super().__init__()
                self.model0 = model0
                self.model1 = model1

            def forward(self, img):
                cluster, hid_x = self.model0(img, None, None, loss=False)
                cls = self.model1(hid_x, cluster)
                return cls

        m0 = model.model[0].to('cuda:0')
        m0.train()
        m1 = model.model[1].to('cuda:0')
        m1.train()
        for param in m0.parameters():
            param.requires_grad = True
        image.requires_grad = True
        m = M(m0, m1).to('cuda:0')
        m.train()
        save_path = os.path.join(save_dir, 'grad_cam')
        os.makedirs(save_path, exist_ok=True)
        for class_label in torch.unique(label).tolist():
            class_indices = torch.where(label == class_label)[0]
            class_images = image[class_indices]
            class_labels = label[class_indices]
            grad_cam(m, m.model0.encoder.resnet.layer4.conv1,
                     class_images, class_labels, save_path=save_path)
        m0.eval()
        m1.eval()

    def visualize_centroids(test_pl_trainer):
        import torch.nn as nn
        import matplotlib.pyplot as plt
        import numpy as np

        class CentroidsReconstruction(pl.LightningModule):
            def __init__(self, model0, logit_tensor, save_dir, lr=0.0005):
                super().__init__()
                self.encoder = model0.encoder
                self.decoder = model0.decoder
                self.pool = model0.pool
                self.act = model0.act
                self.fc = model0.fc
                self.logit_tensor = logit_tensor
                self.unet.eval()
                for param in self.unet.parameters():
                    param.requires_grad = False
                for param in self.fc.parameters():
                    param.requires_grad = False
                self.loss_fn = nn.MSELoss()
                self.save_dir = save_dir
                self.lr = lr
                self.x = nn.Parameter(torch.randn(logit_tensor.size(0), 1, 64, 64, requires_grad=True))

            def forward(self):
                hid_x = self.encoder(self.x).reshape(self.x.size(0), -1, 4, 4)
                x_hat = self.decoder(hid_x)
                hid_x = self.pool(hid_x).squeeze()
                logit_pred = self.act(self.fc(hid_x))
                return logit_pred, x_hat

            def training_step(self, batch, batch_idx):
                logit_pred, x_hat = self.forward()
                loss_x = self.loss_fn(x_hat, self.x)
                loss_logit = self.loss_fn(self.logit_tensor, logit_pred)
                total_loss = loss_x + loss_logit
                self.log("train_loss", total_loss, prog_bar=True)
                self.log("val_loss", total_loss, prog_bar=True)

                if self.current_epoch % 1000 == 0:
                    self.visualize_centroids()

                return total_loss

            def train_dataloader(self):
                from torch.utils.data import DataLoader, Dataset

                class DummyDataset(Dataset):
                    def __init__(self, length):
                        self.length = length

                    def __len__(self):
                        return self.length

                    def __getitem__(self, idx):
                        return torch.zeros(1)  # 返回一个空的张量

                return DataLoader(DummyDataset(length=1), batch_size=1)

            def configure_optimizers(self):
                return torch.optim.Adam([self.x], lr=self.lr)

            def visualize_centroids(self):
                save_path = os.path.join(self.save_dir, 'centroids')
                os.makedirs(save_path, exist_ok=True)

                def _create_grid(visualizations):
                    num_images = len(visualizations)
                    grid_size = int(np.ceil(np.sqrt(num_images)))
                    num_fill = grid_size * grid_size - num_images
                    fill_image = np.zeros_like(visualizations[0])
                    visualizations += [fill_image] * num_fill
                    grid_visualization = np.vstack([np.hstack(visualizations[i * grid_size:(i + 1) * grid_size])
                                                    for i in range(grid_size)])
                    return grid_visualization

                centroid_images = [self.x[i, 0, :, :].detach().cpu().numpy() for i in range(self.x.shape[0])]
                grid_centroids_images = _create_grid(centroid_images)
                plt.imsave(os.path.join(save_path, f'centroids_epoch_{self.current_epoch}.png'), grid_centroids_images,
                           cmap='gray')

            def on_train_end(self) -> None:
                # 在训练结束时保存最终的质心图像
                self.visualize_centroids()

        m = CentroidsReconstruction(model.model[0],
                                    model.model[0].mc.kmeans.centroids.clone().detach().requires_grad_(True),
                                    save_dir=trainer.logger.log_dir)
        test_pl_trainer.fit(m)

    def visualize_masks(test_pl_trainer):
        import torch.nn as nn
        import matplotlib.pyplot as plt
        import numpy as np

        import torch.nn as nn

        image = torch.cat([dataset[i]['image'].unsqueeze(0)
                           for dataset in [datamodule.test_dataset]
                           for i in range(len(dataset))],
                          dim=0).to('cuda:0')
        b, c, h, w = image.size()
        image = image.reshape(b * c, 1, h, w)
        label = torch.cat([torch.tensor([dataset[i]['label']], dtype=torch.long).unsqueeze(0)
                           for dataset in [datamodule.test_dataset]
                           for i in range(len(dataset))],
                          dim=0).to('cuda:0')
        label = label.repeat_interleave(c, dim=0)
        if cfg.get("model").get("model_params")[1].get("num_classes") == 2:
            label = torch.where(label >= 1, 1, 0)
        label = label.reshape(-1)

        class M(nn.Module):
            def __init__(self, model0, model1):
                super().__init__()
                self.model0 = model0
                self.model1 = model1

            def forward(self, img):
                cluster, hid_x = self.model0(img, None, None, loss=False)
                cls = self.model1(hid_x, cluster)
                return cls

        m0 = model.model[0].to('cuda:0')
        m1 = model.model[1].to('cuda:0')
        m = M(m0, m1).to('cuda:0')
        pred = torch.argmax(m(image), dim=1)

        cluster = m0.visualize_cluster(image, None)

        cluster_df = pd.DataFrame({
            'cluster': cluster,
            'label': label.reshape(-1).cpu().numpy(),  # 将 label 转换为 NumPy 数组并保存
            'pred': pred.reshape(-1).cpu().detach().numpy(),  # 将 pred 转换为 NumPy 数组并保存
        })

        # 计算每个 cluster 中 label=1 的比例
        label_1_proportion = (
            cluster_df[cluster_df['label'] == 1]
            .groupby('cluster')
            .size()
            .div(cluster_df.groupby('cluster').size())
            .reset_index(name='proportion')
        )

        label_1_proportion = label_1_proportion.sort_values(by='proportion', ascending=False).reset_index(drop=True)

        # 定义三组 cluster
        high_risk_clusters = []
        medium_risk_clusters = []
        low_risk_clusters = []

        for idx, row in label_1_proportion.iterrows():
            cluster_id = int(row['cluster'])
            if idx < 2:  # 前两个数量最多的 cluster 为高风险
                high_risk_clusters.append(cluster_id)
            elif idx < 4:  # 接下来的两个数量次多的 cluster 为中风险
                medium_risk_clusters.append(cluster_id)
            else:  # 剩下的为低风险
                low_risk_clusters.append(cluster_id)

        class MaskReconstruction(pl.LightningModule):
            def __init__(self, model0, logit_tensor, save_dir, lr=0.0005):
                super().__init__()
                self.m0 = model0
                self.decoder = model0.decoder
                self.encoder = model0.encoder
                self.pool = model0.pool
                self.act = model0.act
                self.fc = model0.fc
                self.logit_tensor = logit_tensor
                self.m0.eval()
                for param in self.m0.parameters():
                    param.requires_grad = False
                self.loss_fn = nn.MSELoss()
                
                from models.loss import PerceptualLoss
                self.perceptual_loss = PerceptualLoss()
                self.save_dir = save_dir
                self.lr = lr
                self.img = torch.cat([dataset[i]['image'].unsqueeze(0)
                                      for dataset in [datamodule.test_dataset]
                                      for i in range(len(dataset))],
                                     dim=0).reshape(-1, 1, 64, 64)
                self.img = self.img[torch.isin(torch.from_numpy(cluster), torch.tensor(high_risk_clusters))]
                self.x = nn.Parameter(torch.randn(self.img.size(0), 1, 64, 64, requires_grad=True))
                self.threshold = 0.7

            def forward(self, use_mask=True):
                self.img = self.img.to(self.device)
                mask = torch.sigmoid(self.x)  # 使用 Sigmoid 函数将 mask 转换为 [0, 1] 范围
                if use_mask:
                    x = self.img * mask
                else:
                    x = self.img
                hid_x = self.encoder(x).reshape(x.size(0), -1, 4, 4)
                x_hat = self.decoder(hid_x)
                hid_x = self.pool(hid_x).squeeze()
                logit_pred = self.act(self.fc(hid_x))
                return logit_pred, x_hat

            def training_step(self, batch, batch_idx):
                logit_pred, x_hat = self.forward()
                mask = torch.sigmoid(self.x)  # 使用 Sigmoid 函数将 mask 转换为 [0, 1] 范围
                mask = (mask > self.threshold).float()
                loss_x = self.loss_fn(x_hat, self.img * mask)
                indices = self.m0.mc.get_cluster(logit_pred).to(self.device).unsqueeze(1)
                cluster = torch.gather(self.logit_tensor, 0, indices.expand(-1, self.logit_tensor.shape[1]))
                loss_logit = self.loss_fn(cluster, logit_pred)
                total_loss = loss_x + loss_logit + torch.clamp(mask.mean() - (1 - self.threshold), min=0)
                self.log("train_loss", total_loss, prog_bar=True)
                self.log("val_loss", total_loss, prog_bar=True)

                if self.current_epoch % 1000 == 0:
                    self.visualize_masks()

                return total_loss

            def on_train_start(self) -> None:
                save_path = os.path.join(self.save_dir, 'masks')
                os.makedirs(save_path, exist_ok=True)
                original_images = self.img.detach().cpu().numpy()
                grid_original_images = self._create_grid([original_images[i, 0, :, :]
                                                          for i in range(original_images.shape[0])])
                plt.imsave(os.path.join(save_path, f'original_images.png'),
                           grid_original_images, cmap='gray')

                _, recon_images = self.forward()
                grid_recon_images = self._create_grid([recon_images[i, 0, :, :]
                                                       for i in range(recon_images.shape[0])])
                plt.imsave(os.path.join(save_path, f'recon_images.png'),
                           grid_recon_images, cmap='gray')

            @staticmethod
            def _create_grid(visualizations):
                num_images = len(visualizations)
                grid_size = int(np.ceil(np.sqrt(num_images)))
                num_fill = grid_size * grid_size - num_images
                fill_image = np.zeros_like(visualizations[0])
                visualizations += [fill_image] * num_fill
                grid_visualization = np.vstack([np.hstack(visualizations[i * grid_size:(i + 1) * grid_size])
                                                for i in range(grid_size)])
                return grid_visualization

            def visualize_masks(self):
                import matplotlib.pyplot as plt
                import os

                save_path = os.path.join(self.save_dir, 'masks')
                os.makedirs(save_path, exist_ok=True)

                masks = torch.sigmoid(self.x)
                input_images = self.img * masks
                # masks = torch.where(masks > self.threshold, masks, torch.zeros_like(masks))
                masks = masks.detach().cpu().numpy()
                cmap = plt.get_cmap('viridis')  # 使用 'viridis' 颜色映射
                colored_masks = [cmap(mask[0, :, :]) for mask in masks]  # 将每个掩码映射为彩色图像
                grid_colored_masks = self._create_grid(colored_masks)
                plt.imsave(os.path.join(save_path, f'masks_epoch_{self.current_epoch}.png'), grid_colored_masks)

                input_images = input_images.detach().cpu().numpy()
                grid_input_images = self._create_grid([input_images[i, 0, :, :]
                                                       for i in range(input_images.shape[0])])
                plt.imsave(os.path.join(save_path, f'input_epoch_{self.current_epoch}.png'),
                           grid_input_images, cmap='gray')

                _, recon_images = self.forward()
                recon_images = recon_images.detach().cpu().numpy()
                grid_recon_images = self._create_grid([recon_images[i, 0, :, :]
                                                       for i in range(recon_images.shape[0])])
                plt.imsave(os.path.join(save_path, f'recon_epoch_{self.current_epoch}.png'),
                           grid_recon_images, cmap='gray')

            def train_dataloader(self):
                from torch.utils.data import DataLoader, Dataset

                class DummyDataset(Dataset):
                    def __init__(self, length):
                        self.length = length

                    def __len__(self):
                        return self.length

                    def __getitem__(self, idx):
                        return torch.zeros(1)  # 返回一个空的张量

                return DataLoader(DummyDataset(length=1), batch_size=1)

            def configure_optimizers(self):
                return torch.optim.Adam([self.x], lr=self.lr)

        m = MaskReconstruction(model.model[0],
                               model.model[0].mc.kmeans.centroids.clone().detach().requires_grad_(True),
                               save_dir=trainer.logger.log_dir)
        test_pl_trainer.fit(m)

    def statistic_test(save_dir):
        import torch.nn as nn
        import torch.nn.functional as f
        from scipy.stats import ttest_rel
        image = torch.cat([dataset[i]['image'].unsqueeze(0)
                           for dataset in [datamodule.test_dataset]
                           for i in range(len(dataset))],
                          dim=0).to('cuda:0')
        b, c, h, w = image.size()
        image = image.reshape(b * c, 1, h, w)

        # label = torch.cat([torch.tensor([dataset[i]['label']], dtype=torch.long).unsqueeze(0)
        #                    for dataset in [datamodule.test_dataset]
        #                    for i in range(len(dataset))],
        #                   dim=0).to('cuda:0')
        # label = label.repeat_interleave(c, dim=0).reshape(-1)
        # if cfg.get("model").get("model_params")[1].get("num_classes") == 2:
        #     label = torch.where(label >= 1, 1, 0)

        class M(nn.Module):
            def __init__(self, model0, model1):
                super().__init__()
                self.model0 = model0
                self.model1 = model1

            def forward(self, img):
                cluster, hid_x = self.model0(img, None, None, loss=False)
                cls = self.model1(hid_x, cluster)
                cls = f.softmax(cls, dim=1)
                return cls

        m0 = model.model[0].to('cuda:0')
        m1 = model.model[1].to('cuda:0')
        m = M(m0, m1).to('cuda:0')
        pred = f.softmax(m(image), dim=1)

        prob_class_0 = pred[:, 0].cpu().detach().numpy()
        prob_class_1 = pred[:, 1].cpu().detach().numpy()
        t_statistic, p_value = ttest_rel(prob_class_0, prob_class_1)
        logger.info(f"t-statistic: {t_statistic}, p-value: {p_value}")

        save_path = os.path.join(save_dir, 'statistic')
        os.makedirs(save_path, exist_ok=True)
        pd.DataFrame({
            't-statistic': [t_statistic],
            'p-value': [p_value],
        }).to_csv(os.path.join(save_path, 't-test.csv'), index=False)

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
            for version_dir in version_dirs:
                ckpt_dir = os.path.join(version_dir, 'checkpoints')
                if os.path.exists(ckpt_dir):
                    # Find all .ckpt files in the directory
                    ckpt_files = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))
                    for ckpt_file in ckpt_files:
                        # Perform testing
                        version_number = os.path.basename(version_dir).split('_')[-1]
                        # Create a new TensorBoardLogger with a specific version
                        tb_logger = TensorBoardLogger(save_dir=work_dir, version=f'version_{version_number}')
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
            if cfg.get('visual', False):
                # visualize_grad(trainer.logger.log_dir)
                # visualize_anchor(trainer.logger.log_dir)
                # visualize_cluster(trainer.logger.log_dir)
                visualize_masks(trainer)
                # visualize_centroids(trainer)
    else:
        trainer.fit(model, datamodule, ckpt_path=cfg.get("ckpt_path", None))
        logger.info("training finished.")

        ckpt_path = os.path.join(trainer.logger.log_dir, 'checkpoints')
        version_number = os.path.basename(trainer.logger.log_dir).split('_')[-1]
        for v, ckpt_file in enumerate(os.listdir(ckpt_path)):
            # Create a new TensorBoardLogger with a specific version
            tb_logger = TensorBoardLogger(save_dir=work_dir, version=f'version_{version_number}_test_{v}')
            test_trainer_cfg = trainer_config.copy()
            test_trainer_cfg['max_epochs'] = 30000
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

            if cfg.get('visual', False):
                visualize_grad(test_trainer.logger.log_dir)
                visualize_anchor(test_trainer.logger.log_dir)
                visualize_cluster(test_trainer.logger.log_dir)
                visualize_centroids(test_trainer)
                logger.info(f"visualization finished for {base_name}.")
    # trainer.reset_train_dataloader()  # hydra will not reset dataloader automatically
    # trainer.reset_val_dataloader()  # if you use multirun, you need to reset dataloader manually


if __name__ == "__main__":
    main()
