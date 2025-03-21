# python import
import logging
from typing import Any
# package import
import torch
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import (Accuracy, Precision, Recall, F1Score, AUROC, MeanAbsoluteError, MeanSquaredError,
                          ConfusionMatrix, MetricCollection)
from torchmetrics.image import (PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure,
                                LearnedPerceptualImagePatchSimilarity)
from torchmetrics.utilities.plot import plot_confusion_matrix
# local import
import models
from utils.util import get_multi_attr, patches2images

logger = logging.getLogger(__name__)
__all__ = ['ExampleModule']


class ExampleModule(pl.LightningModule):
    def __init__(self,
                 model: str,
                 model_params: dict,
                 optimizer: str,
                 optimizer_params: dict,
                 criterion: str,
                 criterion_params: dict = None,
                 lr_scheduler: str = None,
                 lr_scheduler_params: dict = None,
                 lr_scheduler_other_params: dict = None):
        super().__init__()
        # model structure settings
        self.model = getattr(models, model)(**model_params)

        # optimizer settings
        self.optimizer = getattr(torch.optim, optimizer)(self.parameters(), **optimizer_params)

        # loss function settings
        self.criterion = get_multi_attr([torch.nn, models], {criterion: criterion_params})[0]

        # lr_scheduler settings
        lr_lt = [lr_scheduler, lr_scheduler_params, lr_scheduler_other_params]
        assert all(var is None for var in lr_lt) or all(var is not None for var in lr_lt), \
            'if lr_scheduler is valid, lr_scheduler_params and lr_scheduler_other_params must be provided'
        if lr_scheduler is None:
            self.lr_scheduler = None
        else:
            lr_scheduler_func = getattr(torch.optim.lr_scheduler, lr_scheduler)  # StepLR, ReduceLROnPlateau, etc.
            self.lr_scheduler = {
                'scheduler': lr_scheduler_func(self.optimizer, **lr_scheduler_params),
                **lr_scheduler_other_params,  # monitor, interval, frequency, etc.
            }

        # metrics
        n_cls = model_params.get('num_classes', model_params.get('out_features', model_params.get('out_channels', 10)))
        multi_cls_param = dict(average='macro', task='multiclass', num_classes=n_cls)
        self.confusion_matrix = ConfusionMatrix(num_classes=n_cls, task='multiclass')
        self._train_cls_metrics = MetricCollection({
            "accuracy": Accuracy(**multi_cls_param),
            "precision": Precision(**multi_cls_param),
            "recall": Recall(**multi_cls_param),
            "f1": F1Score(**multi_cls_param),
            "auc": AUROC(**multi_cls_param),
        }, prefix="train/")
        self._train_recon_metrics = MetricCollection({
            "psnr": PeakSignalNoiseRatio(),
            "ssim": StructuralSimilarityIndexMeasure(),
            "lpips": LearnedPerceptualImagePatchSimilarity(),
            "recon_mae": MeanAbsoluteError(),
            "recon_mse": MeanSquaredError(),
        }, prefix="train/")
        self._train_reg_metrics = MetricCollection({
            "mae": MeanAbsoluteError(),
            "mse": MeanSquaredError(),
        }, prefix="train/")
        self._val_cls_metrics = self._train_cls_metrics.clone(prefix="val/")
        self._val_recon_metrics = self._train_recon_metrics.clone(prefix="val/")
        self._val_reg_metrics = self._train_reg_metrics.clone(prefix="val/")
        self._test_cls_metrics = self._train_cls_metrics.clone(prefix="test/")
        self._test_recon_metrics = self._train_recon_metrics.clone(prefix="test/")
        self._test_reg_metrics = self._train_reg_metrics.clone(prefix="test/")
        self.cls_metrics = {
            'train': self._train_cls_metrics,
            'val': self._val_cls_metrics,
            'test': self._test_cls_metrics,
        }
        self.recon_metrics = {
            'train': self._train_recon_metrics,
            'val': self._val_recon_metrics,
            'test': self._test_recon_metrics,
        }
        self.reg_metrics = {
            'train': self._train_reg_metrics,
            'val': self._val_reg_metrics,
            'test': self._test_reg_metrics,
        }

    def configure_optimizers(self):
        """
        set optimizer and lr_scheduler(optional)
        """
        if self.lr_scheduler is not None:
            return [self.optimizer], [self.scheduler]
        else:
            return self.optimizer

    @staticmethod
    def get_batch(batch):
        if isinstance(batch, list):
            return batch
        elif isinstance(batch, dict):
            # some attributes from (pre)transform
            pass
        else:
            raise ValueError('Invalid batch type')

    @staticmethod
    def get_batch_size(batch):
        if isinstance(batch, list):
            return batch[0].size(0)
        elif isinstance(batch, dict):
            return batch['image'].size(0)
        else:
            raise ValueError('Invalid batch type')

    def training_step(self, batch, batch_idx):
        x, y = self.get_batch(batch)
        model_params = x if isinstance(x, tuple) else (x,)
        y_hat = self.model(*model_params)
        # self._update_metrics(y_hat, y, "train")  # not necessary, only debug

        # be sure that y_hat params first and y params later in your criterion function
        criterion_params = (y_hat if isinstance(y_hat, tuple) else (y_hat,)) + (y if isinstance(y, tuple) else (y,))
        loss = self.criterion(*criterion_params)
        return loss

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        # log step loss
        # train/loss is the key for callback
        loss_dt = {f'train/{k}': float(v) for k, v in outputs.items()}
        self.log_dict(loss_dt, batch_size=self.get_batch_size(batch), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = self.get_batch(batch)
        model_params = x if isinstance(x, tuple) else (x,)
        y_hat = self.model(*model_params)
        self._update_metrics(y_hat, y, "val")

        # be sure that y_hat params first and y params later in your criterion function
        criterion_params = (y_hat if isinstance(y_hat, tuple) else (y_hat,)) + (y if isinstance(y, tuple) else (y,))
        loss = self.criterion(*criterion_params)
        return loss

    def on_validation_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        # log step loss
        # val/loss is the key for callback
        outputs = {'loss': outputs} if isinstance(outputs, torch.Tensor) else outputs
        loss_dt = {f'val/{k}': float(v) for k, v in outputs.items()}
        self.log_dict(loss_dt, batch_size=self.get_batch_size(batch), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = self.get_batch(batch)
        model_params = x if isinstance(x, tuple) else (x,)
        y_hat = self.model(*model_params)
        # `patch_coords` comes from `GridPatchDataset`, used for saving images
        if isinstance(batch, dict):
            extra_params = {'index': batch.get('index', None), 'coords': batch.get('patch_coords', None)}
        else:
            extra_params = {}
        self._update_metrics(y_hat, y, "test", **extra_params)

    def on_train_epoch_end(self):
        # not necessary, only debug
        for metrics_dict in [self.cls_metrics['train'], self.reg_metrics['train'], self.recon_metrics['train']]:
            for metric_name, metric in metrics_dict.items():
                # re-comment `update_metrics` in training_step if you want to use this
                if metric.update_count > 0:
                    res = metric.compute()
                    self.log(metric_name, res, prog_bar=True)
                    metric.reset()

    def on_validation_epoch_end(self):
        for metrics_dict in [self.cls_metrics['val'], self.reg_metrics['val'], self.recon_metrics['val']]:
            for metric_name, metric in metrics_dict.items():
                if metric.update_count > 0:
                    res = metric.compute()
                    self.log(metric_name, res, prog_bar=True)
                    logger.info(f"Epoch {self.current_epoch} - {metric_name}: {res}")
                    metric.reset()

    def on_test_epoch_end(self):
        if self.confusion_matrix.update_count > 0:
            plt, _ = plot_confusion_matrix(self.confusion_matrix.compute())
            self.logger.experiment.add_figure(f"test_confusion_matrix", plt)
        for metrics_dict in [self.cls_metrics['test'], self.reg_metrics['test'], self.recon_metrics['test']]:
            for metric_name, metric in metrics_dict.items():
                if metric.update_count > 0:
                    res = metric.compute()
                    self.log(metric_name, res, prog_bar=True)
                    logger.info(f"{metric_name}: {res}")
                    metric.reset()

    def _update_metrics(self, y_hat_tp, y_tp, stage, **kwargs):
        # ensure matched y and y_hat is same
        # zip will stop at the shortest length
        y_hat_tp = y_hat_tp if isinstance(y_hat_tp, tuple) else (y_hat_tp,)
        y_tp = y_tp if isinstance(y_tp, tuple) else (y_tp,)

        for y_hat, y in zip(y_hat_tp, y_tp):
            if len(y_hat.shape) == 2:
                if y_hat.shape[1] == 1:  # Regression task
                    self._regression_metrics(y_hat, y, stage)
                elif y_hat.shape[1] >= 2:  # Multi-class classification task
                    self._multiclass_classification_metrics(y_hat, y, stage)
                    if stage == "test":
                        self.confusion_matrix.update(y_hat, y)
                else:
                    raise ValueError("Invalid shape for y_hat.")
            elif len(y_hat.shape) == 4:  # Image reconstruction task
                self._image_reconstruction_metrics(y_hat, y, stage)
                if stage == "test":
                    self._save_reconstruction_images(y_hat, **kwargs)

    def _save_reconstruction_images(self, images, index, coords=None):
        if coords is not None:
            images, image_indices = patches2images(image_indices=index, patches=images, coords=coords)
            for i, (image, image_index) in enumerate(zip(images, image_indices)):
                self.logger.experiment.add_image(f"test_image_{image_index}", image,
                                                 dataformats='CHW', global_step=self.global_step)
        else:
            for i, (image, image_index) in enumerate(zip(images, index)):
                self.logger.experiment.add_image(f"test_image_{float(image_index)}", torch.round(image * 255).byte(),
                                                 dataformats='CHW', global_step=self.global_step)

    def _regression_metrics(self, y_hat, y, stage):
        self.reg_metrics[stage].update(y_hat.reshape(-1), y)

    def _multiclass_classification_metrics(self, y_hat, y, stage):
        probs = torch.softmax(y_hat, dim=1)
        self.cls_metrics[stage].update(probs, y)

    def _image_reconstruction_metrics(self, y_hat, y, stage):
        # input range is [0, 1], adjust to [-1, 1] for LPIPS
        y_hat_lpips = y_hat.repeat(1, 3, 1, 1) * 2 - 1 if y_hat.size(1) == 1 else y_hat * 2 - 1
        y_lpips = y.repeat(1, 3, 1, 1) * 2 - 1 if y.size(1) == 1 else y * 2 - 1

        # 更新其他指标
        for metric_name, metric in self.recon_metrics[stage].items():
            if metric_name != "lpips":
                metric.update(y_hat, y)
        # 单独更新lpips
        self.recon_metrics[stage]["lpips"].update(y_hat_lpips, y_lpips)
