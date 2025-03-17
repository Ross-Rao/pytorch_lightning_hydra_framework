# python import
import logging
# package import
import torch
import lightning.pytorch as pl
from torch.nn.functional import l1_loss, mse_loss
from torchmetrics.functional.classification import auroc, accuracy, precision, recall, f1_score
from torchmetrics.image import (PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure,
                                LearnedPerceptualImagePatchSimilarity)
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
            # index, neighbor_index: UpdatePatchIndexd
            # image_slice: DropSliced
            index = batch.get('index', None)
            neighbor_index = batch.get('neighbor_index', None)
            index = index.reshape(-1) if isinstance(index, torch.Tensor) else None
            neighbor_index = neighbor_index.long().reshape(-1) if isinstance(neighbor_index, torch.Tensor) else None
            x = (batch['image'], index, neighbor_index)
            # y = (batch['label'].reshape(-1), batch['image_slice'])  # and metadata
            y = (batch['label'].reshape(-1))  # and metadata
            return x, y
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

    def on_train_epoch_start(self, max_search_ratio=0.5, frequency=20):
        # on_train_epoch_start may not suitable for update model parameters, maybe works?
        # we update anchor in training_step with anchor_update_frequency
        current_epoch = self.current_epoch
        update_epoch = [i for i in range(0, self.trainer.max_epochs, frequency)]
        if current_epoch in update_epoch:
            max_epoch = self.trainer.max_epochs
            search_ratio = current_epoch / max_epoch * max_search_ratio
            self.criterion.mc.update_anchor(search_rate=search_ratio)

    def training_step(self, batch, batch_idx):
        x, y = self.get_batch(batch)
        model_params = x if isinstance(x, tuple) else (x, )
        y_hat = self.model(*model_params)
        # self.log_general_validation_metrics(y_hat, y, "train")  # not necessary, only debug

        # be sure that y_hat params first and y params later in your criterion function
        criterion_params = (y_hat if isinstance(y_hat, tuple) else (y_hat, )) + (y if isinstance(y, tuple) else (y,))
        loss = self.criterion(*criterion_params)
        loss, loss_dt = loss if isinstance(loss, tuple) else (loss, {})
        assert isinstance(loss, torch.Tensor), "loss must be a tensor"
        assert isinstance(loss_dt, dict), "loss_dt must be a dict"
        if loss_dt:
            loss_dt = {f'train/{k}': float(v) for k, v in loss_dt.items()}
            logger.info(f"train epoch: {self.current_epoch}, step: {batch_idx},\n"
                        f"loss_dt: {loss_dt}")
            self.log_dict(loss_dt, batch_size=self.get_batch_size(batch), prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)  # val_loss is the key for callback
        logger.info(f"train_loss: {float(loss)}")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.get_batch(batch)
        model_params = x if isinstance(x, tuple) else (x, )
        y_hat = self.model(*model_params)
        self.log_general_validation_metrics(y_hat, y, "val")

        # be sure that y_hat params first and y params later in your criterion function
        criterion_params = (y_hat if isinstance(y_hat, tuple) else (y_hat, )) + (y if isinstance(y, tuple) else (y,))
        loss = self.criterion(*criterion_params)
        loss, loss_dt = loss if isinstance(loss, tuple) else (loss, {})
        assert isinstance(loss, torch.Tensor), "loss must be a tensor"
        assert isinstance(loss_dt, dict), "loss_dt must be a dict"
        if loss_dt:
            loss_dt = {f'val/{k}': float(v) for k, v in loss_dt.items()}
            logger.info(f"val epoch: {self.current_epoch}, step: {batch_idx},\n"
                        f"loss_dt: {loss_dt}")
            self.log_dict(loss_dt, batch_size=self.get_batch_size(batch), prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)  # val_loss is the key for callback
        logger.info(f"val_loss: {float(loss)}")
        return loss

    def test_step(self, batch, batch_idx):
        x, y = self.get_batch(batch)
        model_params = x if isinstance(x, tuple) else (x, )
        y_hat = self.model(*model_params)
        self.log_general_validation_metrics(y_hat, y, "test")
        # if 'patch_coords' in batch.keys():
        #     self._save_reconstruction_images(images=y_hat[1], index=batch['original_index'],
        #                                      coords=batch['patch_coords'])
        # else:
        #     self._save_reconstruction_images(images=y_hat[1], index=batch['index'])

    def log_general_validation_metrics(self, y_hat_tp, y_tp, stage):
        y_hat_tp = y_hat_tp if isinstance(y_hat_tp, tuple) else (y_hat_tp,)
        y_tp = y_tp if isinstance(y_tp, tuple) else (y_tp,)
        # ensure matched y and y_hat is same
        # zip will stop at the shortest length
        for y_hat, y in zip(y_hat_tp, y_tp):
            if len(y_hat.shape) == 2 or len(y_hat.shape) == 3:
                if len(y_hat.shape) == 3:
                    y_hat = y_hat.reshape(-1, y_hat.shape[-1])
                    y = y.unsqueeze(1).repeat(1, 3).reshape(-1)
                if y_hat.shape[1] == 1:  # Regression task
                    self.log_dict(self._regression_metrics(y_hat, y, stage), batch_size=y.size(0))
                    logger.info(f"evaluation: {self._regression_metrics(y_hat, y, stage)}")

                elif y_hat.shape[1] == 2:  # Binary classification task
                    self.log_dict(self._binary_classification_metrics(y_hat, y, stage), batch_size=y.size(0))
                    logger.info(f"evaluation: {self._binary_classification_metrics(y_hat, y, stage)}")
                    self.logger.experiment.add_pr_curve(f"{stage}_pr_curve", y, y_hat[:, 1],
                                                        global_step=self.global_step)

                elif y_hat.shape[1] > 2:  # Multi-class classification task
                    self.log_dict(self._multiclass_classification_metrics(y_hat, y, stage), batch_size=y.size(0))
                    logger.info(f"evaluation: {self._multiclass_classification_metrics(y_hat, y, stage)}")
                else:
                    raise ValueError("Invalid shape for y_hat.")
            elif len(y_hat.shape) == 4:  # Image reconstruction task
                self.log_dict(self._image_reconstruction_metrics(y_hat, y, stage), batch_size=y.size(0))
                logger.info(f"evaluation: {self._image_reconstruction_metrics(y_hat, y, stage)}")

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

    @staticmethod
    def _regression_metrics(y_hat, y, stage):
        return {
            f"{stage}/mae": l1_loss(y_hat.reshape(-1), y),
            f"{stage}/rmse": mse_loss(y_hat.reshape(-1), y).sqrt()
        }

    @staticmethod
    def _binary_classification_metrics(y_hat, y, stage, task="binary"):
        pred = torch.argmax(y_hat, dim=1)  # Get class predictions
        probs = torch.sigmoid(y_hat[:, 1])  # Get probabilities for the positive class
        param_dt = dict(task=task)
        return {
            f"{stage}/accuracy": float(accuracy(pred, y, **param_dt)),
            f"{stage}/precision": float(precision(pred, y, **param_dt)),
            f"{stage}/recall": float(recall(pred, y, **param_dt)),
            f"{stage}/f1": float(f1_score(pred, y, **param_dt)),
            f"{stage}/auc": float(auroc(probs, y, **param_dt))
        }

    @staticmethod
    def _multiclass_classification_metrics(y_hat, y, stage, average='macro', task="multiclass"):
        pred = torch.argmax(y_hat, dim=1)  # Get class predictions
        probs = torch.softmax(y_hat, dim=1)
        param_dt = dict(average=average, num_classes=y_hat.shape[1], task=task)
        return {
            f"{stage}/accuracy": float(accuracy(pred, y, **param_dt)),
            f"{stage}/precision": float(precision(pred, y, **param_dt)),
            f"{stage}/recall": float(recall(pred, y, **param_dt)),
            f"{stage}/f1": float(f1_score(pred, y, **param_dt)),
            f"{stage}/auc": float(auroc(probs, y, **param_dt))
        }

    @staticmethod
    def _image_reconstruction_metrics(y_hat, y, stage):
        # input range [0, 1]
        psnr_metric = PeakSignalNoiseRatio().to(y.device)
        ssim_metric = StructuralSimilarityIndexMeasure().to(y.device)
        lpips_metric = LearnedPerceptualImagePatchSimilarity().to(y.device)
        y_hat_lpips = y_hat.repeat(1, 3, 1, 1) * 2 - 1 if y_hat.size(1) == 1 else y_hat * 2 - 1
        y_lpips = y.repeat(1, 3, 1, 1) * 2 - 1 if y.size(1) == 1 else y * 2 - 1
        return {
            f"{stage}/psnr": float(psnr_metric(y_hat, y)),
            f"{stage}/ssim": float(ssim_metric(y_hat, y)),
            f"{stage}/lpips": float(lpips_metric(y_hat_lpips, y_lpips)),
            f"{stage}/mae": float(l1_loss(y_hat, y)),
            f"{stage}/mse": float(mse_loss(y_hat, y))
        }
