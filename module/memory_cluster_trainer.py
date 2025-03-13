# python import
import logging
# package import
import torch
import pytorch_lightning as pl
from torch.nn.functional import l1_loss, mse_loss
from torchmetrics.functional.classification import auroc, accuracy, precision, recall, f1_score
# local import
import models as models

logger = logging.getLogger(__name__)


class ExampleModule(pl.LightningModule):
    def __init__(self,
                 model: str,
                 model_params: dict,
                 optimizer: str,
                 optimizer_params: dict,
                 criterion: str,
                 lr_scheduler: str = None,
                 lr_scheduler_params: dict = None,
                 lr_scheduler_other_params: dict = None):
        super().__init__()
        # model structure settings
        self.model = getattr(models, model)(**model_params)

        # optimizer settings
        self.optimizer = getattr(torch.optim, optimizer)(self.parameters(), **optimizer_params)

        # loss function settings
        self.criterion = getattr(torch.nn, criterion)()

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
        # for external test data, index is not provided
        image = batch['image']
        index = batch.get('index', None)
        neighbor = batch.get('neighbor_index', None)
        label = batch['label']
        return image, index, neighbor, label

    def _step(self, batch):
        x, index, neighbor, _ = self.get_batch(batch)
        return self.model(x, index, neighbor)

    def on_train_epoch_start(self, max_search_ratio=0.5):
        # on_train_epoch_start may not suitable for update model parameters, maybe works?
        # we update anchor in training_step with anchor_update_frequency
        current_epoch = self.current_epoch
        update_epoch = [20, 60, 100, 140, 180, 220, 260, 300]
        if current_epoch in update_epoch:
            max_epoch = self.trainer.max_epochs
            search_ratio = current_epoch / max_epoch * max_search_ratio
            self.model.mc.update_anchor(search_rate=search_ratio)

    def training_step(self, batch, batch_idx):
        x, index, neighbor, cls = self.get_batch(batch)
        y_hat, recon_loss, instance_loss, anchor_loss = self._step(batch)
        self_loss = recon_loss + instance_loss + anchor_loss    # self_loss is NCELoss for self-supervised learning

        if len(y_hat.shape) == 2:
            if y_hat.shape[1] == 1:  # Regression task
                y_hat = y_hat.reshape(-1)  # Flatten the output

        cls_loss = self.criterion(y_hat, cls)
        loss = 10 * cls_loss + self_loss
        self.log("train_loss", loss, on_epoch=True,
                 logger=True, batch_size=x.size(0), prog_bar=True)  # train_loss is the key for callback
        loss_dict = {
            "train/cls_loss": float(cls_loss),
            "train/recon_loss": float(recon_loss),  # tensor to float
            "train/instance_loss": float(instance_loss),
            "train/anchor_loss": float(anchor_loss),
            "train/self_loss": float(self_loss)
        }
        self.log_dict(loss_dict, on_epoch=True, logger=True, batch_size=x.size(0), prog_bar=True)

        logger.info(f"epoch: {self.current_epoch} " + str(loss_dict))
        logger.info(f"epoch: {self.current_epoch} train_loss: {float(loss)}")
        return loss

    def validation_step(self, batch, batch_idx):
        x, index, neighbor, cls = self.get_batch(batch)
        y_hat, pred_indices, pred_flag = self._step(batch)

        self.log_dict({
            'val/anchor_ratio': (pred_flag >= 0).float().mean(),
        }, on_epoch=True, logger=True, batch_size=x.size(0))

        loss = self.criterion(y_hat, cls)  # only cls loss
        self.log("val_loss", loss, on_epoch=True,
                 logger=True, batch_size=x.size(0))  # val_loss is the key for callback

        self.routine(y_hat, cls, "val")
        return loss

    def test_step(self, batch, batch_idx):
        x, index, neighbor, cls = self.get_batch(batch)
        y_hat, pred_indices, pred_flag = self._step(batch)

        self.log_dict({
            'test/anchor_ratio': (pred_flag >= 0).float().mean(),
        }, on_epoch=True, logger=True, batch_size=x.size(0))

        self.routine(y_hat, cls, "test")

    def routine(self, y_hat, y, stage):
        if len(y_hat.shape) == 2:
            if y_hat.shape[1] == 1:
                # Regression task
                self.log_dict(self._regression_metrics(y_hat, y, stage),
                              on_epoch=True, logger=True, batch_size=y.size(0))
                logger.info(f"evaluation: {self._regression_metrics(y_hat, y, stage)}")
            elif y_hat.shape[1] == 2:
                # Binary classification task
                self.log_dict(self._binary_classification_metrics(y_hat, y, stage),
                              on_epoch=True, logger=True, batch_size=y.size(0))
                logger.info(f"evaluation: {self._binary_classification_metrics(y_hat, y, stage)}")
                self.logger.experiment.add_pr_curve(f"{stage}_pr_curve", y, y_hat[:, 1], global_step=self.global_step)
            elif y_hat.shape[1] > 2:
                # Multi-class classification task
                self.log_dict(self._multiclass_classification_metrics(y_hat, y, stage),
                              on_epoch=True, logger=True, batch_size=y.size(0))
                logger.info(f"evaluation: {self._multiclass_classification_metrics(y_hat, y, stage)}")
            else:
                raise ValueError("Invalid shape for y_hat.")

    @staticmethod
    def _regression_metrics(y_hat, y, stage):
        return {
            f"{stage}/mae": float(l1_loss(y_hat.reshape(-1), y)),
            f"{stage}/rmse": float(mse_loss(y_hat.reshape(-1), y).sqrt())
        }

    @staticmethod
    def _binary_classification_metrics(y_hat, y, stage):
        pred = torch.argmax(y_hat, dim=1)  # Get class predictions
        probs = torch.sigmoid(y_hat[:, 1])  # Get probabilities for the positive class
        return {
            f"{stage}/accuracy": float(accuracy(pred, y, task="binary")),
            f"{stage}/precision": float(precision(pred, y, task="binary")),
            f"{stage}/recall": float(recall(pred, y, task="binary")),
            f"{stage}/f1": float(f1_score(pred, y, task="binary")),
            f"{stage}/auc": float(auroc(probs, y, task="binary"))
        }

    @staticmethod
    def _multiclass_classification_metrics(y_hat, y, stage):
        pred = torch.argmax(y_hat, dim=1)  # Get class predictions
        probs = torch.softmax(y_hat, dim=1)
        num_classes = y_hat.shape[1]
        return {
            f"{stage}/accuracy": float(accuracy(pred, y, average='macro', num_classes=num_classes, task="multiclass")),
            f"{stage}/precision": float(precision(pred, y, average='macro', num_classes=num_classes, task="multiclass")),
            f"{stage}/recall": float(recall(pred, y, average='macro', num_classes=num_classes, task="multiclass")),
            f"{stage}/f1": float(f1_score(pred, y, average='macro', num_classes=num_classes, task="multiclass")),
            f"{stage}/auc": float(auroc(probs, y, average='macro', num_classes=num_classes, task="multiclass"))
        }
