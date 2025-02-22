import torch
import pytorch_lightning as pl
from torch.nn.functional import l1_loss, mse_loss
from torchmetrics.functional.classification import auroc, accuracy, precision, recall, f1_score
# local import
import module.models as models


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

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        x, y = batch
        y_hat = self.model(x)

        if len(y_hat.shape) == 2:
            if y_hat.shape[1] == 1:  # Regression task
                y_hat = y_hat.reshape(-1)  # Flatten the output

        loss = self.criterion(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss)  # train_loss is the key for callback
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss)  # val_loss is the key for callback
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        if len(y_hat.shape) == 2:
            if y_hat.shape[1] == 1:
                # Regression task
                self.log_dict(self._regression_metrics(y_hat, y, "test"))
            elif y_hat.shape[1] == 2:
                # Binary classification task
                self.log_dict(self._binary_classification_metrics(y_hat, y, "test"))
                self.logger.experiment.add_pr_curve("test_pr_curve", y, y_hat[:, 1], global_step=self.global_step)
            elif y_hat.shape[1] > 2:
                # Multi-class classification task
                self.log_dict(self._multiclass_classification_metrics(y_hat, y, "test"))
            else:
                raise ValueError("Invalid shape for y_hat.")

    @staticmethod
    def _regression_metrics(y_hat, y, stage):
        return {
            f"{stage}/mae": l1_loss(y_hat.reshape(-1), y),
            f"{stage}/rmse": mse_loss(y_hat.reshape(-1), y).sqrt()
        }

    @staticmethod
    def _binary_classification_metrics(y_hat, y, stage):
        pred = torch.argmax(y_hat, dim=1)  # Get class predictions
        probs = y_hat[:, 1]  # Get probabilities for the positive class
        return {
            f"{stage}/accuracy": accuracy(pred, y, task="binary"),
            f"{stage}/precision": precision(pred, y, task="binary"),
            f"{stage}/recall": recall(pred, y, task="binary"),
            f"{stage}/f1": f1_score(pred, y, task="binary"),
            f"{stage}/auc": auroc(probs, y, task="binary")
        }

    @staticmethod
    def _multiclass_classification_metrics(y_hat, y, stage):
        pred = torch.argmax(y_hat, dim=1)  # Get class predictions
        num_classes = y_hat.shape[1]
        return {
            f"{stage}/accuracy": accuracy(pred, y, average='macro', num_classes=num_classes, task="multiclass"),
            f"{stage}/precision": precision(pred, y, average='macro', num_classes=num_classes, task="multiclass"),
            f"{stage}/recall": recall(pred, y, average='macro', num_classes=num_classes, task="multiclass"),
            f"{stage}/f1": f1_score(pred, y, average='macro', num_classes=num_classes, task="multiclass"),
            f"{stage}/auc": auroc(y_hat, y, average='macro', num_classes=num_classes, task="multiclass")
        }
