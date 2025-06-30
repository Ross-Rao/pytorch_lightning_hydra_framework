# python import
import logging
from typing import Any, Union
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
from utils import lr_scheduler as custom_lr_scheduler

logger = logging.getLogger(__name__)
__all__ = ['ExampleModule']


class ExampleModule(pl.LightningModule):
    def __init__(self,
                 model: Union[str, list[str]],
                 model_params: Union[dict, list[dict]],
                 optimizer: Union[str, list[str]],
                 optimizer_params: Union[dict, list[dict]],
                 criterion: Union[str, list[str]],
                 criterion_params: Union[dict, list[dict]] = None,
                 lr_scheduler: Union[str, list[str]] = None,
                 lr_scheduler_params: Union[dict, list[dict]] = None,
                 lr_scheduler_other_params: Union[dict, list[dict]] = None):
        super().__init__()
        # model structure settings
        assert isinstance(model, list) == isinstance(model_params, list), \
            "model and model_params must either both be lists or neither be lists"
        self.model = getattr(models, model)(**model_params) \
            if isinstance(model_params, dict) and isinstance(model, str) else \
            torch.nn.ModuleList([getattr(models, m)(**mp).to(self.device) for m, mp in zip(model, model_params)])

        # optimizer settings
        assert isinstance(optimizer, list) == isinstance(optimizer_params, list), \
            "optimizer and optimizer_params must either both be lists or neither be lists"
        self.optimizer = getattr(torch.optim, optimizer)(self.parameters(), **optimizer_params) \
            if isinstance(optimizer, str) and isinstance(optimizer_params, dict) else \
            [getattr(torch.optim, opt)(m.parameters(), **opt_p)
             for m, opt, opt_p in zip(self.model, optimizer, optimizer_params)]

        # loss function settings
        self.criterion = get_multi_attr([torch.nn, models], {criterion: criterion_params})[0] \
            if isinstance(criterion, str) else \
            torch.nn.ModuleList(
                [get_multi_attr([torch.nn, models], {c: cp})[0]
                 for c, cp in zip(criterion, criterion_params)]
            )

        # lr_scheduler settings
        lr_lt = [lr_scheduler, lr_scheduler_params, lr_scheduler_other_params]
        assert all(var is None for var in lr_lt) or all(var is not None for var in lr_lt), \
            'if lr_scheduler is valid, lr_scheduler_params and lr_scheduler_other_params must be provided'
        if lr_scheduler is None:
            self.lr_scheduler = None
        else:
            # StepLR, ReduceLROnPlateau, etc.
            lr_scheduler_func = get_multi_attr([torch.optim.lr_scheduler, custom_lr_scheduler], lr_scheduler) \
            if isinstance(lr_scheduler, str) else \
                [get_multi_attr([torch.optim.lr_scheduler, custom_lr_scheduler], ls) for ls in lr_scheduler]
            if isinstance(lr_scheduler_func, list):
                self.lr_scheduler = [{**{'scheduler': ls(opt, **ls_p)}, **ls_op} 
                                     for opt, ls, ls_p, ls_op in zip(self.optimizer, lr_scheduler_func,
                                                                     lr_scheduler_params, lr_scheduler_other_params)]
            else:
                if isinstance(self.optimizer, list):
                    self.lr_scheduler = [{**{'scheduler': lr_scheduler_func(opt, **lr_scheduler_params)},
                                          **lr_scheduler_other_params}
                                         for opt in self.optimizer]
                else:
                    self.lr_scheduler = {
                        'scheduler': lr_scheduler_func(self.optimizer, **lr_scheduler_params),
                        **lr_scheduler_other_params,  # monitor, interval, frequency, etc.
                    }

        # metrics
        temp_dt = model_params if isinstance(model_params, dict) else {k: v for d in model_params for k, v in d.items()}
        n_cls = temp_dt.get('num_classes', temp_dt.get('out_features', temp_dt.get('out_channels', 10)))
        multi_cls_param = dict(average='macro', task='multiclass', num_classes=n_cls)
        self.confusion_matrix = ConfusionMatrix(num_classes=n_cls, task='multiclass').eval()
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
        self._val_cls_metrics = self._train_cls_metrics.clone(prefix="val/").eval()
        self._val_recon_metrics = self._train_recon_metrics.clone(prefix="val/").eval()
        self._val_reg_metrics = self._train_reg_metrics.clone(prefix="val/").eval()
        self._test_cls_metrics = self._train_cls_metrics.clone(prefix="test/").eval()
        self._test_recon_metrics = self._train_recon_metrics.clone(prefix="test/").eval()
        self._test_reg_metrics = self._train_reg_metrics.clone(prefix="test/").eval()
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

        self.tensor_to_index = {}
        self.i_count = 0
        self.max_search_ratio = 1.0
        self.anchor_update_frequency = 20
        self.stage_change_epoch = 100

    def get_index(self, tensors):
        if not self.training and self.current_epoch == 0:
            ff = 60000
        else:
            ff = 0
        import hashlib
        res = []
        for i, tensor in enumerate(tensors):
            # 计算 tensor 的哈希值
            tensor_hash = hashlib.sha256(torch.flatten(tensor).cpu().numpy().tobytes()).hexdigest()
            # 将哈希值和 index 保存到字典中
            if tensor_hash not in self.tensor_to_index:
                self.tensor_to_index[tensor_hash] = self.i_count + ff
                self.i_count += 1
            res.append(self.tensor_to_index[tensor_hash])
        res = torch.tensor(res)
        return res

    def configure_optimizers(self):
        """
        Set optimizer and lr_scheduler(optional)
        """
        optimizer, lr_scheduler = self.optimizer, self.lr_scheduler
        if lr_scheduler is not None:
            if isinstance(optimizer, list):
                return [optimizer[0]], [lr_scheduler[0]]
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    # --------------------------------------------------------------------------------------- #
    def get_batch(self, batch):
        if isinstance(batch, list):
            image, label = batch
            b, c, h, w = image.size()
            index = self.get_index(image).to(image.device)
            index = c * index.unsqueeze(1) + torch.arange(c).to(image.device)
            index = index.reshape(-1)
            if image.size(-1) != 64 or image.size(-2) != 64:
                image = torch.nn.functional.interpolate(image, size=(64, 64), mode='bilinear', align_corners=False)
            return (image, index, None), label
        elif isinstance(batch, dict):
            image = batch.get('image')
            index = batch.get('index', None)
            b, c, h, w = image.size()
            image = image.reshape(b * c, 1, h, w)
            if index is None or not isinstance(index, torch.Tensor):
                index = self.get_index(image)
            else:
                if not self.training:
                    index += 1000000
            neighbor_index = [(c * index.unsqueeze(1) + torch.roll(torch.arange(c), i).to(image.device)).reshape(-1)
                              for i in range(1, c)]
            neighbor_index = torch.stack(neighbor_index, dim=1)
            index = c * index.unsqueeze(1) + torch.arange(c).to(image.device)
            index = index.reshape(-1)
            x = (image, index, neighbor_index)
            y = batch['label'].reshape(-1) if batch['label'].shape == torch.Size([b, 1]) else batch['label']
            y = y.repeat_interleave(c, dim=0)
            if self.model[1].num_classes == 2 and y.ndim == 1:
                y = torch.where(y >= 1, 1, 0)
            return x, y
        else:
            raise ValueError('Invalid batch type')

    # --------------------------------------------------------------------------------------- #

    @staticmethod
    def get_batch_size(batch):
        if isinstance(batch, list):
            return batch[0].size(0)
        elif isinstance(batch, dict):
            return batch['image'].size(0)
        else:
            raise ValueError('Invalid batch type')

    # --------------------------------------------------------------------------------------- #
    def on_train_epoch_start(self):
        max_search_ratio = self.max_search_ratio
        frequency = self.anchor_update_frequency
        stage_change_epoch = self.stage_change_epoch
        # frequency: update anchor frequency
        # stage_change_epoch: change stage from 1 to 2
        if self.current_epoch == stage_change_epoch:
            self.model[0].get_all_cluster()
            self.trainer.optimizers = [self.optimizer[1]]

        if self.current_epoch == 0:
            self.i_count = 0

        # on_train_epoch_start may not suitable for update model parameters, maybe works for buffer update
        # we update anchor in training_step with anchor_update_frequency
        self.trainer.train_dataloader.shuffle = True
        update_epoch = [i for i in range(frequency, stage_change_epoch - frequency + 1, frequency)]
        if self.current_epoch in update_epoch:
            search_ratio = self.current_epoch / (stage_change_epoch - frequency) * max_search_ratio
            self.model[0].update_anchor(search_rate=search_ratio)

    # --------------------------------------------------------------------------------------- #

    def model_step(self, batch, batch_idx):
        x, y = self.get_batch(batch)
        model_params = x if isinstance(x, tuple) else (x,)
        # y_hat = self.model(*model_params)
        # --------------------------------------------------------------------------------------- #
        if self.current_epoch < self.stage_change_epoch and self.trainer.state.stage != "test":
            loss_dt = self.model[0](*model_params, loss=True)
            return (), loss_dt
        else:
            for param in self.model[0].parameters():
                param.requires_grad = False
            cluster, hid_x = self.model[0](*model_params, loss=False)
            y_hat = self.model[1](hid_x, cluster)
            if self.trainer.state.stage != "test":
                return y, y_hat
            else:
                return y, (y_hat, cluster)
        # --------------------------------------------------------------------------------------- #

    def criterion_step(self, y, y_hat):
        # be sure that y_hat params first and y params later in your criterion function
        criterion_params = (y_hat if isinstance(y_hat, tuple) else (y_hat,)) + (y if isinstance(y, tuple) else (y,))
        if self.current_epoch < self.stage_change_epoch and self.trainer.state.stage != "test":
            loss_dt = self.criterion[0](*criterion_params)
        else:
            loss_dt = self.criterion[1](*criterion_params)
        # # --------------------------------------------------------------------------------------- #
        # loss = loss_dt if isinstance(loss_dt, torch.Tensor) else loss_dt['loss']
        # if self.training:
        #     opt0, opt1 = self.optimizers()
        #     sch0, sch1 = self.lr_schedulers()
        #     if self.current_epoch < self.stage_change_epoch:
        #         opt0.zero_grad()
        #         self.manual_backward(loss)
        #         opt0.step()
        #         sch0.step(loss)
        #     else:
        #         opt1.zero_grad()
        #         self.manual_backward(loss)
        #         opt1.step()
        #         sch1.step(loss)
        # # --------------------------------------------------------------------------------------- #
        # loss = self.criterion(*criterion_params)
        return loss_dt

    def training_step(self, batch, batch_idx):
        # --------------------------------- #
        # if self.current_epoch >= self.stage_change_epoch:
        #     from monai.transforms import MixUpD
        #     import torch.nn.functional as f
        #     # please modify the following code in .../site-packages/monai/transforms/regularization/array.py:85
        #     # mixweight = weight[(Ellipsis,) + (None,) * len(dims)].to(data.device)
        #     # please modify the following code in .../site-packages/monai/transforms/regularization/array.py:82
        #     # if len(dims) not in [1, 3, 4]:
        #     # if use mix-up, you are recommend to use shuffle in dataloader
        #     if self.model.num_classes == 2:
        #         batch['label'] = torch.where(batch['label'] >= 1, 1, 0)
        #     batch['label'] = f.one_hot(batch['label'], num_classes=self.model.num_classes).float()
        #     mix = MixUpD(keys=['image', 'label'], batch_size=self.get_batch_size(batch))
        #     batch = mix(batch)
        # n = self.expand_ratio * self.get_batch_size(batch)
        # import torch.nn.functional as f
        # if self.mix.get(batch_idx, False):
        #     mix, mix_ratio = self.mix[batch_idx]
        # else:
        #     import numpy as np
        #     mix = torch.randint(0, self.get_batch_size(batch), (n, 2)).to(self.device)
        #     mix_ratio = torch.tensor(np.random.beta(1, 1, size=n)).to(self.device)
        #     self.mix[batch_idx] = (mix, mix_ratio)
        # if self.model.num_classes == 2:
        #     batch['label'] = torch.where(batch['label'] >= 1, 1, 0)
        # batch['label'] = f.one_hot(batch['label'], num_classes=self.model.num_classes).float()
        # mix_image = (batch['image'][mix[:, 0]] * mix_ratio.unsqueeze(1).unsqueeze(2).unsqueeze(3) +
        #              batch['image'][mix[:, 1]] * (1 - mix_ratio).unsqueeze(1).unsqueeze(2).unsqueeze(3))
        # mix_label = (batch['label'][mix[:, 0]] * mix_ratio.unsqueeze(1) +
        #              batch['label'][mix[:, 1]] * (1 - mix_ratio).unsqueeze(1))
        # index = torch.arange(n).to(self.device) + batch_idx * n
        # batch['image'], batch['label'], batch['index'] = mix_image.float(), mix_label, index
        # --------------------------------- #
        y, y_hat = self.model_step(batch, batch_idx)
        # self._update_metrics(y_hat, y, "train")  # not necessary, only debug

        loss = self.criterion_step(y, y_hat)
        return loss

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        # log step loss
        loss_dt = {f'train/{k}': float(v) for k, v in outputs.items()}
        self.log_dict(loss_dt, batch_size=self.get_batch_size(batch), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        y, y_hat = self.model_step(batch, batch_idx)
        self._update_metrics(y_hat, y, "val")

        loss = self.criterion_step(y, y_hat)
        return loss

    def on_validation_batch_end(
            self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        # log step loss
        outputs = {'loss': outputs} if isinstance(outputs, torch.Tensor) else outputs
        loss_dt = {f'val/{k}': float(v) for k, v in outputs.items()}
        self.log_dict(loss_dt, batch_size=self.get_batch_size(batch), prog_bar=True)

    def test_step(self, batch, batch_idx):
        y, y_hat = self.model_step(batch, batch_idx)

        # `patch_coords` comes from `GridPatchDataset`, used for saving images
        if isinstance(batch, dict):
            extra_params = {'index': batch.get('index', None), 'coords': batch.get('patch_coords', None)}
        else:
            extra_params = {}
        self._update_metrics(y_hat, y, "test", **extra_params)

        # # --------------------------------- #
        # img, index, _ = x
        # stored_index, flags = self.model.get_anchor(img)
        # df = pd.DataFrame({'input_index': index.cpu(), 'stored_index': stored_index.cpu(), 'flags': flags.cpu()})
        # df['flags'] = df['flags'].apply(lambda flag: 'anchor' if flag >= 0 else 'instance')
        # df.to_csv(os.path.join(self.logger.log_dir, f'anchor_{batch_idx}.csv'), index=False)
        # # --------------------------------- #

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get('train/loss')
        if train_loss is not None:
            self.log('train_loss', train_loss)  # train_loss is the key for callback
            logger.info(f"\nEpoch {self.current_epoch} - train_loss: {train_loss}")  # print train loss to log file
        # not necessary, only debug
        for metrics_dict in [self.cls_metrics['train'], self.reg_metrics['train'], self.recon_metrics['train']]:
            for metric_name, metric in metrics_dict.items():
                # re-comment `update_metrics` in training_step if you want to use this
                if metric.update_count > 0:
                    res = metric.compute()
                    self.log(metric_name, res, prog_bar=True)
                    metric.reset()

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get('val/loss')
        if val_loss is not None:
            self.log('val_loss', val_loss)  # val_loss is the key for callback
            logger.info(f"\nEpoch {self.current_epoch} - val_loss: {val_loss}")  # print val loss to log file
        for metrics_dict in [self.cls_metrics['val'], self.reg_metrics['val'], self.recon_metrics['val']]:
            for metric_name, metric in metrics_dict.items():
                if metric.update_count > 0:
                    res = metric.compute()
                    self.log(metric_name, res, prog_bar=True)
                    logger.info(f"Epoch {self.current_epoch} - {metric_name}: {res}")
                    metric.reset()

    def on_test_epoch_end(self):
        if self.confusion_matrix.update_count > 0:
            cm = self.confusion_matrix.compute()
            plt, _ = plot_confusion_matrix(cm)
            self.logger.experiment.add_figure(f"test_confusion_matrix", plt)
            logger.info(f"Confusion Matrix:\n{cm}")
            self.confusion_matrix.reset()
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
            if len(y_hat.shape) == 2 or len(y_hat.shape) == 3:
                if len(y_hat.shape) == 3:
                    y_hat = y_hat.reshape(-1, y_hat.shape[-1])
                    y = y.unsqueeze(1).repeat(1, 3).reshape(-1)
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
                # if stage == "test":
                #     self._save_reconstruction_images(y_hat, **kwargs)

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
        y_hat_lpips = torch.clamp(y_hat_lpips, -1, 1)

        # 更新其他指标
        for metric_name, metric in self.recon_metrics[stage].items():
            if metric_name != f"{stage}/lpips":
                metric.update(y_hat, y)
        # 单独更新lpips
        self.recon_metrics[stage]["lpips"].update(y_hat_lpips, y_lpips)
