import torch
import pytorch_lightning as pl

OPTIM_DT = {
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD,
    'AdamW': torch.optim.AdamW,
}
LR_SCHEDULER_DT = {
    'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'StepLR': torch.optim.lr_scheduler.StepLR,
    'MultiStepLR': torch.optim.lr_scheduler.MultiStepLR,
}


class ExampleModule(pl.LightningModule):
    def __init__(self,
                 model: str,
                 model_params: dict,
                 optimizer: str,
                 optim_params: dict,
                 lr_scheduler: str = None,
                 lr_scheduler_params: dict = None,
                 lr_scheduler_other_params: dict = None):
        super().__init__()
        # model structure
        # optimizer and lr_scheduler settings
        self.optimizer = optimizer
        self.optim_params = optim_params
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params
        self.lr_scheduler_other_params = lr_scheduler_other_params

        assert self.lr_scheduler_params == self.lr_scheduler == self.lr_scheduler_other_params, \
            'if lr_scheduler is valid, lr_scheduler_params must be provided'

        # model structure

    def configure_optimizers(self):
        """
        choose optimizer and lr_scheduler(optional)
        """
        opt_func = OPTIM_DT[self.optimizer]
        optimizer = opt_func(self.parameters(), **self.optim_params)
        if self.lr_scheduler is not None:
            lr_scheduler_func = LR_SCHEDULER_DT[self.lr_scheduler]  # get lr_scheduler function
            scheduler = {
                'scheduler': lr_scheduler_func(optimizer, **self.lr_scheduler_params),
                **self.lr_scheduler_other_params,  # monitor, interval, frequency, etc.
            }
            return [optimizer], [scheduler]
        else:
            return optimizer
