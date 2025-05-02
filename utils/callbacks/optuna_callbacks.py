from optuna.integration import PyTorchLightningPruningCallback


__all__ = ["CustomPruningCallback"]


class CustomPruningCallback(PyTorchLightningPruningCallback):
    def __init__(self, start_epoch=0, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = start_epoch

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_epoch:
            super().on_validation_end(trainer, pl_module)
