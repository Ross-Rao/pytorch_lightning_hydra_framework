from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning_framework.DataModule.random_dataset import RandomDataModule
from pytorch_lightning_framework.Model.simple_model import SimpleModel


__all__ = ['get_callbacks']


def get_callbacks(callback_cfg: dict):
    """
    transform callback config to callback instance, config are expected from yaml files.

    :param callback_cfg: dict of callback config, key must be in cfg_dt.keys()
    :return: list of callback instance
    """
    cfg_dt = {
        'ModelCheckpoint': ModelCheckpoint,
    }
    callback_lt = []
    for key in cfg_dt.keys():
        callback = cfg_dt[key](**callback_cfg[key])
        callback_lt.append(callback)
    return callback_lt


if __name__ == "__main__":
    # 初始化数据模块和模型
    data_module = RandomDataModule(batch_size=32)
    model = SimpleModel()

    # 配置 ModelCheckpoint 回调
    callback_dt = {
        'ModelCheckpoint': dict(filename='model_{epoch:02d}_{val_loss:.2f}', monitor='val_loss',
                                save_top_k=3, mode='min', save_last=True)
    }
    callbacks = get_callbacks(callback_dt)

    # 初始化 Trainer
    trainer = Trainer(
        max_epochs=100,
        callbacks=callbacks,
        accelerator="auto",  # 自动选择设备（CPU/GPU）
        enable_progress_bar=True
    )

    # 训练模型
    trainer.fit(model, datamodule=data_module)
