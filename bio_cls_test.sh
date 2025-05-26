#!/bin/bash

# Set the CONFIGS_LOCATION environment variable if needed,
# else the default location(cwd) is used.
#export CONFIGS_LOCATION=/path/to/configs



python bio_cls.py +visual=True +test_mode=True \
dataset.loader.train_loader.batch_size=128 model.model_params.1.num_classes=2 \
+ckpt_path="exp/2025-05-03_09-21-34_job/1/lightning_logs/version_0/checkpoints/model_epoch\=101_val_loss\=0.88.ckpt" \
dataset.load.fold=1 trainer.max_epochs=50000



#python main.py -m +param=1,2,3  # Example of multirun
