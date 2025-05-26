#!/bin/bash

# Set the CONFIGS_LOCATION environment variable if needed,
# else the default location(cwd) is used.
#export CONFIGS_LOCATION=/path/to/configs

Execute the Python script, please use correct conda env.
python anchor_cls.py +test_mode=True \
dataset.loader.train_loader.batch_size=128 model.model_params.1.num_classes=2 \
+ckpt_path="exp/2025-04-12_22-26-35_job/0/lightning_logs/version_0/checkpoints/model_epoch\=169_val_loss\=0.92.ckpt" \
dataset.load.fold=0 trainer.max_epochs=50000

python anchor_cls.py +test_mode=True \
dataset.loader.train_loader.batch_size=128 model.model_params.1.num_classes=2 \
+ckpt_path="exp/2025-04-12_22-26-35_job/1/lightning_logs/version_0/checkpoints/model_epoch\=101_val_loss\=0.56.ckpt" \
dataset.load.fold=1 trainer.max_epochs=50000

python anchor_cls.py +test_mode=True \
dataset.loader.train_loader.batch_size=128 model.model_params.1.num_classes=2 \
+ckpt_path="exp/2025-04-12_22-26-35_job/2/lightning_logs/version_0/checkpoints/model_epoch\=103_val_loss\=0.50.ckpt" \
dataset.load.fold=2 trainer.max_epochs=50000

python anchor_cls.py +test_mode=True \
dataset.loader.train_loader.batch_size=128 model.model_params.1.num_classes=2 \
+ckpt_path="exp/2025-04-12_22-26-35_job/3/lightning_logs/version_0/checkpoints/last.ckpt" \
dataset.load.fold=3 trainer.max_epochs=50000

python anchor_cls.py +test_mode=True \
dataset.loader.train_loader.batch_size=128 model.model_params.1.num_classes=2 \
+ckpt_path="exp/2025-04-12_22-26-35_job/4/lightning_logs/version_0/checkpoints/model_epoch\=107_val_loss\=0.47.ckpt" \
dataset.load.fold=4 trainer.max_epochs=50000


#python main.py -m +param=1,2,3  # Example of multirun
