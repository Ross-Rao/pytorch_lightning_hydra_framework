# IRCL

## Install

### install torch

```
conda create -n mvi-ibs python=3.9
# select specific CUDA version
# CUDA 11.8
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# CUDA 12.6
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

### pip install

```
pip install -r requirements.txt
pip install monai[itk]
```

### cuda kernal

```
cd models/adaptive_clustering_transformer/adaptive_clustering_attention/extensions
pip install -e .
```

## Usage

### config
```
export DATASET_LOCATION=~/Ross/datasets
export EXPERIMENT_LOCATION=~/Ross/exp_results/IRCL-UNet
```

### run
```
python anchor_cls.py -m dataset.load.fold=0,1,2,3,4
```