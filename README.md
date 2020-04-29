Whole-slide classification pipeline &mdash; end-to-end
======

This repository will give an overview on how to use [streaming](https://github.com/DIAGNijmegen/StreamingCNN) to train whole slides to single labels. Streaming is an implementation of convolutions using tiling and gradient checkpointing to safe memory.

Papers until now about this method:

- Full paper (in review; arXiv preprint): http://arxiv.org/abs/1911.04432
- MIDL 2018 (abstract, proof of concept): https://openreview.net/forum?id=HJ7lIcjoM

## Network

For now, only the *ResNet-34* implementation is checked. Other networks could be implemented.

## Input sizes

Recommended image sizes:

- 4096x4096 for spacing 4.0 (2.5x)
- 8192x8192 for spacing 2.0 (5x)
- 16384x16384 for spacing 1.0 (10x)

## Steps

### 0. Prepare train.csv and val.csv

For this pipeline you will need two csv files: `train` and `val.csv`. The syntax is easy:

```csv
slide_id,label
TRAIN_1,1
TRAIN_2,1
...
```

### 1. Prepare data

```bash
python trim_tissue.py \
    --csv='' \
    --slide-dir='' \
    --filetype='tif' \
    --save-dir='' \
    --output-spacing=1.0
```

### 2. Train network!
```bash
python train.py \
    --name=test-name \
    --train_csv='train.csv' \
    --val_csv='val.csv' \
    --data_dir='/local/data' \
    --save_dir='/home/user/models' \
    --lr=2e-4 \
    --num_workers=1 \
    --tile_size=5120
```


