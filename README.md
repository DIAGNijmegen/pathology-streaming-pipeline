Streaming CNNs
======

This repository will give an overview on how to use streaming to train whole slides to single labels.

## Parameters

For simplicity the network is fixed, you will train a **resnet34**. 
Ask me if you want to train something else.

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

*Time: few seconds per slide*

```bash
python trim_tissue.py \
    --csv='' \
    --slide-dir='' \
    --save-dir='' \
```

### 2. Crop tissue from the slide 

*Time: ~8 seconds per slide*

This will extract the tissue from the slides and save it in a single .jpg image. Takes roughly 8 seconds per slide. Can be run on a CPU only job.

You have to look up what your spacing is at each level.
```bash
# docker:
./c-submit --require-mem="32g" --require-cpus="8" --priority="low" <name> <ticket> 48 oni:11500/johnmelle/pathology_base_cuda_10:3
# command:
python3.7 -u <path to repo>extract_tissue.py --num_processes 8 --whole_tissue --save_level=2
```

3. Packing the slides into one .tar archive for convenience
```bash
# you can run this on a cluster node in a tmux-session
tmux
cd $path_to_repo
tar cf tissue.tar.gz tissue
```
### IMPORTANT
In all the following commands `nproc_per_node` and `batch_size` should be equal to number of GPUs you use for each task (since batches are distributed over GPUs). If you want to change the effective minibatch-size change `accumulate_batch`. The effective minibatch size is `batch_size` * `accumulate_batch`.

### 3. Extract features from whole slides, using imagenet weights

*Time: roughly one day of extraction*

1. Use train.py to extract features (epochs here means how many augmented samples you want to encode)

```bash
# 2 gpu
./c-submit \
    --gpu-count=2 \
    --require-mem="64g" \
    --require-cpus="22" \
    --constraint='Turing' \
    --priority="high" \
    <name> <ticket> 96 oni:11500/johnmelle/pathology_base_cuda_10:3
```

```bash
OMP_NUM_THREADS=4 python3.7 -m torch.distributed.launch --nproc_per_node=2 train.py \
    --name=features \
    --epochs=3 \
    --batch_size=2 \
    --image_size=16384 \
    --extract_features \
    --num_workers=2 \
    --data_tar=$path_to_repo + tissue.tar
    --save_path=$path_to_repo + features.tar
```

### 4. Train only last part of network using saved features

*Time: roughly one day of training*

```bash
# 2 gpu
./c-submit \
    --gpu-count=2 \
    --require-mem="64g" \
    --require-cpus="22" \
    --constraint='Turing' \
    --priority="high" \
    <name> <ticket> 96 oni:11500/johnmelle/pathology_base_cuda_10:3
```

```bash
OMP_NUM_THREADS=4 python3.7  -m torch.distributed.launch --nproc_per_node=2 train.py \
    --name=train-with-features \
    --epochs=100 \
    --lr=1e-5 \
    --batch_size=2 \
    --accumulate_batch=8 \
    --image_size=16384 \
    --train_with_features \
    --num_workers=4 \
    --data_tar=$path_to_repo + features.tar.gz
```

### 5. Finetune whole network with whole tissue 

*Time: multiple days of finetuning*

Estimates:

```
- spacing 2.0: 20 seconds per slide
- spacing 1.0: 1 minute per slide
```

If you have 500 slides:
```
- 1 GPU  = ~9 hours per epoch
- 2 GPUs = ~4:30 hours per epoch
- 4 GPUs = ~2:15 hours per epoch
- 8 GPUs = ~1:07 hours per epoch --> rarely possible on cluster
```
Schedule command
```bash
# 2 gpu
./c-submit \
    --gpu-count=2 \
    --require-mem="64g" \
    --require-cpus="22" \
    --constraint='Turing' \
    --priority="high" \
    <name> <ticket> 96 oni:11500/johnmelle/pathology_base_cuda_10:3

# 4 gpu
./c-submit \
    --gpu-count=4 \
    --require-mem="128g" \
    --require-cpus="44" \
    --constraint='Turing' \
    --priority="high" \
    <name> <ticket> 96 oni:11500/johnmelle/pathology_base_cuda_10:3
```
Python command
```bash
OMP_NUM_THREADS=4 python3.7 -m torch.distributed.launch --nproc_per_node=2 train.py \
    --name=finetune-16384 \
    --resume_name=train-with-features-sgd-momentum \
    --resume_epoch=35 \
    --image_size=16384 \
    --epochs=100 \
    --lr=1e-5 \
    --batch_size=2 \
    --accumulate_batch=8 \
    --image_size=16384 \
    --num_workers=1 \
    --train_from_transfer 
    --data_tar=$path_to_repo + tissue.tar.gz
```
