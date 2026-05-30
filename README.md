# 🧠 Behavior Foundation Model for humonoid robots

This project presents a development framework for training a Behavior Foundation Model (BFM), including core modules, training pipeline, and a vision-language-conditioned goal interface. It also includes sample model definitions for agent behavior and latent representation learning.

![Training Workflow](./assets/TrainigWorkflow.jpg)


# 📝 Updates
> V2 27 Apr. 2026 
 - Offline trainir, Immitation learning like 


# 🎓  Tutorial for training
To train model on your own data use this [tutoral](tutorial.md)

# 📊 Dataset used in project 

https://huggingface.co/datasets/b4r4d41z/Bimanual-humanid-tasks


# Offline BC training on NAS HDF5 datasets

For HDF5 training over the NAS, use the kernel CIFS mount (or a local SSD) rather than the Ubuntu file-manager GVFS SMB path. The recommended dataset root is:

```bash
NAS_H5="/mnt/tank6124_sharefolders/Datasets/BFM_dataset/hdf5"
```

If a path starts with `/run/user/1000/gvfs/...`, the training and inspection tools print:

```text
You are using a GVFS SMB path. For HDF5 training, CIFS mount or local SSD is recommended.
```

The offline BC trainer now supports two loading modes:

- `--data_loading eager` keeps the legacy behavior and preloads selected HDF5 files into `OfflineTrajectoryBuffer` after the train/validation file split.
- `--data_loading streaming` builds only a lightweight `(file, timestep)` index and reads samples lazily through `torch.utils.data.DataLoader` workers. This is recommended for RGB HDF5 datasets on NAS.

## Inspect and benchmark the dataset

```bash
PYTHONPATH=$PWD python scripts/data/inspect_hdf5_dataset.py \
  --data "$NAS_H5" \
  --recursive \
  --max_files 73

PYTHONPATH=$PWD python scripts/bc/benchmark_loader.py \
  --data "$NAS_H5" \
  --recursive \
  --max_files 10 \
  --mode both \
  --num_workers 0,4,8 \
  --use_images true \
  --cameras head,left_wrist,right_wrist \
  --image_size 224 \
  --batch_size 16 \
  --batches 50 \
  --pin_memory \
  --persistent_workers \
  --prefetch_factor 2
```

## State-only debug training

`--use_images false` avoids reading image tensors from HDF5. The current vision policy still expects image keys, so the streaming dataset supplies zero dummy images for model compatibility.

```bash
NAS_H5="/mnt/tank6124_sharefolders/Datasets/BFM_dataset/hdf5"

PYTHONPATH=$PWD python scripts/bc/train.py \
  --data "$NAS_H5" \
  --recursive \
  --max_files 10 \
  --data_loading streaming \
  --use_images false \
  --updates 1000 \
  --batch_size 256 \
  --model_device cuda:0 \
  --num_workers 4 \
  --save_path checkpoints/bc/state_only_debug.pt
```

## One-camera training

```bash
NAS_H5="/mnt/tank6124_sharefolders/Datasets/BFM_dataset/hdf5"

PYTHONPATH=$PWD python scripts/bc/train.py \
  --data "$NAS_H5" \
  --recursive \
  --max_files 10 \
  --data_loading streaming \
  --use_images true \
  --cameras head \
  --image_size 224 \
  --updates 1000 \
  --batch_size 32 \
  --model_device cuda:0 \
  --num_workers 8 \
  --pin_memory \
  --persistent_workers \
  --prefetch_factor 2 \
  --amp \
  --amp_dtype bf16 \
  --save_path checkpoints/bc/head_camera_debug.pt
```

## Full three-camera training

```bash
NAS_H5="/mnt/tank6124_sharefolders/Datasets/BFM_dataset/hdf5"

PYTHONPATH=$PWD python scripts/bc/train.py \
  --data "$NAS_H5" \
  --recursive \
  --max_files 73 \
  --data_loading streaming \
  --use_images true \
  --cameras head,left_wrist,right_wrist \
  --image_size 224 \
  --updates 5000 \
  --batch_size 16 \
  --model_device cuda:0 \
  --num_workers 8 \
  --pin_memory \
  --persistent_workers \
  --prefetch_factor 2 \
  --amp \
  --amp_dtype bf16 \
  --tensorboard \
  --save_path checkpoints/bc/exo_and_external_streaming.pt
```
