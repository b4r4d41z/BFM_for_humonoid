# Stating Offline training | Behavioral Cloning

## Recommended NAS HDF5 path

Use the CIFS mount for NAS training:

```bash
NAS_H5="/mnt/tank6124_sharefolders/Datasets/BFM_dataset/hdf5"
```

Avoid using `/run/user/1000/gvfs/...` as the recommended HDF5 path; GVFS SMB can be slow and unstable for DataLoader/HDF5 workloads.


1. inside terminal launch tensorboard and training

```bash
tensorboard --logdir runs/bc --host 0.0.0.0 --port 6006 &

PYTHONPATH=$PWD python scripts/bc/train.py \
  --data "/media/lab/New Volume/hdf5/Sorting_food" \
  --updates 5000 \
  --batch_size 32 \
  --model_device cuda:0 \
  --buffer_device cpu \
  --tensorboard \
  --save_path checkpoints/bc/sorting_food_exp1
```

# Testing trained policies from checkpoint | Behavioral Cloning

1. To start choose certain policy inside checkpoints folder:

```bash
python -m scripts.bc.play_isaaclab \
  --task Template-Test-Training-Direct-v0 \
  --checkpoint /home/lab/Desktop/ivan/BFM_for_humonoid/checkpoints/bc/best.pt \
  --action_mode arm_plus_gripper_bridge \
  --allow_provisional_mapping \
  --gripper_open_prototype 0 100 0 0 0 0 \
  --gripper_closed_prototype 69 99 42 44 61 60 \
  --gripper_open_threshold 0.35 \
  --gripper_close_threshold 0.65 \
  --max_steps 300 \
  --render \
  --debug
```

# Stating Isaacc_Lab env

from ` /.../BFM_for_humonoid/Test_training/source/Test_training `

```bash
python -m pip install -e .
```

start env with agent from ` /.../BFM_for_humonoid/Test_training `

```bash 
python scripts/zero_agent.py --task Template-Test-Training-Direct-v0 --max_steps 5000 --log_every 200 --watchdog_sec 10
```

It creates the Gym environment for the given task ID and steps it with zero (or fixed) actions to quickly verify that the scene/environment initializes correctly and the simulation starts.

---

```bash 
python scripts/rsl_rl/train.py --task Template-Test-Training-Direct-v0 --num_envs 1 --seed 0
```

**Tests for modificcation**

>   buffers "smoke" test

```bash 
python Test_training/scripts/bfm/check_buffer.py \
  --data "/media/lab/New Volume/hdf5/Sorting_food" \
  --max_files 2 \
  --batch_size 4 \
  --seq_batch_size 2 \
  --seq_len 8
``` 