import sys
from pathlib import Path

from torch.utils.data import DataLoader

PROJECT_ROOT = Path("/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training")
sys.path.append(str(PROJECT_ROOT))

from learning.bfm.data.stream_loader import HDF5DataStreamLoader
from learning.bfm.data.batch_assembly import assemble_bfm_batch


hdf5_dir = Path("/media/lab/New Volume/simulation_data_collection/hdf5")
files = sorted(hdf5_dir.glob("*.h5"))

if not files:
    raise FileNotFoundError(f"No .h5 files found in: {hdf5_dir}")

for file_path in files:
    print("\nChecking:", file_path)

    dataset = HDF5DataStreamLoader(str(file_path))
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=assemble_bfm_batch,
    )

    sample = dataset[0]
    print("sample obs full:", sample["obs"]["state"]["full"].shape)
    print("sample action full:", sample["action"]["full"].shape)

    batch = next(iter(loader))
    print("batch obs full:", batch["obs"]["state"]["full"].shape)
    print("batch action full:", batch["action"]["full"].shape)