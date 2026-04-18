
import sys
import os
import numpy as np
import h5py

from pathlib import Path

from tqdm.auto import tqdm

from thermosim.utils.io import load_irb_txt



def write_dict_to_h5(group, data):
    for key, value in data.items():
        key = str(key)

        if isinstance(value, dict):
            subgrp = group.create_group(key)
            write_dict_to_h5(subgrp, value)

        elif isinstance(value, (int, float, np.integer, np.floating)):
            group.attrs[key] = value

        elif isinstance(value, str):
            group.attrs[key] = value

        elif isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value)
            group.create_dataset(key, data=arr)

        else:
            # fallback: store as string
            group.attrs[key] = str(value)



def write_thermograms(csv_files, out_file=None):
    if not csv_files:
        raise ValueError("csv_files is empty")

    csv_files = list(csv_files)

    # Infer output file
    if out_file is None:
        folder = Path(csv_files[0]).parent
        root   = folder.parent
        out_file = root / f"{folder.name}.h5"
    else:
        out_file = Path(out_file)

    with h5py.File(out_file, "w") as f:
        dset = None

        settings_grp = f.create_group("settings")
        params_grp   = f.create_group("params")

        for i, csv_file in enumerate(tqdm(csv_files)):
            arr, settings, params = load_irb_txt(csv_file)
            frame = arr.astype(np.float32)  # <-- FIX

            if dset is None:
                H, W = frame.shape

                dset = f.create_dataset(
                    "thermogram",
                    shape=(len(csv_files), H, W),
                    dtype=np.float32,
                    compression="gzip",
                    chunks=(1, H, W)
                )

                # write metadata once (assumed constant)
                write_dict_to_h5(settings_grp, settings)
                write_dict_to_h5(params_grp, params)


            else:
                # optional safety: ensure consistent shape
                if frame.shape != (H, W):
                    raise ValueError(f"Frame {i} has shape {frame.shape}, expected {(H, W)}")

            dset[i] = frame

    print(f"Wrote thermograms + metadata to: {out_file}")





def main(dir_path):
    dir_path = Path(dir_path)

    csv_files = sorted(dir_path.glob("*.csv"))

    print(f"found {len(csv_files)} in path {dir_path}")

       
    write_thermograms(csv_files)





if __name__ == "__main__":
    
    dir_path = "data/Wölbungskorrektur Messung 3"
    
    main(dir_path)