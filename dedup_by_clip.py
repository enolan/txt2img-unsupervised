# Deduplicate by approximate CLIP embedding. This is pretty fast, but requires storing the full set
# of CLIP embeddings (quantized down to 16 bits per component/1,546 bytes per embedding) in memory.
# If that becomes a problem this could be adapted to do it with sqlite or something.
import argparse
import json
import numpy as np

from pathlib import Path
from tqdm import tqdm

from txt2img_unsupervised.load_pq_dir import load_pq_dir_to_infinidata


def gen_dup_list(dset):
    """Given a dataset (as a TableView), generate a list of lists of the names of images with
    approximately duplicate CLIP embeddings. This is done by quantizing the components to 16 bits.
    """
    batch_size = 16384

    clips_dict = {}
    for batch_idx, batch in enumerate(
        tqdm(
            dset.batch_iter(
                batch_size=batch_size, drop_last_batch=False, threads=8, readahead=8
            ),
            total=len(dset) // batch_size + (1 if len(dset) % batch_size != 0 else 0),
        )
    ):
        clips = batch["clip_embedding"]
        nans = np.any(np.isnan(clips), axis=1)
        assert nans.shape == (len(clips),)
        if np.any(nans):
            print(
                f"Found images with NaN in their CLIP embeddings: {batch['name'][nans]}"
            )

        keys = [
            vec.tobytes() if ~nans[i] else "NaN"
            for i, vec in enumerate(
                np.clip(clips * 2**15, -(2**15), 2**15 - 1).astype(np.int16)
            )
        ]

        for i, key in enumerate(keys):
            clips_dict.setdefault(key, []).append(batch["name"][i])

    print(f"Found {len(clips_dict)} unique CLIP embeddings out of {len(dset)} images")

    dups = [names for names in clips_dict.values() if len(names) > 1]
    return dups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pq-dir", type=Path, help="Directory containing parquet files", required=True
    )
    parser.add_argument(
        "--out", type=Path, help="Path to write the duplicate list to", required=True
    )

    args = parser.parse_args()

    dset = load_pq_dir_to_infinidata(args.pq_dir).select_columns(
        {"name", "clip_embedding"}
    )
    dups_list = gen_dup_list(dset)

    with open(args.out, "w") as f:
        json.dump(dups_list, f)


if __name__ == "__main__":
    main()
