from datasets import Dataset
from pathlib import Path
from tqdm import tqdm
from xdg_base_dirs import xdg_cache_home

import hashlib
import infinidata
import numpy as np
import pyarrow.parquet as pq


def load_pq_dir(dir_path):
    pq_paths = [str(p) for p in sorted(dir_path.glob("**/*.parquet"))]
    return Dataset.from_parquet(pq_paths).with_format("numpy")


def load_pq_dir_to_infinidata(dir_path):
    """Load a directory of parquet files into a single TableView."""
    paths = sorted(dir_path.glob("**/*.parquet"))
    tvs = []

    for path in tqdm(paths, unit="parquet files"):
        tv = load_pq_to_infinidata(path)
        if tv is not None:
            tvs.append(tv)
    print(f"Loaded dataset into {len(tvs)} TableViews")
    out = infinidata.TableView.concat(tvs)
    print(f"Concatenated into one TableView with {len(out)} rows")
    return out


def load_pq_to_infinidata(path):
    """Load a parquet file into a list of TableViews."""
    # Our caching scheme is to use the hash of the path and the mtime of the file
    path = path.resolve()
    path_hash = hashlib.sha256(str(path).encode()).hexdigest()

    cache_dir = xdg_cache_home() / "txt2img_unsupervised-parquets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"{path_hash}-pq.infinidata"
    if cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
        return infinidata.TableView.load_from_disk(cache_dir, cache_path.name)
    else:
        pq_file = pq.ParquetFile(path)
        tvs = []
        with tqdm(unit="rows", total=pq_file.metadata.num_rows, leave=False) as pbar:
            for batch in pq_file.iter_batches():
                batch_df = batch.to_pandas()
                batch_dict = {
                    col: np.stack(batch_df[col].to_numpy()) for col in batch_df.columns
                }
                tvs.append(infinidata.TableView(batch_dict))
                pbar.update(len(tvs[-1]))
        if len(tvs) == 0:
            return None
        else:
            tvs_rows = sum(map(len, tvs))
            assert tvs_rows == pq_file.metadata.num_rows
            tv = infinidata.TableView.concat(tvs)
            assert len(tv) == pq_file.metadata.num_rows
            tv.save_to_disk(cache_dir, cache_path.name)
            return tv
