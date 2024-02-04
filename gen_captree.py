import argparse
import datetime
import json
import numpy as np

from pathlib import Path

from txt2img_unsupervised.spherical_space_partitioning import (
    CapTree,
    load_pq_dir_to_infinidata,
)


def main():
    parser = argparse.ArgumentParser(
        description="Build a captree from a set of txt2img-unsupervised parquet files."
    )
    parser.add_argument("--pq-dir", type=Path, required=True)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--outlier-removal-level", type=float, default=1)
    parser.add_argument("--max-leaf-size", type=int, default=None)
    parser.add_argument("--k-means-iters", type=int, default=200)
    parser.add_argument("--summary-file", type=Path, default=None)
    parser.add_argument("--write-dup-blacklist", type=Path, default=None)
    parser.add_argument("--read-dup-blacklist", type=Path, default=None)
    parser.add_argument("--paranoid", action="store_true")
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--thin", action="store_true")
    args = parser.parse_args()

    if args.save_dir.exists():
        print(f"Save dir {args.save_dir} exists, exiting")
        exit(1)

    get_timestamp = lambda: datetime.utcnow().isoformat()
    print(f"Time at start: {get_timestamp()}")

    dset_all = load_pq_dir_to_infinidata(args.pq_dir).shuffle(seed=19900515)
    print(f"Loaded dataset with {len(dset_all)} rows")
    dset = dset_all.new_view(slice(0, int(len(dset_all) * 0.99)))
    print(f"Train set size: {len(dset)}")
    print(f"Time after split: {get_timestamp()}")

    if args.read_dup_blacklist is not None:
        with open(args.read_dup_blacklist, "r") as f:
            found_duplicates = json.load(f)
        blacklist = [
            item
            for sublist in (names[1:] for names in found_duplicates)
            for item in sublist
        ]
        print(f"Found {len(blacklist)} blacklisted images")
        pre_count = len(dset)
        blacklist = set(blacklist)
        dset = dset.remove_matching_strings("name", blacklist)
        print(f"Removed {pre_count - len(dset)} blacklisted images")
        print(f"Time after blacklist: {get_timestamp()}")
    if args.subset is not None:
        dset = dset.new_view(slice(args.subset))

    print(f"Time after subset/before building tree: {get_timestamp()}")

    tree = CapTree(
        dset=dset,
        batch_size=args.batch_size,
        k=args.k,
        outlier_removal_level=args.outlier_removal_level,
        max_leaf_size=args.max_leaf_size,
        iters=args.k_means_iters,
        dup_check=True if args.write_dup_blacklist is not None else False,
    )
    tree.split_rec()

    print(f"Time after building tree: {get_timestamp()}")

    tree.shuffle_leaves()

    print(f"Time after shuffling: {get_timestamp()}")

    if args.paranoid:
        # This is a pretty slow check, but I don't 100% trust the code
        tree._check_invariants()

    # minimum possible depth, given branching factor is k and leaves can have at most k^2 vectors
    min_depth = int(np.ceil(np.log(len(dset)) / np.log(tree.max_leaf_size)))
    print(f"Tree depth: {tree.depth()}, minimum possible depth: {min_depth}")

    if args.summary_file is not None:
        args.summary_file.write_text(json.dumps(tree.to_summary(), indent=2))

    if len(tree.found_duplicates) > 0:
        dup_set_count = len(tree.found_duplicates)
        dup_total_count = sum(len(dup_set) for dup_set in tree.found_duplicates)
        if args.write_dup_blacklist is None:
            print(
                f"Found {dup_set_count} sets of duplicates, containing {dup_total_count} total images! You should rerun and generate a blacklist with --write-dup-blacklist!"
            )
            print("Counts will be wrong and sampling will be non-uniform!")
        else:
            print(
                f"Found {len(tree.found_duplicates)} sets of duplicates, containing {dup_total_count} total images. writing to {args.write_dup_blacklist}"
            )
            with open(args.write_dup_blacklist, "w") as f:
                json.dump(tree.found_duplicates, f, indent=2)

    print(f"Time at end: {get_timestamp()}")

    print(f"Saving to {args.save_dir}")
    tree.save_to_disk(args.save_dir, thin=args.thin)


if __name__ == "__main__":
    main()
