"""Merge n parquet files containing cap-image pairs, outputting a single file containing one row
for each image with all the caps."""

import argparse
import numpy as np
import tempfile

from datasets import Dataset
from pathlib import Path
from tqdm import tqdm, trange

from txt2img_unsupervised.load_pq_dir import load_pq_dir


def merge_dsets(dsets, out_dir, out_chunk_size, cap_count):
    # Given a list of datasets containing cap-image pairs, iterate over the datasets in lockstep,
    # associating rows with the same name and merging the caps into a single row, then write the
    # merged rows to a series of parquet files.

    # The lockstep iteration needs the datasets to be sorted so the images appear in the same order.
    # For some reason sorting resets the format :/
    dsets = [dset.sort(column_names="name").with_format("numpy") for dset in dsets]

    dset_idxs = [0] * len(dsets)  # The index of the next row to read from each dataset
    out_buf = []
    rows_outputted_cnt = 0
    unmatched_rows_cnt = 0
    pqs_written_cnt = 0

    with tqdm(total=sum(len(dset) for dset in dsets), unit="input rows") as pbar:

        def update_pbar_postfix():
            pbar.set_postfix(
                {
                    "rows_outputted": rows_outputted_cnt,
                    "unmatched_rows": unmatched_rows_cnt,
                }
            )

        while any(dset_idxs[i] < len(dset) for i, dset in enumerate(dsets)):
            # Find the earliest name at the current indices
            names = [
                dset[dset_idxs[i]]["name"]
                for i, dset in enumerate(dsets)
                if dset_idxs[i] < len(dset)
            ]
            earliest_name = min(names)
            # Find the indices of the datasets with that name
            matching_idxs = [
                i
                for i, dset in enumerate(dsets)
                if dset_idxs[i] < len(dset)
                and dset[dset_idxs[i]]["name"] == earliest_name
            ]
            match_cnt = len(matching_idxs)
            # Check that the number of matching rows is equal to the cap count
            if match_cnt != cap_count:
                for i in matching_idxs:
                    dset_idxs[i] += 1
                unmatched_rows_cnt += match_cnt
                update_pbar_postfix()
                pbar.update(match_cnt)
                continue
            # Create the output row from the matches
            keys = dsets[matching_idxs[0]].column_names
            assert "cap_center" in keys, "All datasets must have a 'cap_center' column"
            assert (
                "cap_max_cos_distance" in keys
            ), "All datasets must have a 'cap_max_cos_distance' column"
            for i in matching_idxs[1:]:
                assert (
                    dsets[i].column_names == keys
                ), "All datasets must have the same schema"
            # Parquet doesn't support multi-dimensional arrays so we need to flatten the caps
            out_cap_centers = [
                dsets[i][dset_idxs[i]]["cap_center"] for i in matching_idxs
            ]
            assert all(
                np.any(cap_center != out_cap_centers[0])
                for cap_center in out_cap_centers[1:]
            ), "cap centers are identical :("
            out_cap_centers = np.concatenate(out_cap_centers)
            assert out_cap_centers.shape == (
                cap_count * 768,
            ), f"out_cap_centers shape {out_cap_centers.shape}"
            assert (
                out_cap_centers.dtype == np.float32
            ), f"out_cap_centers dtype {out_cap_centers.dtype}"

            out_cap_max_cos_distances = [
                dsets[i][dset_idxs[i]]["cap_max_cos_distance"] for i in matching_idxs
            ]
            assert all(
                np.any(max_cos_distance != out_cap_max_cos_distances[0])
                for max_cos_distance in out_cap_max_cos_distances[1:]
            ), "cap max cos distances are identical :("
            assert all(
                max_dist.dtype == np.float32 for max_dist in out_cap_max_cos_distances
            )
            out_cap_max_cos_distances = np.array(out_cap_max_cos_distances)
            assert out_cap_max_cos_distances.shape == (
                cap_count,
            ), f"out_cap_max_cos_distances shape {out_cap_max_cos_distances.shape}"
            assert (
                out_cap_max_cos_distances.dtype == np.float32
            ), f"out_cap_max_cos_distances dtype {out_cap_max_cos_distances.dtype}"

            # Check that everything except the caps is equal across the matches
            keys_to_copy = set(keys) - {"cap_center", "cap_max_cos_distance"}
            for k in keys_to_copy:
                vals = [dsets[i][dset_idxs[i]][k] for i in matching_idxs]
                assert all(
                    np.array_equal(vals[0], v) for v in vals
                ), f"Values for key {k} do not match across matches"

            # Create the output row
            first_matching_row = dsets[matching_idxs[0]][dset_idxs[matching_idxs[0]]]
            out_row = {k: first_matching_row[k] for k in keys_to_copy}
            out_row["cap_center"] = out_cap_centers
            out_row["cap_max_cos_distance"] = out_cap_max_cos_distances

            out_buf.append(out_row)
            rows_outputted_cnt += 1
            pbar.update(match_cnt)
            update_pbar_postfix()
            for i in matching_idxs:
                dset_idxs[i] += 1

            # Flush if neccessary
            if len(out_buf) >= out_chunk_size or all(
                dset_idxs[i] == len(dset) for i, dset in enumerate(dsets)
            ):
                out_dict = {k: np.stack([row[k] for row in out_buf]) for k in keys}
                out_dset = Dataset.from_dict(out_dict)
                out_path = out_dir / f"merged_capexamples_{pqs_written_cnt:06d}.parquet"
                tqdm.write(f"Writing {len(out_buf)} rows to {out_path}")
                out_dset.to_parquet(out_path, compression="zstd")
                out_buf = []
                pqs_written_cnt += 1


def _test_merge_n_dsets(n):
    # run get-test-data.sh to download the test data
    test_data_dir = Path(__file__).parent / "test-images/capped-examples"
    in_dsets = [
        Dataset.from_parquet(str(test_data_dir / f"test-caps-{i}.parquet"))
        for i in range(n)
    ]
    assert all(len(dset) == len(in_dsets[0]) for dset in in_dsets[1:])

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        merge_dsets(in_dsets, out_dir, 8192, n)
        out_dset = load_pq_dir(out_dir)
        print(f"Loaded merged dataset with {len(out_dset)} rows")

    # Generating caps from n images results in n - 2 cap-image pairs, so our output should have
    # n minus 2 times as many inputs datasets as there are. Unless some of the two missing pairs in
    # the input datasets overlap, which is extremely unlikely and doesn't happen in our test data.
    assert len(out_dset) == len(in_dsets[0]) - (2 * (n - 1))

    # Iterate over the rows in the merged dataset, checking that everything was copied correctly
    in_dsets = [
        dset.sort(column_names="name").with_format("numpy") for dset in in_dsets
    ]
    in_idxs = [0] * len(in_dsets)

    for i in trange(len(out_dset)):
        # advance in_idxs until all the input rows match the name of the output row
        out_name = out_dset[i]["name"]
        for j in range(n):
            assert in_idxs[j] < len(in_dsets[j])
            while in_dsets[j][in_idxs[j]]["name"] != out_name:
                in_idxs[j] += 1
                assert in_idxs[j] < len(in_dsets[j])

        # Check that the caps were copied correctly
        out_cap_centers = out_dset[i]["cap_center"]
        assert out_cap_centers.shape == (n * 768,)
        assert out_cap_centers.dtype == np.float32
        expected_out_caps = np.concatenate(
            [in_dsets[j][in_idxs[j]]["cap_center"] for j in range(n)]
        )
        np.testing.assert_array_equal(out_cap_centers, expected_out_caps)

        out_cap_max_cos_distances = out_dset[i]["cap_max_cos_distance"]
        assert out_cap_max_cos_distances.shape == (n,)
        assert out_cap_max_cos_distances.dtype == np.float32
        expected_out_caps = np.array(
            [in_dsets[j][in_idxs[j]]["cap_max_cos_distance"] for j in range(n)]
        )
        np.testing.assert_array_equal(out_cap_max_cos_distances, expected_out_caps)

        # Check that the other columns were copied correctly
        common_keys = set(in_dsets[0].column_names) - {
            "cap_center",
            "cap_max_cos_distance",
        }
        for j in range(n):
            for k in common_keys:
                np.testing.assert_array_equal(
                    out_dset[i][k], in_dsets[j][in_idxs[j]][k]
                )


def test_merge_2_dsets():
    _test_merge_n_dsets(2)


def test_merge_3_dsets():
    _test_merge_n_dsets(3)


def test_merging_same_caps_fails():
    dset_path = (
        Path(__file__).parent / "test-images/capped-examples/test-caps-0.parquet"
    )
    dset = Dataset.from_parquet(str(dset_path))

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        try:
            merge_dsets([dset, dset], out_dir, 8192, 2)
        except AssertionError as e:
            return
        assert False, "Merging the same dataset twice should raise an AssertionError"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing files to merge",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--out-chunk-size", type=int, default=8192)
    parser.add_argument(
        "--cap-count",
        type=int,
        required=True,
        help="Target number of caps per image (images with more or fewer caps will be discarded)",
    )
    args = parser.parse_args()

    pq_paths = [str(p) for p in sorted(args.input_dir.glob("**/*.parquet"))]
    print(f"Found {len(pq_paths)} parquet files in {args.input_dir}")

    dsets = [Dataset.from_parquet(path).with_format("numpy") for path in pq_paths]
    print(
        f"Loaded {len(dsets)} datasets with {sum(len(dset) for dset in dsets)} total rows"
    )
    assert all(
        len(dset) > 0 for dset in dsets
    ), "All datasets must have at least one row"

    merge_dsets(dsets, args.output_dir, args.out_chunk_size, args.cap_count)


if __name__ == "__main__":
    main()
