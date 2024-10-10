"""Automate downloading, preprocessing, and uploading images"""
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional
import json
import shutil
import subprocess
import tempfile


@dataclass
class DatasetMetadata:
    """Description of a preprocessed image dataset."""

    resolution: int
    tag: Optional[str]

    def __str__(self):
        return f"{self.resolution}x{self.resolution}" + (
            f"-{self.tag}" if self.tag is not None else ""
        )


def group_to_chunks(tarinfos, chunk_size):
    """Give a list of dicts from rclone's "lsjson" command, group them into chunks such that each
    chunk is less than chunk_size bytes."""
    out = []
    while len(tarinfos) > 0:
        this_chunk = []
        this_chunk_size = 0
        while True:
            if len(tarinfos) == 0 or this_chunk_size + tarinfos[0]["Size"] > chunk_size:
                break
            else:
                this_chunk.append(tarinfos[0])
                this_chunk_size += tarinfos[0]["Size"]
                tarinfos = tarinfos[1:]
        out.append(this_chunk)
    return out


def get_file_info(path: str):
    """Get information about files in a given directory from rclone's "lsjson" command."""
    return json.loads(
        subprocess.check_output(
            [
                "rclone",
                "lsjson",
                "-R",
                "--files-only",
                "--no-modtime",
                "--no-mimetype",
                "--fast-list",
                "-vvv",
                f"r2:txt2img-unsupervised-dataset/{path}",
            ]
        )
    )


def get_preprocessed_parquets(metadata: DatasetMetadata):
    """Get a list of parquet files that have been preprocessed for a given resolution."""
    ret = get_file_info(f"preprocessed/{metadata}")
    for file in ret:
        assert file["Name"].endswith(
            ".parquet"
        ), "Got a non-parquet file in preprocessed directory"
    return ret


def get_unprocessed_tarballs(metadata: DatasetMetadata) -> List[dict]:
    """Get a list of tarballs that have not been preprocessed to a given resolution."""
    original_tarballs = get_file_info(f"original-tarballs")
    existing_parquets = get_preprocessed_parquets(metadata)

    already_processed = {pq["Name"] for pq in existing_parquets}

    out = []
    for tarball in original_tarballs:
        name = tarball["Name"]
        if name.startswith("reddit_"):
            print(f"Skipping {name} because it's a reddit tarball")
            continue
        assert name.endswith(
            ".tar"
        ), "Got a non-tarball file in original-tarballs directory"
        name = name[:-4]
        has_stills = f"{name}-deduped.parquet" in already_processed
        has_video_stills = f"{name}-video_stills.parquet" in already_processed
        if not (has_stills or has_video_stills):
            out.append(tarball)

    return out


def download_files(files: list[dict], destdir: Path) -> None:
    """Download a list of files using rclone."""
    with tempfile.NamedTemporaryFile(mode="w") as files_from:
        for file in files:
            files_from.write(f"{file['Path']}\n")
        files_from.flush()
        print(f"Wrote files-from to {files_from.name}")
        subprocess.check_call(
            [
                "rclone",
                "copy",
                "--progress",
                "--no-traverse",
                "--transfers",
                "16",
                "--multi-thread-streams",
                "16",
                "--files-from",
                files_from.name,
                "r2:txt2img-unsupervised-dataset/original-tarballs",
                str(destdir),
            ]
        )


def untar_files(tarballs: list[Path]) -> None:
    """Untar a list of tarballs."""
    for tarball in tqdm(tarballs, desc="untarring"):
        subprocess.check_call(["tar", "-xf", str(tarball), "-C", str(tarball.parent)])


def get_dirs(path: Path):
    still_dirs = path.glob("*/deduped")
    video_still_dirs = path.glob("*/video_stills")
    return list(still_dirs) + list(video_still_dirs)


def assert_all_same_parent(paths: list[Path]) -> None:
    parent = paths[0].parent
    for path in paths:
        assert (
            path.parent == parent
        ), f"Expected all paths to have the same parent ({parent}), but {path} does not."


def preprocess_images(dirs: list[Path], res: int, batch_size: int) -> list[Path]:
    assert_all_same_parent([dir.parent for dir in dirs])

    cmd = [
        "python",
        "preprocess_images.py",
        "--batch-size",
        str(batch_size),
        "--res",
        str(res),
        "--ckpt",
        "vq-f4.ckpt",
        "--autoencoder-cfg",
        "vq-f4-cfg.yaml",
        "--random-crop",
    ] + [str(d) for d in dirs]
    print(f"Running {' '.join(cmd)}")
    subprocess.check_call(cmd)

    pqs = [p.parent.with_name(f"{p.parent.name}-{p.name}.parquet") for p in dirs]
    for pq in pqs:
        assert pq.exists(), f"Expected {pq} to exist after preprocessing"

    return pqs


def upload_pqs(pqs: list[Path], metadata: DatasetMetadata) -> None:
    parent = pqs[0].parent
    assert_all_same_parent(pqs)
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as files_from:
        for pq in pqs:
            files_from.write(f"{str(pq.relative_to(parent))}\n")
        files_from.flush()
        print(f"Wrote files-from to {files_from.name}")
        subprocess.check_call(
            [
                "rclone",
                "copy",
                "--progress",
                "--no-traverse",
                "--transfers",
                "16",
                "--multi-thread-streams",
                "16",
                "--files-from",
                files_from.name,
                str(pqs[0].parent),
                f"r2:txt2img-unsupervised-dataset/preprocessed/{metadata}",
            ]
        )


def process_tars(
    tars: list[dict], metadata: DatasetMetadata, batch_size: int, workdir: Path
) -> None:
    """Download, preprocess, and upload a list of tarballs (as described by the output of rclone
    lsjson)."""

    assert len(list(workdir.iterdir())) == 0, f"Expected {workdir} to be empty"

    print("Downloading tarballs")
    download_files(tars, workdir)

    untar_files([workdir / t["Name"] for t in tars])
    src_dirs = get_dirs(workdir)
    print(f"Got {len(src_dirs)} directories of images: {src_dirs}")

    print("Preprocessing images")
    pqs = preprocess_images(src_dirs, metadata.resolution, batch_size)

    print("Uploading parquet files")
    upload_pqs(pqs, metadata)

    print("Cleaning up")
    for p in workdir.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
        elif p.is_file():
            p.unlink()


# NOTE: randomize which tarballs we download
