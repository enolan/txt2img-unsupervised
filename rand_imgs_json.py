"""Generate a json file with a random sample of images from the dataset. For website demo."""

import argparse
import json
import shutil
import sqlite3
import subprocess
import tempfile

from pathlib import Path
from tqdm import tqdm


def get_random_images(db_path, num_images):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    imgs = conn.execute(
        "SELECT warc, id, extension FROM files WHERE processed = 1 ORDER BY random() LIMIT ?",
        (num_images,),
    ).fetchall()
    conn.close()
    return imgs


def get_warcs(warcs):
    with tempfile.NamedTemporaryFile(mode="w") as files_from:
        for warc in warcs:
            files_from.write(f"{warc}.tar\n")
        files_from.flush()
        subprocess.run(
            [
                "rclone",
                "-P",
                "copy",
                "--no-traverse",
                "--transfers",
                "16",
                "--multi-thread-streams",
                "16",
                "--files-from",
                files_from.name,
                f"r2:txt2img-unsupervised-dataset/original-tarballs",
                "warcs",
            ],
            check=True,
        )
    for warc in warcs:
        subprocess.run(["tar", "-C", "warcs", "-xf", f"warcs/{warc}.tar"], check=True)
        Path(f"warcs/{warc}.tar").unlink()


def get_image_files(img_metadata, outdir, batch_size):
    warc_imgs = {}
    for i, (warc, _, _) in enumerate(img_metadata):
        warc_imgs.setdefault(warc, []).append(img_metadata[i])
    print(f"Downloading {len(img_metadata)} images from {len(warc_imgs)} warcs")
    with tqdm(total=len(warc_imgs)) as pbar:
        while len(warc_imgs) > 0:
            warcs_this_batch = list(warc_imgs.keys())[:batch_size]
            batch = {k: warc_imgs[k] for k in warcs_this_batch}
            print(f"Batch: {batch}")
            get_warcs(list(batch.keys()))
            for warc, imgs in batch.items():
                for warc, id, extension in imgs:
                    candidates = list(Path("warcs").glob(f"{warc}/**/{id}{extension}"))
                    if len(candidates) > 0:
                        if len(candidates) > 1:
                            print(
                                f"Found multiple candidates for {warc}/{id}{extension} 🤷 {candidates}"
                            )
                        candidates[0].rename(outdir / f"{id}{extension}")
                    else:
                        print(f"Could not find {warc}/{id}{extension} 😢")
                        return
                pbar.update(1)
            for warc in batch.keys():
                shutil.rmtree(f"warcs/{warc}")
                del warc_imgs[warc]


def label_images(imgs, out):
    """Label each image as NSFW or not"""
    for img in imgs:
        if img.suffix == ".mp4":
            subprocess.run(["vlc", str(img)], check=True)
        else:
            subprocess.run(["feh", "-FZ", img], check=True)
        label = input("Is this image NSFW? (y/n) ")
        out.append(label == "y")
    assert len(out) == len(imgs)


def format_as_json(url_prefix, path, nsfw):
    stat = path.stat()
    return {
        "url": f"{url_prefix}/{str(path)}",
        "nsfw": nsfw,
        "timestamp": int(stat.st_mtime),
    }


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    parser.add_argument(
        "--db", type=str, required=True, help="Path to the sqlite3 database"
    )
    parser.add_argument(
        "--num-images", type=int, required=True, help="Number of images to select"
    )
    parser.add_argument(
        "--path-prefix", type=str, required=True, help="Prefix to add to the image URLs"
    )
    parser.add_argument(
        "--out", type=str, required=True, help="Path to write the output JSON to"
    )
    args = parser.parse_args()

    imgs_metadata = get_random_images(args.db, args.num_images)
    imgs_paths = get_image_files(imgs_metadata)
    labels = label_images(imgs_paths)
    img_objs = [
        format_as_json(args.path_prefix, path, nsfw)
        for path, nsfw in zip(imgs_paths, labels)
    ]

    with open(args.out, "w") as f:
        json.dump(img_objs, f, indent=2)


if __name__ == "__main__":
    main()
