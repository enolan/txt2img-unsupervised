"""Process imgur archives from ArchiveTeam/Internet Archive"""
import CloseableQueue
import concurrent.futures
import hashlib
import imageio_ffmpeg  # type: ignore[import]
import internetarchive as ia
import os
import random
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
from CloseableQueue import CloseableQueue as CQueue
from pathlib import Path
from PIL import Image
from threading import Thread
from tqdm import tqdm
from typing import Tuple, Union


def setup_db(db_path: Path) -> sqlite3.Connection:
    """Set up a database to store metadata about the images and videos we've collected. Primarily
    for deduplication."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS warcs (
            id TEXT PRIMARY KEY NOT NULL,
            processed INTEGER NOT NULL,
            uploaded INTEGER NOT NULL
        ) STRICT"""
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
            id TEXT PRIMARY KEY NOT NULL,
            warc TEXT NOT NULL,
            extension TEXT NOT NULL,
            blake2b TEXT NOT NULL,
            processed INTEGER NOT NULL,
            FOREIGN KEY (warc) REFERENCES warcs (id)
        ) STRICT
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_file_blake2b ON files (blake2b)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_file_warc ON files (warc)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_file_blake2b_processed ON files (blake2b) WHERE processed = 1"
    )
    return conn


def fetch_warc_names(conn: sqlite3.Connection) -> None:
    """Fetch the names of all imgur warcs that exist on IA and add them to the DB."""
    warcs = list(ia.search_items("collection:archiveteam_imgur"))
    warcs = [w["identifier"] for w in warcs]
    with conn:
        conn.executemany(
            "INSERT OR IGNORE INTO warcs (id, processed, uploaded) VALUES (?, 0, 0)",
            [(w,) for w in warcs],
        )


def warc_id_to_url(id: str) -> str:
    """Convert a warc id to a url."""
    files = list(ia.get_files(id))
    warcs = [f for f in files if f.format == "Web ARChive ZST"]
    if len(warcs) == 1:
        return warcs[0].url
    elif len(warcs) > 0:
        tqdm.write(f"Found {len(warcs)} warcs for {id}??? Choosing biggest.")
        max_size = 0
        for warc in warcs:
            if warc.size > max_size:
                max_size = warc.size
                max_warc = warc
        return max_warc.url
    else:
        assert False, f"Found no warcs for {id}"


def download_warc(id: str, dest_dir: Path) -> Path:
    """Download a warc from the internet archive. This will make a new file under dest_dir
    named after the id. Returns the path to the compressed warc, which will be in dest_dir.
    """
    warc_url = warc_id_to_url(id)
    with open(dest_dir / f"{id}.arialog", "w") as f:
        subprocess.run(
            ["aria2c", "-s", "32", "-x", "16", warc_url],
            check=True,
            cwd=dest_dir,
            stdout=f,
            stderr=f,
        )
    filename_prefix = re.match(r"archiveteam_(imgur_.*)", id).group(1)
    warc_path = list(dest_dir.glob(f"{filename_prefix}*.warc.zst"))
    assert len(warc_path) == 1, f"Expected one warc, found {len(warc_path)}"
    return warc_path[0]


def decompress_and_extract_warc(compressed_warc_path: Path, dest_dir: Path) -> None:
    """Decompress and extract a WARC. After this, the dest_dir will contain a directory named
    i.imgur.com"""
    # Decompress using their weird zstd variant format.
    with tempfile.NamedTemporaryFile(dir=dest_dir, delete=False) as f:
        subprocess.run(["zstdwarccat", compressed_warc_path], stdout=f, check=True)
        decompressed_warc_path = Path(f.name)
    compressed_warc_path.unlink()
    # Extract the warc
    subprocess.run(
        [
            "python",
            "-m",
            "warcat",
            "--output-dir",
            dest_dir,
            "extract",
            decompressed_warc_path,
        ],
        check=True,
    )
    # Delete the warc
    decompressed_warc_path.unlink()
    # Delete the HTML files
    htmldir = dest_dir / "imgur.com"
    if htmldir.exists():
        shutil.rmtree(htmldir)
    else:
        tqdm.write(f"No html in {compressed_warc_path} 🤷")


def extract_from_dir(in_path: Path, out_path: Path) -> Tuple[list[Path], list[Path]]:
    """Extract the full-resolution original images and mp4s from a directory extracted from a warc.
    in_path should be the i.imgur.com subdir created after extracting a WARC. out_path will be
    filled with images and videos. Returns lists of what was selected and rejected."""
    # Collect IDs. Use a dict because there's more than one file per ID.
    ids: dict[str, None] = {}
    for img_path in in_path.iterdir():
        # Most IDs are 7 characters, some are five. Some five char ids have a suffix after an
        # underscore for alternate formats.
        if len(img_path.stem) == 7:
            if img_path.stem[5] == "_":
                ids[img_path.stem[:5]] = None
            else:
                ids[img_path.stem] = None
        elif len(img_path.stem) == 5:
            ids[img_path.stem] = None

    # Select the file we want for each ID. They come in lots of formats.
    paths = []
    no_original_files = 0
    for id in ids.keys():
        still_paths = []
        vid_paths = []
        for extension in [
            ".png",
            ".webp",
            ".jpg",
        ]:
            # If a still image was uploaded, this should be the original
            img_path = (in_path / id).with_suffix(extension)
            if img_path.exists():
                still_paths.append(img_path)
                # Sometimes there's both a png and a jpg, at the same resolution. I assume the
                # jpg is a transcode? Not sure.
                break
        assert len(still_paths) <= 1, f"ID {id} has multiple still files"
        for extension in [".mp4", ".gif"]:
            # If there's an mp4 and a gif, use the mp4
            img_path = (in_path / id).with_suffix(extension)
            if img_path.exists():
                vid_paths.append(img_path)
                break
        assert len(vid_paths) <= 1, f"found multiple video files for ID {id}"
        if len(still_paths) + len(vid_paths) == 0:
            # This is weirdly common
            no_original_files += 1
        elif len(vid_paths) == 1:
            paths.append(vid_paths[0])
        else:
            assert len(still_paths) == 1, f"ID {id} has no stills and is not a video"
            paths.append(still_paths[0])

    # Keep a list of rejected paths for debugging
    paths_map = dict([(path, None) for path in paths])
    rejected_paths = []
    for path in in_path.iterdir():
        if not (path in paths_map):
            rejected_paths.append(path)

    # Move the files to the output directory
    for path in paths:
        shutil.move(path, out_path / path.name)

    tqdm.write(f"No original files for {no_original_files} IDs")

    return paths, rejected_paths


def hash_dir(path: Path, conn: sqlite3.Connection, warc: str) -> None:
    """Hash the files in a directory and add them to the db."""
    toinsert = []
    tqdm.write(f"Hashing files in {path} for warc {warc}")
    for img_path in tqdm(list(path.iterdir()), desc="hashing"):
        hash_state = hashlib.blake2b()
        with open(img_path, "rb") as f:
            for chunk in iter(lambda: f.read(16384), b""):
                hash_state.update(chunk)
        hash = hash_state.hexdigest()
        toinsert.append((img_path.stem, warc, img_path.suffix, hash))
    with conn:
        # Sometimes an image is in multiple warcs, so we use INSERT OR IGNORE. No idea how that
        # happened.
        cur = conn.executemany(
            "INSERT OR IGNORE INTO files VALUES  (?, ?, ?, ?, 0)", toinsert
        )
        tqdm.write(
            f"Inserted {cur.rowcount} files into db, {len(toinsert) - cur.rowcount} already there."
        )


def dedup_dir(
    src_path: Path, dest_path: Path, conn: sqlite3.Connection
) -> Tuple[int, int]:
    """Go through a directory of images and videos, moving them to the destination if they aren't
    duplicates."""
    taken, skipped = 0, 0

    # We do this in chunks because otherwise it holds a lock on the DB for way too long and other
    # transactions time out.
    chunk_size = 1000
    files_ctr = 0
    conn.execute("BEGIN IMMEDIATE")

    for img_path in tqdm(list(src_path.iterdir()), desc="deduping"):
        # Has a duplicate of this file been processed?
        duplicates = conn.execute(
            """SELECT COUNT(*) FROM files WHERE
            blake2b = (SELECT blake2b FROM files WHERE id = ?) AND processed = 1""",
            (img_path.stem,),
        ).fetchone()[0]
        if duplicates == 0:
            img_path.rename(dest_path / img_path.name)
            taken += 1
            # Mark this file as processed
            conn.execute(
                "UPDATE files SET processed = 1 WHERE id = ?", (img_path.stem,)
            )
        else:
            skipped += 1

        files_ctr += 1
        if files_ctr % chunk_size == 0:
            conn.commit()
            conn.execute("BEGIN IMMEDIATE")

    conn.commit()
    return taken, skipped


def make_video_stills(
    src_path: Path, still_dest_path: Path, vid_dest_path: Path
) -> None:
    """Go through a directory with mp4s and gifs in it (among other files) and make still images of
    them. Moves the original video files to vid_dest_path and puts the generated stills in
    still_dest_path. Leaves other files where they are."""
    vid_paths = list(src_path.glob("*.mp4")) + list(src_path.glob("*.gif"))
    # With a tmpfs, this is CPU bound, so we do it in parallel

    def make_video_still(vid_path: Path) -> None:
        out_path = still_dest_path / (vid_path.name + ".png")
        # Select a random frame
        try:
            frames = imageio_ffmpeg.count_frames_and_secs(vid_path)[0]
            if frames == 0:
                tqdm.write(f"Skipping {vid_path} because it has no frames")
                return
            frame_to_save = random.randint(0, frames - 1)
            gen = imageio_ffmpeg.read_frames(vid_path)
            metadata = gen.__next__()
            for n in range(frame_to_save - 1):  # first output is metadata dict
                gen.__next__()
            frame_buffer = gen.__next__()
            frame_pil = Image.frombytes(
                mode="RGB", size=metadata["source_size"], data=frame_buffer
            )
            frame_pil.save(out_path)
        except (RuntimeError, StopIteration) as e:
            # Sometimes it fails when calling gen.__next__(), I'm not
            # sure why. Maybe the frame counts reported are inaccurate?
            tqdm.write(f"Couldn't process {vid_path} with ffmpeg: {e}")
        vid_path.rename(vid_dest_path / vid_path.name)

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        with tqdm(total=len(vid_paths), desc="video stills") as pbar:
            futs = [
                executor.submit(make_video_still, vid_path) for vid_path in vid_paths
            ]
            for fut in concurrent.futures.as_completed(futs):
                fut.result()
                pbar.update(1)


# Thread-safe connection pool
class SQLiteConnectionPool:
    def __init__(self, db_name: str) -> None:
        self.db_name = db_name
        self.local = threading.local()

    def get_conn(self) -> sqlite3.Connection:
        if not hasattr(self.local, "conn"):
            self.local.conn = sqlite3.connect(self.db_name, timeout=600)
            self.local.conn.execute("PRAGMA FOREIGN_KEYS = ON")
        return self.local.conn  # type: ignore[no-any-return]


def process_warc(
    warc: Path, warc_id: str, workdir: Path, pool: SQLiteConnectionPool
) -> Tuple[Path, Path, Path]:
    """Run all processing steps on a warc, resulting in a directory full of original
    still images, a directory full of stills extracted from videos and a directory full of the
    original videos. Workdir should be an empty directory used specifically for this warc.
    """
    conn = pool.get_conn()

    warc_extracted_dir = workdir / "warc_extracted"
    warc_extracted_dir.mkdir()
    tqdm.write(f"Extracting {warc}...")
    decompress_and_extract_warc(warc, warc_extracted_dir)

    orig_images_dir = workdir / "orig_images"
    orig_images_dir.mkdir()
    tqdm.write(f"Extracting original images from {warc}...")
    extract_from_dir(warc_extracted_dir / "i.imgur.com", orig_images_dir)

    tqdm.write(f"Hashing original images from {warc}...")
    hash_dir(orig_images_dir, conn, warc_id)

    tqdm.write(f"Deduplicating original images from {warc}...")
    deduped_dir = workdir / "deduped"
    deduped_dir.mkdir()
    taken, skipped = dedup_dir(orig_images_dir, deduped_dir, conn)
    tqdm.write(f"Deduplicated {warc}, took {taken} skipped {skipped}")

    tqdm.write(f"Extracting video stills from {warc}...")
    video_stills_dir = workdir / "video_stills"
    video_stills_dir.mkdir()
    videos_dir = workdir / "videos"
    videos_dir.mkdir()
    make_video_stills(deduped_dir, video_stills_dir, videos_dir)

    with conn:
        tqdm.write(f"Marking {warc} as processed")
        cur = conn.execute("UPDATE warcs SET processed = 1 WHERE id = ?", (warc_id,))
        tqdm.write(f"Marking {warc} as processed updated {cur.rowcount} rows")
    return deduped_dir, video_stills_dir, videos_dir


def ia_id_from_warc_filename(filename: str) -> str:
    """Compute the original IA id from a warc filename."""
    # Warc names look like imgur_20230513085622_e65ad1a5.1683789516.megawarc.warc.zst
    # we want the original IA id which looks like archiveteam_imgur_20230513085622_e65ad1a5.
    match = re.match(r"imgur_([^.]*)\..*", filename)
    assert match is not None, f"Couldn't parse IA id from filename {filename}"
    return "archiveteam_imgur_" + match.group(1)


def dl_and_process_warc(
    id: str, workdir: Path, pool: SQLiteConnectionPool
) -> Tuple[Path, Path, Path]:
    """Download and process a warc. Returns paths to a directory containing the original images,
    a directory containing stills extracted from videos, and a directory containing the original
    videos files. same as proces_warc"""
    conn = pool.get_conn()
    with conn:
        # Check if this warc has already been processed
        processed = conn.execute(
            "SELECT processed FROM warcs WHERE id = ?", (id,)
        ).fetchone()[0]
        assert processed == 0, f"Warc {id} has already been processed"
    tqdm.write(f"Downloading {id}...")
    warc_path = download_warc(id, workdir)
    tqdm.write(f"Downloaded {id} to {warc_path}")
    return process_warc(warc_path, id, workdir, pool)


def upload_tar(pool: SQLiteConnectionPool, tar_path: Path, log_path: Path) -> None:
    """Upload a tarball to R2."""
    conn = pool.get_conn()
    with conn:
        # Check if this tar has already been uploaded
        uploaded = conn.execute(
            "SELECT uploaded FROM warcs WHERE id = ?", (tar_path.stem,)
        ).fetchone()[0]
        assert uploaded == 0, f"Tar {tar_path} has already been uploaded"
    tqdm.write(f"Uploading {tar_path}...")
    with log_path.open("w") as f:
        subprocess.run(
            [
                "rclone",
                "copyto",
                "-P",
                "--s3-upload-concurrency",
                "16",
                str(tar_path),
                f"r2:txt2img-unsupervised-dataset/original-tarballs/{tar_path.name}",
            ],
            check=True,
            stdout=f,
            stderr=f,
        )
    with conn:
        conn.execute("UPDATE warcs SET uploaded = 1 WHERE id = ?", (tar_path.stem,))
    tqdm.write(f"Uploaded {tar_path}")


def process_warcs(
    pool: SQLiteConnectionPool,
    warcs: list[Union[Path, str]],
    workdir: Path,
    outdir: Path,
) -> None:
    """Process a list of warcs sequentially. warcs should be a list of either Path objects or IA ids."""
    assert outdir.exists(), f"Output directory {outdir} doesn't exist"
    assert workdir.exists(), f"Work directory {workdir} doesn't exist"
    # Pipeline so downloading, processing, and uploading can be simultaneous
    # Queue of warcs that are ready to be processed, either downloaded or already present
    ready_warc_queue = CQueue(maxsize=1)
    # Queue of tarballs ready to be uploaded
    ready_tarball_queue = CQueue(maxsize=1)

    def downloader_main() -> None:
        for warc in warcs:
            if type(warc) is str:
                ia_id: str = warc
            elif issubclass(type(warc), Path):
                ia_id = ia_id_from_warc_filename(warc.name)
            else:
                assert False, f"Invalid type for warc {warc}"
            conn = pool.get_conn()
            already_processed = conn.execute(
                "SELECT COUNT(*) FROM warcs WHERE id = ? and processed = 1", (ia_id,)
            ).fetchone()[0]
            assert already_processed == 0, f"Warc {ia_id} has already been processed"
            sub_workdir = workdir / ia_id
            sub_workdir.mkdir()
            if type(warc) is str:
                tqdm.write(f"Starting download of {warc}")
                warc_path = download_warc(warc, sub_workdir)
                ready_warc_queue.put((sub_workdir, warc_path))
                tqdm.write(f"Download of {warc} complete")
            else:
                tqdm.write(f"Using existing warc {warc}")
                ready_warc_queue.put((sub_workdir, warc))
        ready_warc_queue.close()

    downloader_thread = Thread(target=downloader_main, name="downloader")
    downloader_thread.start()

    def uploader_main() -> None:
        for tarball in CloseableQueue.dequeue(ready_tarball_queue):
            tqdm.write(f"Starting upload of {tarball}")
            upload_tar(pool, tarball, tarball.with_suffix(".log"))
            tqdm.write(f"Upload of {tarball} complete")
            tarball.unlink()

    uploader_thread = Thread(target=uploader_main, name="uploader")
    uploader_thread.start()

    with tqdm(total=len(warcs), desc="Processing warcs") as pbar:
        for sub_workdir, warc_path in CloseableQueue.dequeue(ready_warc_queue):
            tqdm.write(f"Starting processing of {warc_path}")
            ia_id = sub_workdir.name
            try:
                deduped_dir, video_stills_dir, videos_dir = process_warc(
                    warc_path, ia_id, sub_workdir, pool
                )
            except Exception as e:
                tqdm.write(f"⚠️⚠️⚠️Error processing {warc_path}: {e}, skipping")
                shutil.rmtree(sub_workdir)
                continue
            tqdm.write(f"Processing of {warc_path} complete")
            this_outdir = outdir / ia_id
            this_outdir.mkdir()
            tqdm.write(f"Copying deduped images to {this_outdir}")
            shutil.move(deduped_dir, this_outdir / "deduped")
            tqdm.write(f"Copying video stills to {this_outdir}")
            shutil.move(video_stills_dir, this_outdir / "video_stills")
            tqdm.write(f"Copying videos to {this_outdir}")
            shutil.move(videos_dir, this_outdir / "videos")
            shutil.rmtree(sub_workdir)
            tar_path = this_outdir.with_suffix(".tar")
            tqdm.write(f"Creating tar {tar_path}")
            subprocess.run(
                ["tar", "-cf", tar_path, "-C", outdir, ia_id],
                check=True,
            )
            tqdm.write(f"Tar done, deleting {this_outdir}")
            shutil.rmtree(this_outdir)
            tqdm.write(f"Tarball {tar_path} ready for upload")
            ready_tarball_queue.put(tar_path)
            pbar.update(1)

    ready_tarball_queue.close()
    tqdm.write("Waiting for last uploads...")
    uploader_thread.join()
    downloader_thread.join()

    tqdm.write("Done processing warcs 🎉")


def get_random_warc_ids(conn: sqlite3.Connection, n: int) -> list[str]:
    """Get n random warc ids that haven't been processed yet."""
    return [
        row[0]
        for row in conn.execute(
            "SELECT id FROM warcs WHERE processed = 0 ORDER BY RANDOM() LIMIT ?",
            (n,),
        ).fetchall()
    ]
