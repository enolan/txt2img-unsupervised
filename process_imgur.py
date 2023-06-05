"""Process imgur archives from ArchiveTeam/Internet Archive"""
import concurrent.futures
import hashlib
import imageio_ffmpeg  # type: ignore[import]
import internetarchive as ia
import random
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
from pathlib import Path
from PIL import Image
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


def download_warc(id: str, dest_dir: Path) -> Path:
    """Download a warc from the internet archive. This will make a new directory under dest_dir
    named after the id. Returns the path to the compressed warc, which will be in dest_dir.
    """
    ia.download(id, formats="Web ARChive ZST", destdir=str(dest_dir))
    zsts = list(dest_dir.glob("*.zst"))
    assert len(zsts) == 1, f"Expected one zst file, found {len(zsts)}"
    out_path = dest_dir / zsts[0].name
    zsts[0].rename(out_path)
    shutil.rmtree(dest_dir / id)
    return out_path


def decompress_and_extract_warc(compressed_warc_path: Path, dest_dir: Path) -> None:
    """Decompress and extract a WARC. After this, the dest_dir will contain a directory named
    i.imgur.com"""
    # Decompress using their weird zstd variant format.
    with tempfile.NamedTemporaryFile(dir=dest_dir, delete=False) as f:
        subprocess.run(["zstdwarccat", compressed_warc_path], stdout=f)
        decompressed_warc_path = Path(f.name)
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
    shutil.rmtree(dest_dir / "imgur.com")


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
    conn.commit()
    return taken, skipped


def make_video_stills(
    src_path: Path, still_dest_path: Path, vid_dest_path: Path
) -> None:
    """Go through a directory with mp4s and gifs in it (among other files) and make still images of
    them. Moves the original video files to vid_dest_path and puts the generated stills in
    still_dest_path. Leaves other files where they are."""
    vid_paths = list(src_path.glob("*.mp4")) + list(src_path.glob("*.gif"))
    for vid_path in tqdm(vid_paths, desc="video stills"):
        out_path = still_dest_path / (vid_path.name + ".png")
        # Select a random frame
        try:
            frames = imageio_ffmpeg.count_frames_and_secs(vid_path)[0]
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


# Thread-safe connection pool
class SQLiteConnectionPool:
    def __init__(self, db_name: str) -> None:
        self.db_name = db_name
        self.local = threading.local()

    def get_conn(self) -> sqlite3.Connection:
        if not hasattr(self.local, "conn"):
            self.local.conn = sqlite3.connect(self.db_name)
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
    hash_dir(orig_images_dir, conn, warc.stem)

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
    warc_dir = download_warc(id, workdir)
    warc = list(warc_dir.glob("*.warc.zst"))[0]
    return process_warc(warc, id, workdir, pool)


def process_warcs_threaded(
    pool: SQLiteConnectionPool,
    warcs: list[Union[Path, str]],
    workdir: Path,
    outdir: Path,
    max_workers: int = 16,
) -> None:
    """Process a list of warcs in parallel. warcs should be a list of either Path objects or IA ids."""
    assert outdir.exists(), f"Output directory {outdir} doesn't exist"
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for warc in warcs:
            warc_type = type(warc)
            if type(warc) is str:
                ia_id: str = warc
            elif issubclass(warc_type, Path):
                # MyPy's narrowing is broken here
                ia_id = ia_id_from_warc_filename(warc.name)  # type: ignore[union-attr]
            else:
                assert False, f"Invalid type for warc {warc}"
            sub_workdir = workdir / ia_id
            sub_workdir.mkdir()
            if warc_type is str:
                tqdm.write(f"Queueing {ia_id} for download and processing")
                futures[
                    executor.submit(dl_and_process_warc, ia_id, sub_workdir, pool)
                ] = ia_id
            elif issubclass(warc_type, Path):
                tqdm.write(f"Queueing {warc} for processing")
                futures[
                    executor.submit(process_warc, warc, ia_id, sub_workdir, pool)  # type: ignore[arg-type]
                ] = ia_id
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc="warcs"
        ):
            deduped_dir, video_stills_dir, videos_dir = future.result()
            # Move the results to the output directory
            this_outdir = outdir / futures[future]
            this_outdir.mkdir()
            shutil.move(deduped_dir, this_outdir / "deduped")
            shutil.move(video_stills_dir, this_outdir / "video_stills")
            shutil.move(videos_dir, this_outdir / "videos")


def get_random_warc_ids(conn: sqlite3.Connection, n: int) -> list[str]:
    """Get n random warc ids that haven't been processed yet."""
    return [
        row[0]
        for row in conn.execute(
            "SELECT id FROM warcs WHERE processed = 0 ORDER BY RANDOM() LIMIT ?",
            (n,),
        ).fetchall()
    ]
