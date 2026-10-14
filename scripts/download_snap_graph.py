"""BrandPulse AI - Stanford SNAP Graph Acquisition Utility.

Downloads real social network topology datasets directly from Stanford SNAP:
- twitter_combined.txt.gz (Ego-networks combined: 81,306 nodes, 1,768,149 edges)
- higgs-retweet_network.edgelist.gz (Higgs Twitter retweet diffusion graph)
- higgs-mention_network.edgelist.gz (Higgs user-to-user mention graph)

Includes live URL verification, chunked streaming, download progress,
integrity checking, and automatic gzip decompression into data/raw/snap/.
"""

from __future__ import annotations

import gzip
import logging
import shutil
import time
from pathlib import Path
from typing import Any
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("download_snap")

SNAP_TARGETS = [
    {
        "name": "twitter_combined",
        "url": "https://snap.stanford.edu/data/twitter_combined.txt.gz",
        "archive_filename": "twitter_combined.txt.gz",
        "extracted_filename": "twitter_combined.txt",
        "description": "Stanford SNAP Twitter Ego Networks combined edge list",
    },
    {
        "name": "higgs_retweet",
        "url": "https://snap.stanford.edu/data/higgs-retweet_network.edgelist.gz",
        "archive_filename": "higgs-retweet_network.edgelist.gz",
        "extracted_filename": "higgs-retweet_network.edgelist",
        "description": "Higgs Twitter Retweet network",
    },
    {
        "name": "higgs_mention",
        "url": "https://snap.stanford.edu/data/higgs-mention_network.edgelist.gz",
        "archive_filename": "higgs-mention_network.edgelist.gz",
        "extracted_filename": "higgs-mention_network.edgelist",
        "description": "Higgs Twitter Mention network",
    },
]


def check_url_live(url: str, timeout: int = 10) -> bool:
    """Send a HEAD request to verify that the remote file is accessible."""
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return response.status_code in (200, 302, 301)
    except Exception as e:
        logger.warning(f"HEAD check failed for {url}: {e}. Trying GET with stream...")
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                return r.status_code == 200
        except Exception:
            return False


def download_stream(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> bool:
    """Download a file with streaming and basic retry logic."""
    temp_path = destination.with_suffix(".tmp")
    for attempt in range(1, 4):
        try:
            logger.info(f"Downloading {url} (Attempt {attempt}/3)...")
            with requests.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                total_bytes = int(response.headers.get("content-length", 0))
                downloaded = 0
                with open(temp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_bytes > 0:
                                percent = (downloaded / total_bytes) * 100
                                if downloaded % (10 * chunk_size) == 0:
                                    logger.info(f"  Downloaded: {downloaded / (1024*1024):.1f} MB ({percent:.1f}%)")
                temp_path.rename(destination)
                logger.info(f"Successfully saved to {destination} ({downloaded / (1024*1024):.2f} MB)")
                return True
        except Exception as e:
            logger.warning(f"Attempt {attempt} failed: {e}")
            if temp_path.exists():
                temp_path.unlink()
            time.sleep(2 * attempt)
    return False


def decompress_gzip(gz_path: Path, output_path: Path) -> bool:
    """Decompress a .gz file to plain text."""
    try:
        logger.info(f"Decompressing {gz_path.name} -> {output_path.name}...")
        with gzip.open(gz_path, "rb") as f_in, open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        logger.info(f"Extracted to {output_path} ({output_path.stat().st_size / (1024*1024):.2f} MB)")
        return True
    except Exception as e:
        logger.error(f"Failed to decompress {gz_path}: {e}")
        return False


def acquire_snap_data(output_dir: Path | None = None) -> dict[str, Any]:
    """Execute download and extraction of SNAP graph datasets."""
    if output_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        output_dir = project_root / "data" / "raw" / "snap"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}

    for target in SNAP_TARGETS:
        name = target["name"]
        gz_file = output_dir / target["archive_filename"]
        txt_file = output_dir / target["extracted_filename"]

        if txt_file.exists() and txt_file.stat().st_size > 0:
            logger.info(f"Dataset '{name}' already present at {txt_file} (skipping download).")
            # Count lines
            line_count = 0
            with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                for _ in f:
                    line_count += 1
            summary[name] = {
                "status": "ready",
                "file": str(txt_file),
                "size_mb": round(txt_file.stat().st_size / (1024 * 1024), 2),
                "edge_count": line_count,
            }
            continue

        if not check_url_live(target["url"]):
            logger.error(f"URL unreachable: {target['url']}")
            summary[name] = {"status": "unreachable", "url": target["url"]}
            continue

        if not gz_file.exists() or gz_file.stat().st_size == 0:
            success = download_stream(target["url"], gz_file)
            if not success:
                summary[name] = {"status": "download_failed"}
                continue

        ok = decompress_gzip(gz_file, txt_file)
        if ok:
            line_count = 0
            with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                for _ in f:
                    line_count += 1
            summary[name] = {
                "status": "ready",
                "file": str(txt_file),
                "size_mb": round(txt_file.stat().st_size / (1024 * 1024), 2),
                "edge_count": line_count,
            }
        else:
            summary[name] = {"status": "decompression_failed"}

    return summary


if __name__ == "__main__":
    results = acquire_snap_data()
    print("\n=== SNAP ACQUISITION SUMMARY ===")
    for k, v in results.items():
        print(f"{k:20s}: {v}")
