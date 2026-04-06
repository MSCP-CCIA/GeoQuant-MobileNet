"""
Downloader and formatter for the CUB-200-2011 dataset.
Downloads the raw tarball, extracts it, and reorganizes it into PyTorch ImageFolder format.
"""

import os
import tarfile
import urllib.request
import shutil
from pathlib import Path
import sys

# Dynamic path injection to use the project's custom logger
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.logger import logger

# Official Caltech Data URL (more stable than legacy links)
CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"


def download_progress_hook(count: int, block_size: int, total_size: int) -> None:
    """Callback to show download progress in the logger."""
    percent = int(count * block_size * 100 / total_size)
    if percent % 10 == 0 and percent <= 100:
        # Avoids log spam by printing on the same line
        sys.stdout.write(f"\rDownloading CUB-200-2011... {percent}%")
        sys.stdout.flush()


def format_cub_dataset(base_dir: Path) -> None:
    """
    Reads the dataset metadata and moves images into train/test split directories.
    """
    extracted_dir = base_dir / "CUB_200_2011"
    images_file = extracted_dir / "images.txt"
    split_file = extracted_dir / "train_test_split.txt"
    images_dir = extracted_dir / "images"

    train_dir = base_dir / "train"
    test_dir = base_dir / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading dataset metadata (images.txt and train_test_split.txt)...")

    with open(images_file, "r") as f:
        images = {line.split()[0]: line.split()[1] for line in f.readlines()}

    with open(split_file, "r") as f:
        splits = {line.split()[0]: int(line.split()[1]) for line in f.readlines()}

    logger.info("Reorganizing files into train and test directories...")

    for img_id, img_path in images.items():
        is_train = splits[img_id] == 1
        target_split_dir = train_dir if is_train else test_dir

        # Extract class name from the original path (e.g., '001.Black_footed_Albatross/img.jpg')
        class_name = img_path.split("/")[0]
        class_dir = target_split_dir / class_name
        class_dir.mkdir(exist_ok=True)

        src_file = images_dir / img_path
        dst_file = class_dir / Path(img_path).name

        if src_file.exists():
            shutil.move(str(src_file), str(dst_file))

    # Cleanup: Remove the original extracted files that are no longer used
    logger.info("Cleaning up temporary extracted files...")
    shutil.rmtree(extracted_dir)


def main():
    target_dir = project_root / "data" / "raw" / "cub200"
    target_dir.mkdir(parents=True, exist_ok=True)

    tar_path = target_dir / "CUB_200_2011.tgz"

    # 1. Download
    if not (target_dir / "train").exists():
        if not tar_path.exists():
            logger.info(f"Starting download from {CUB_URL}")
            try:
                urllib.request.urlretrieve(CUB_URL, tar_path, reporthook=download_progress_hook)
                print()  # New line after progress bar
                logger.info("Download completed.")
            except Exception as e:
                logger.error(f"Failed to download dataset: {e}")
                return
        else:
            logger.info("Tarball already exists. Skipping download.")

        # 2. Extract
        logger.info("Extracting tarball... This may take a moment.")
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=target_dir)
        except Exception as e:
            logger.error(f"Failed to extract tarball: {e}")
            return

        # 3. Format
        format_cub_dataset(target_dir)

        # 4. Cleanup tarball
        try:
            tar_path.unlink()
            logger.info("Removed original tarball to save space.")
        except Exception as e:
            logger.warning(f"Could not remove tarball: {e}")

        logger.info(f"CUB-200-2011 is successfully prepared at {target_dir}")
    else:
        logger.info(f"Dataset already formatted and ready at {target_dir}")


if __name__ == "__main__":
    main()