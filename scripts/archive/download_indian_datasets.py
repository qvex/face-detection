from pathlib import Path
from dataclasses import dataclass
import subprocess
import sys
from typing import Optional
from src.core.types import Success, Failure, Result

@dataclass(frozen=True, slots=True)
class DatasetInfo:
    name: str
    source: str
    path: Path
    estimated_size_gb: float
    image_count: int

def check_kaggle_api() -> bool:
    try:
        import kaggle
        return True
    except ImportError:
        return False

def install_kaggle_api() -> Result[str, str]:
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "kaggle"],
            check=True,
            capture_output=True
        )
        return Success("Kaggle API installed successfully")
    except subprocess.CalledProcessError as e:
        return Failure(f"Failed to install Kaggle API: {e.stderr.decode()}")

def download_imfdb(data_root: Path) -> Result[DatasetInfo, str]:
    if not check_kaggle_api():
        install_result = install_kaggle_api()
        if isinstance(install_result, Failure):
            return install_result

    dataset_path = data_root / "raw" / "IMFDB"
    dataset_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("IMFDB Dataset Download (Indian Movie Face Database)")
    print("=" * 70)
    print("\nDataset: IMFDB from Kaggle")
    print("Size: ~3.5 GB")
    print("Images: 34,512 images")
    print(f"Destination: {dataset_path}")
    print("\nNote: Requires Kaggle API credentials")
    print("Setup: https://www.kaggle.com/docs/api")
    print("1. Create account on kaggle.com")
    print("2. Go to Account > Create New API Token")
    print("3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)")
    print("\n" + "-" * 70)
    print("Starting download...")
    print("-" * 70)

    try:
        import kaggle
        kaggle.api.dataset_download_files(
            "vasukipatel/face-recognition-dataset",
            path=str(dataset_path),
            unzip=True
        )

        return Success(DatasetInfo(
            name="IMFDB",
            source="Kaggle: vasukipatel/face-recognition-dataset",
            path=dataset_path,
            estimated_size_gb=3.5,
            image_count=34512
        ))
    except Exception as e:
        return Failure(f"Failed to download IMFDB: {str(e)}")

def clone_ifexd(data_root: Path) -> Result[DatasetInfo, str]:
    dataset_path = data_root / "raw" / "IFExD"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("IFExD Dataset Clone (Indian Face Expression Database)")
    print("=" * 70)
    print("\nDataset: IFExD from GitHub")
    print("Size: ~500 MB")
    print("Images: ~1000+ images with expressions")
    print(f"Destination: {dataset_path}")
    print("License: BSD")
    print("\n" + "-" * 70)
    print("Starting clone...")
    print("-" * 70)

    try:
        subprocess.run(
            ["git", "clone", "https://github.com/ravi0531rp/IFExD.git", str(dataset_path)],
            check=True,
            capture_output=True
        )

        return Success(DatasetInfo(
            name="IFExD",
            source="GitHub: ravi0531rp/IFExD",
            path=dataset_path,
            estimated_size_gb=0.5,
            image_count=1000
        ))
    except subprocess.CalledProcessError as e:
        return Failure(f"Failed to clone IFExD: {e.stderr.decode()}")
    except FileNotFoundError:
        return Failure("Git not found. Please install Git: https://git-scm.com/")

def download_all_indian_datasets(data_root: Optional[Path] = None) -> list[DatasetInfo]:
    root = data_root if data_root else Path("./data")

    print("\n" + "=" * 70)
    print("INDIAN FACE DATASETS DOWNLOAD")
    print("=" * 70)
    print("\nThis script will download:")
    print("1. IMFDB (Indian Movie Face Database) - 34,512 images, ~3.5 GB")
    print("2. IFExD (Indian Face Expression Database) - ~1000 images, ~500 MB")
    print("\nTotal estimated size: ~4 GB")
    print("Total estimated download time: 10-30 minutes (depending on internet speed)")
    print("=" * 70)

    user_input = input("\nProceed with download? (yes/no): ").strip().lower()
    if user_input not in ['yes', 'y']:
        print("Download cancelled.")
        return []

    successful_downloads = []

    print("\n\n[1/2] Downloading IMFDB...")
    imfdb_result = download_imfdb(root)
    if isinstance(imfdb_result, Success):
        successful_downloads.append(imfdb_result.value)
        print(f"\nSUCCESS: IMFDB downloaded to {imfdb_result.value.path}")
    else:
        print(f"\nFAILED: {imfdb_result.error}")
        print("You can download manually from: https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset")

    print("\n\n[2/2] Cloning IFExD...")
    ifexd_result = clone_ifexd(root)
    if isinstance(ifexd_result, Success):
        successful_downloads.append(ifexd_result.value)
        print(f"\nSUCCESS: IFExD cloned to {ifexd_result.value.path}")
    else:
        print(f"\nFAILED: {ifexd_result.error}")
        print("You can clone manually: git clone https://github.com/ravi0531rp/IFExD.git")

    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"Successfully downloaded: {len(successful_downloads)}/2 datasets")

    for dataset in successful_downloads:
        print(f"\n{dataset.name}:")
        print(f"  Path: {dataset.path}")
        print(f"  Images: ~{dataset.image_count}")
        print(f"  Size: ~{dataset.estimated_size_gb} GB")

    if len(successful_downloads) > 0:
        print("\nNext step: Run 'python scripts/create_test_subset.py' to create baseline test dataset")
    else:
        print("\nNo datasets downloaded. Please resolve errors and try again.")

    return successful_downloads

if __name__ == "__main__":
    download_all_indian_datasets()
