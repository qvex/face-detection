from pathlib import Path
import subprocess
import sys

def check_kaggle_api():
    try:
        import kaggle
        return True
    except ImportError:
        return False

def install_kaggle_api():
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "kaggle"],
            check=True,
            capture_output=True
        )
        print("Kaggle API installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install Kaggle API: {e.stderr.decode()}")
        return False

def download_imfdb(data_root):
    if not check_kaggle_api():
        print("Kaggle API not found. Installing...")
        if not install_kaggle_api():
            return False

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
    print("3. Place kaggle.json in:")
    print("   Windows: C:\\Users\\<username>\\.kaggle\\")
    print("   Linux/Mac: ~/.kaggle/")
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
        print(f"\nSUCCESS: IMFDB downloaded to {dataset_path}")
        return True
    except Exception as e:
        print(f"\nFAILED: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure kaggle.json is in correct location")
        print("2. Verify Kaggle account and API token are valid")
        print("3. Accept dataset terms at: https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset")
        return False

def clone_ifexd(data_root):
    dataset_path = data_root / "raw" / "IFExD"

    if dataset_path.exists():
        print(f"\nIFExD already exists at {dataset_path}")
        return True

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
            capture_output=True,
            text=True
        )
        print(f"\nSUCCESS: IFExD cloned to {dataset_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nFAILED: {e.stderr}")
        return False
    except FileNotFoundError:
        print("\nFAILED: Git not found")
        print("Please install Git: https://git-scm.com/")
        return False

def main():
    data_root = Path("./data")

    print("\n" + "=" * 70)
    print("INDIAN FACE DATASETS DOWNLOAD")
    print("=" * 70)
    print("\nThis script will download:")
    print("1. IMFDB (Indian Movie Face Database) - 34,512 images, ~3.5 GB")
    print("2. IFExD (Indian Face Expression Database) - ~1000 images, ~500 MB")
    print("\nTotal estimated size: ~4 GB")
    print("Total estimated download time: 10-30 minutes (depending on internet speed)")
    print("=" * 70)

    print("\nChoose download option:")
    print("1. Download IMFDB only (Kaggle, requires API setup)")
    print("2. Clone IFExD only (GitHub, requires Git)")
    print("3. Download both (recommended)")

    choice = input("\nEnter choice (1/2/3): ").strip()

    successful = 0
    total = 0

    if choice in ["1", "3"]:
        total += 1
        print("\n\n[1/2] Downloading IMFDB...")
        if download_imfdb(data_root):
            successful += 1

    if choice in ["2", "3"]:
        total += 1
        print("\n\n[2/2] Cloning IFExD...")
        if clone_ifexd(data_root):
            successful += 1

    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"Successfully downloaded: {successful}/{total} datasets")

    if successful > 0:
        print("\nNext step: Run 'python scripts/create_test_subset.py' to create baseline test dataset")
    else:
        print("\nNo datasets downloaded. Please resolve errors and try again.")

    return 0 if successful > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
