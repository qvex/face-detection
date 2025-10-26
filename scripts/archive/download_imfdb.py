from pathlib import Path
import subprocess
import sys

def install_kaggle_api():
    print("Installing Kaggle API...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "kaggle"],
            check=True,
            capture_output=True
        )
        print("Kaggle API installed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install Kaggle API: {e.stderr.decode()}")
        return False

def main():
    data_root = Path("./data")
    dataset_path = data_root / "raw" / "IMFDB"
    dataset_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("IMFDB Dataset Download (Indian Movie Face Database)")
    print("=" * 70)
    print("\nDataset: IMFDB from Kaggle")
    print("Source: vasukipatel/face-recognition-dataset")
    print("Size: ~3.5 GB")
    print("Images: 34,512 images of Indian celebrities")
    print(f"Destination: {dataset_path.absolute()}")
    print("\n" + "=" * 70)
    print("PREREQUISITES:")
    print("=" * 70)
    print("1. Kaggle account (create at https://www.kaggle.com)")
    print("2. Kaggle API token (download from https://www.kaggle.com/settings/account)")
    print("3. Place kaggle.json in:")
    print(f"   Windows: C:\\Users\\{Path.home().name}\\.kaggle\\")
    print("   Linux/Mac: ~/.kaggle/")
    print("\nIf you haven't set up Kaggle API yet:")
    print("  - Visit: https://www.kaggle.com/docs/api")
    print("  - Follow 'Authentication' section")
    print("=" * 70)

    try:
        import kaggle
    except ImportError:
        if not install_kaggle_api():
            return 1
        import kaggle

    print("\nStarting download (this may take 10-20 minutes)...")
    print("-" * 70)

    try:
        kaggle.api.dataset_download_files(
            "vasukipatel/face-recognition-dataset",
            path=str(dataset_path),
            unzip=True
        )

        print("\n" + "=" * 70)
        print("SUCCESS: IMFDB Downloaded Successfully")
        print("=" * 70)
        print(f"Location: {dataset_path.absolute()}")

        image_count = sum(1 for _ in dataset_path.rglob("*.jpg"))
        print(f"Images found: {image_count}")

        print("\nNext step:")
        print("  python scripts/create_test_subset.py")
        print("=" * 70)
        return 0

    except FileNotFoundError as e:
        print("\n" + "=" * 70)
        print("ERROR: Kaggle API Configuration Not Found")
        print("=" * 70)
        print("\nKaggle API credentials not configured.")
        print("\nSetup instructions:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. Save kaggle.json to:")
        print(f"   C:\\Users\\{Path.home().name}\\.kaggle\\kaggle.json")
        print("\nThen run this script again.")
        print("=" * 70)
        return 1

    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR: Download Failed")
        print("=" * 70)
        print(f"Error: {str(e)}")
        print("\nPossible causes:")
        print("1. kaggle.json not in correct location")
        print("2. Invalid API credentials")
        print("3. Dataset terms not accepted (visit dataset URL and accept)")
        print("4. Network connection issue")
        print("\nDataset URL: https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())
