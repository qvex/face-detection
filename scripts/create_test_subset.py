from pathlib import Path
from dataclasses import dataclass
import shutil
import random
from typing import Optional
from collections import defaultdict

from src.core.types import Success, Failure, Result

@dataclass(frozen=True, slots=True)
class TestSubsetConfig:
    num_individuals: int = 10
    images_per_person: int = 10
    total_images: int = 100

def find_identity_folders(dataset_root: Path) -> list[Path]:
    identity_folders = []

    for path in dataset_root.rglob("*"):
        if path.is_dir() and len(list(path.glob("*.jpg"))) >= 10:
            identity_folders.append(path)
        elif path.is_dir() and len(list(path.glob("*.png"))) >= 10:
            identity_folders.append(path)

    return identity_folders

def select_random_identities(
    identity_folders: list[Path],
    num_identities: int
) -> list[Path]:
    if len(identity_folders) < num_identities:
        raise ValueError(
            f"Not enough identities found. Need {num_identities}, found {len(identity_folders)}"
        )

    return random.sample(identity_folders, num_identities)

def copy_images_for_identity(
    source_folder: Path,
    dest_folder: Path,
    num_images: int,
    person_id: int
) -> int:
    images = list(source_folder.glob("*.jpg")) + list(source_folder.glob("*.png"))

    if len(images) < num_images:
        num_images = len(images)

    selected_images = random.sample(images, num_images)

    person_dir = dest_folder / f"person_{person_id:03d}"
    person_dir.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(selected_images, 1):
        dest_path = person_dir / f"{idx:04d}{img_path.suffix}"
        shutil.copy2(img_path, dest_path)

    return len(selected_images)

def create_test_subset_from_imfdb(
    data_root: Path,
    config: Optional[TestSubsetConfig] = None
) -> Result[Path, str]:
    cfg = config if config else TestSubsetConfig()

    imfdb_path = data_root / "raw" / "IMFDB"
    test_path = data_root / "test"

    if not imfdb_path.exists():
        return Failure(f"IMFDB dataset not found at {imfdb_path}. Run download_indian_datasets.py first.")

    print("\n" + "=" * 70)
    print("CREATING TEST SUBSET FROM IMFDB")
    print("=" * 70)
    print(f"Source: {imfdb_path}")
    print(f"Destination: {test_path}")
    print(f"Target: {cfg.num_individuals} individuals, {cfg.images_per_person} images each")
    print("=" * 70)

    print("\nScanning for identity folders...")
    identity_folders = find_identity_folders(imfdb_path)
    print(f"Found {len(identity_folders)} identity folders with 10+ images")

    if len(identity_folders) < cfg.num_individuals:
        return Failure(
            f"Insufficient identities. Need {cfg.num_individuals}, found {len(identity_folders)}"
        )

    print(f"\nSelecting {cfg.num_individuals} random identities...")
    selected_identities = select_random_identities(identity_folders, cfg.num_individuals)

    if test_path.exists():
        print(f"\nWarning: Test directory exists. Removing old data...")
        shutil.rmtree(test_path)

    test_path.mkdir(parents=True, exist_ok=True)

    print("\nCopying images...")
    total_copied = 0
    for person_id, identity_folder in enumerate(selected_identities, 1):
        copied_count = copy_images_for_identity(
            identity_folder,
            test_path,
            cfg.images_per_person,
            person_id
        )
        total_copied += copied_count
        print(f"  Person {person_id:03d}: {copied_count} images from {identity_folder.name}")

    print("\n" + "=" * 70)
    print("TEST SUBSET CREATED SUCCESSFULLY")
    print("=" * 70)
    print(f"Location: {test_path}")
    print(f"Individuals: {cfg.num_individuals}")
    print(f"Total images: {total_copied}")
    print(f"Average images per person: {total_copied / cfg.num_individuals:.1f}")
    print("\nNext step: Run 'python scripts/run_baseline_tests.py' to test models")
    print("=" * 70)

    return Success(test_path)

def create_test_subset_from_ifexd(
    data_root: Path,
    config: Optional[TestSubsetConfig] = None
) -> Result[Path, str]:
    cfg = config if config else TestSubsetConfig()

    ifexd_path = data_root / "raw" / "IFExD"
    test_path = data_root / "test"

    if not ifexd_path.exists():
        return Failure(f"IFExD dataset not found at {ifexd_path}. Run download_indian_datasets.py first.")

    print("\n" + "=" * 70)
    print("CREATING TEST SUBSET FROM IFExD")
    print("=" * 70)
    print(f"Source: {ifexd_path}")
    print(f"Destination: {test_path}")
    print(f"Target: {cfg.num_individuals} individuals, {cfg.images_per_person} images each")
    print("=" * 70)

    print("\nScanning for identity folders...")
    identity_folders = find_identity_folders(ifexd_path)
    print(f"Found {len(identity_folders)} identity folders with 10+ images")

    if len(identity_folders) < cfg.num_individuals:
        return Failure(
            f"Insufficient identities. Need {cfg.num_individuals}, found {len(identity_folders)}"
        )

    print(f"\nSelecting {cfg.num_individuals} random identities...")
    selected_identities = select_random_identities(identity_folders, cfg.num_individuals)

    if test_path.exists():
        print(f"\nWarning: Test directory exists. Removing old data...")
        shutil.rmtree(test_path)

    test_path.mkdir(parents=True, exist_ok=True)

    print("\nCopying images...")
    total_copied = 0
    for person_id, identity_folder in enumerate(selected_identities, 1):
        copied_count = copy_images_for_identity(
            identity_folder,
            test_path,
            cfg.images_per_person,
            person_id
        )
        total_copied += copied_count
        print(f"  Person {person_id:03d}: {copied_count} images from {identity_folder.name}")

    print("\n" + "=" * 70)
    print("TEST SUBSET CREATED SUCCESSFULLY")
    print("=" * 70)
    print(f"Location: {test_path}")
    print(f"Individuals: {cfg.num_individuals}")
    print(f"Total images: {total_copied}")
    print(f"Average images per person: {total_copied / cfg.num_individuals:.1f}")
    print("\nNext step: Run 'python scripts/run_baseline_tests.py' to test models")
    print("=" * 70)

    return Success(test_path)

def main() -> int:
    data_root = Path("./data")

    print("\n" + "=" * 70)
    print("TEST SUBSET CREATION")
    print("=" * 70)
    print("\nAvailable datasets:")
    print("1. IMFDB (Indian Movie Face Database)")
    print("2. IFExD (Indian Face Expression Database)")
    print("=" * 70)

    choice = input("\nSelect dataset source (1 or 2): ").strip()

    if choice == "1":
        result = create_test_subset_from_imfdb(data_root)
    elif choice == "2":
        result = create_test_subset_from_ifexd(data_root)
    else:
        print("Invalid choice. Exiting.")
        return 1

    if isinstance(result, Success):
        return 0
    else:
        print(f"\nERROR: {result.error}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
