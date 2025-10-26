#!/usr/bin/env python3
"""
Create test subset from FRD dataset (Indian celebrities only).
Extracts 10 images from each of 8 Indian celebrities = 80 total images.
"""
import pandas as pd
from pathlib import Path
import shutil
import random
from collections import defaultdict


def create_test_subset():
    """Create test subset from FRD dataset with Indian celebrities only."""

    # Configuration
    dataset_root = Path("dataset/frd")
    csv_path = dataset_root / "Dataset.csv"
    images_root = dataset_root / "Faces" / "Faces"
    test_root = Path("data/test")

    images_per_person = 10

    # Indian celebrities available in the dataset
    indian_celebrities = [
        'Vijay Deverakonda',
        'Priyanka Chopra',
        'Hrithik Roshan',
        'Alia Bhatt',
        'Amitabh Bachchan',
        'Anushka Sharma',
        'Akshay Kumar',
        'Virat Kohli'
    ]

    print("=" * 70)
    print("CREATING TEST SUBSET FROM FRD DATASET (INDIAN CELEBRITIES ONLY)")
    print("=" * 70)
    print(f"Source: {images_root}")
    print(f"Destination: {test_root}")
    print(f"Individuals: {len(indian_celebrities)}")
    print(f"Images per person: {images_per_person}")
    print(f"Total images: {len(indian_celebrities) * images_per_person}")
    print("=" * 70)

    # Verify dataset exists
    if not csv_path.exists():
        print(f"\nERROR: Dataset CSV not found at {csv_path}")
        return 1

    if not images_root.exists():
        print(f"\nERROR: Images directory not found at {images_root}")
        return 1

    # Read CSV
    print("\nReading dataset CSV...")
    df = pd.read_csv(csv_path)
    print(f"Total images in dataset: {len(df)}")

    # Filter for Indian celebrities
    indian_mask = df['label'].isin(indian_celebrities)
    indian_df = df[indian_mask]
    print(f"Indian celebrity images: {len(indian_df)}")

    # Group by person
    images_by_person = defaultdict(list)
    for _, row in indian_df.iterrows():
        person = row['label']
        image_file = row['id']
        images_by_person[person].append(image_file)

    print(f"\nIndian celebrities found:")
    for person in sorted(images_by_person.keys()):
        print(f"  {person:<25} {len(images_by_person[person]):>3} images")

    # Remove old test directory if exists
    if test_root.exists():
        print(f"\nRemoving old test directory...")
        shutil.rmtree(test_root)

    test_root.mkdir(parents=True, exist_ok=True)

    # Create test subset
    print(f"\nCreating test subset...")
    total_copied = 0

    for person_id, person in enumerate(sorted(images_by_person.keys()), 1):
        images = images_by_person[person]

        # Randomly select images_per_person images
        num_to_select = min(images_per_person, len(images))
        selected_images = random.sample(images, num_to_select)

        # Create person directory
        person_dir = test_root / f"person_{person_id:02d}_{person.replace(' ', '_')}"
        person_dir.mkdir(parents=True, exist_ok=True)

        # Copy images
        for idx, image_file in enumerate(selected_images, 1):
            source = images_root / image_file
            dest = person_dir / f"{idx:02d}.jpg"

            if source.exists():
                shutil.copy2(source, dest)
                total_copied += 1
            else:
                print(f"  WARNING: Image not found: {source}")

        print(f"  Person {person_id:02d} ({person:<25}): {len(selected_images)} images copied")

    print("\n" + "=" * 70)
    print("TEST SUBSET CREATED SUCCESSFULLY")
    print("=" * 70)
    print(f"Location: {test_root.absolute()}")
    print(f"Individuals: {len(images_by_person)}")
    print(f"Total images: {total_copied}")
    print(f"Average images per person: {total_copied / len(images_by_person):.1f}")
    print("\nTest subset structure:")
    print("  data/test/")
    for person_dir in sorted(test_root.iterdir()):
        if person_dir.is_dir():
            num_images = len(list(person_dir.glob("*.jpg")))
            print(f"    {person_dir.name}/ ({num_images} images)")

    print("\n" + "=" * 70)
    print("Next step: Run baseline tests")
    print("  python scripts/run_baseline_tests.py")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    import sys
    random.seed(42)  # For reproducibility
    sys.exit(create_test_subset())
