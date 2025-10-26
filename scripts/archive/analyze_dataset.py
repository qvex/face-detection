#!/usr/bin/env python3
"""
Analyze the FRD dataset to identify Indian celebrities and their image counts.
"""
import pandas as pd
from pathlib import Path
from collections import Counter

def main():
    dataset_root = Path("dataset/frd")
    csv_path = dataset_root / "Dataset.csv"

    # Read the CSV
    df = pd.read_csv(csv_path)

    print("=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)
    print(f"Total images: {len(df)}")
    print(f"Unique individuals: {df['label'].nunique()}")

    # Count images per person
    person_counts = df['label'].value_counts()

    # Known Indian celebrities in the dataset
    indian_celebrities = [
        'Hrithik Roshan', 'Vijay Deverakonda', 'Alia Bhatt', 'Priyanka Chopra',
        'Anushka Sharma', 'Deepika Padukone', 'Katrina Kaif', 'Ranveer Singh',
        'Ranbir Kapoor', 'Virat Kohli', 'Shah Rukh Khan', 'Salman Khan',
        'Aamir Khan', 'Akshay Kumar', 'Kareena Kapoor', 'Aishwarya Rai',
        'Madhuri Dixit', 'Amitabh Bachchan', 'Sachin Tendulkar', 'MS Dhoni',
        'Sania Mirza', 'PV Sindhu', 'Disha Patani', 'Shraddha Kapoor',
        'Varun Dhawan', 'Tiger Shroff', 'Sidharth Malhotra', 'Kiara Advani',
        'Rajkummar Rao', 'Ayushmann Khurrana', 'Vicky Kaushal', 'Sara Ali Khan',
        'Janhvi Kapoor', 'Ananya Panday', 'Kartik Aaryan', 'Sushant Singh Rajput',
        'Kangana Ranaut', 'Taapsee Pannu', 'Bhumi Pednekar', 'Nawazuddin Siddiqui'
    ]

    # Filter for Indian celebrities
    indian_mask = df['label'].isin(indian_celebrities)
    indian_df = df[indian_mask]
    indian_counts = indian_df['label'].value_counts()

    print("\n" + "=" * 70)
    print("INDIAN CELEBRITIES IN DATASET")
    print("=" * 70)
    print(f"Total Indian celebrity images: {len(indian_df)}")
    print(f"Unique Indian celebrities: {len(indian_counts)}")

    print("\nIndian celebrities with 10+ images (suitable for baseline testing):")
    print("-" * 70)
    suitable_for_testing = indian_counts[indian_counts >= 10]
    for person, count in suitable_for_testing.items():
        print(f"  {person:<30} {count:>3} images")

    print("\n" + "=" * 70)
    print(f"Total individuals with 10+ images: {len(suitable_for_testing)}")
    print("=" * 70)

    # Show all persons (to identify more Indian celebrities we might have missed)
    print("\nALL INDIVIDUALS IN DATASET (checking for missed Indian names):")
    print("-" * 70)
    for person, count in person_counts.items():
        marker = "[INDIAN]" if person in indian_celebrities else ""
        print(f"  {person:<30} {count:>3} images {marker}")

if __name__ == "__main__":
    main()
