#!/usr/bin/env python3
"""
Run baseline tests on InsightFace using pre-cropped face images with GPU acceleration.
Dataset: 8 Indian celebrities × 10 images each = 80 total images.
"""
import os

# Add CUDA 12.4 and cuDNN to PATH BEFORE importing CUDA libraries
cuda_12_4_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
cudnn_bin = r"C:\Program Files\NVIDIA\CUDNN\v9.14\bin\12.9"
current_path = os.environ.get('PATH', '')
os.environ['PATH'] = f"{cuda_12_4_bin};{cudnn_bin};{current_path}"
os.environ['CUDA_PATH'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"

from pathlib import Path
from dataclasses import dataclass
import time
import cv2
import numpy as np
from collections import defaultdict


@dataclass(frozen=True, slots=True)
class BaselineTestResults:
    model_name: str
    accuracy: float
    avg_speed_ms: float
    false_match_rate: float
    false_non_match_rate: float
    total_comparisons: int
    correct_matches: int
    gpu_enabled: bool


def load_test_images(test_dir: Path) -> dict[str, list[Path]]:
    """Load test images organized by person."""
    images_by_person = {}

    for person_dir in sorted(test_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        person_id = person_dir.name
        image_files = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
        images_by_person[person_id] = sorted(image_files)

    return images_by_person


def test_insightface_recognition_model(
    images_by_person: dict[str, list[Path]],
    threshold: float = 0.4
) -> BaselineTestResults:
    """
    Test InsightFace recognition model on pre-cropped face images.
    Uses direct embedding extraction without face detection.
    """
    print("\n" + "=" * 70)
    print("Testing InsightFace Buffalo_L (Recognition Model Only)")
    print("=" * 70)
    print("Mode: Direct embedding extraction from pre-cropped faces")
    print(f"Similarity threshold: {threshold}")
    print("=" * 70)

    try:
        from insightface.model_zoo import get_model

        # Load only the recognition model (ArcFace ResNet50)
        model = get_model('buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        model.prepare(ctx_id=0)

        # Check if GPU is being used
        gpu_enabled = False
        if hasattr(model, 'session'):
            providers = model.session.get_providers()
            gpu_enabled = 'CUDAExecutionProvider' in providers
            provider_str = providers[0] if providers else "Unknown"
            print(f"\nModel provider: {provider_str}")
            if gpu_enabled:
                print("[GPU ENABLED] Using CUDA 12.4 acceleration")
            else:
                print("[CPU MODE] Running on CPU")

    except Exception as e:
        print(f"Failed to load InsightFace recognition model: {e}")
        return BaselineTestResults("InsightFace_Buffalo_L", 0.0, 0.0, 1.0, 1.0, 0, 0, False)

    embeddings_db = {}
    enrollment_times = []
    verification_times = []

    # Metrics
    correct_matches = 0
    false_matches = 0  # Different persons matched incorrectly
    false_non_matches = 0  # Same person not matched
    total_same_person_comparisons = 0
    total_different_person_comparisons = 0

    # Phase 1: Enrollment - Extract embeddings for first image of each person
    print(f"\nPhase 1: Enrollment (processing first image per person)")
    print("-" * 70)

    for person_id, image_paths in sorted(images_by_person.items()):
        img_path = image_paths[0]
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"  {person_id}: Failed to load image")
            continue

        # Resize to model input size (112x112 for ArcFace)
        img_resized = cv2.resize(img, (112, 112))

        try:
            start = time.perf_counter()
            embedding = model.get_feat(img_resized)
            elapsed = (time.perf_counter() - start) * 1000

            # Flatten embedding from (1, 512) to (512,)
            embedding = embedding.flatten()

            embeddings_db[person_id] = embedding
            enrollment_times.append(elapsed)
            print(f"  {person_id}: Enrolled ({elapsed:.2f}ms, embedding shape: {embedding.shape})")

        except Exception as e:
            print(f"  {person_id}: Failed to extract embedding - {str(e)[:50]}")
            continue

    print(f"\nEnrolled {len(embeddings_db)} identities")

    # Phase 2: Verification - Test remaining images
    print(f"\nPhase 2: Verification (testing remaining images)")
    print("-" * 70)

    for person_id, image_paths in sorted(images_by_person.items()):
        if person_id not in embeddings_db:
            continue

        person_results = []

        # Test images 2-10 for this person
        for img_idx, img_path in enumerate(image_paths[1:], start=2):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img_resized = cv2.resize(img, (112, 112))

            try:
                start = time.perf_counter()
                test_embedding = model.get_feat(img_resized)
                elapsed = (time.perf_counter() - start) * 1000
                verification_times.append(elapsed)

                # Flatten embedding from (1, 512) to (512,)
                test_embedding = test_embedding.flatten()

                # Compare with all enrolled identities
                best_match_id = None
                best_similarity = -1.0

                for enrolled_id, enrolled_embedding in embeddings_db.items():
                    # Cosine similarity
                    similarity = np.dot(test_embedding, enrolled_embedding) / (
                        np.linalg.norm(test_embedding) * np.linalg.norm(enrolled_embedding)
                    )

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match_id = enrolled_id

                # Evaluate result
                if best_match_id == person_id:
                    # Same person comparison
                    total_same_person_comparisons += 1
                    if best_similarity > threshold:
                        correct_matches += 1
                        person_results.append(f"[OK] img{img_idx:02d}: {best_similarity:.3f}")
                    else:
                        false_non_matches += 1
                        person_results.append(f"[FAIL] img{img_idx:02d}: {best_similarity:.3f} (below threshold)")
                else:
                    # Matched wrong person
                    total_different_person_comparisons += 1
                    if best_similarity > threshold:
                        false_matches += 1
                        person_results.append(f"[FAIL] img{img_idx:02d}: {best_similarity:.3f} (wrong: {best_match_id})")
                    else:
                        # Correctly rejected as different person
                        person_results.append(f"[OK] img{img_idx:02d}: {best_similarity:.3f} (correctly rejected)")

            except Exception as e:
                false_non_matches += 1
                total_same_person_comparisons += 1
                person_results.append(f"[ERROR] img{img_idx:02d}: Error - {str(e)[:30]}")

        # Print results for this person
        print(f"  {person_id}:")
        for result in person_results:
            print(f"    {result}")

    # Calculate metrics
    total_comparisons = total_same_person_comparisons + total_different_person_comparisons
    accuracy = correct_matches / total_same_person_comparisons if total_same_person_comparisons > 0 else 0.0
    fmr = false_matches / total_comparisons if total_comparisons > 0 else 0.0
    fnmr = false_non_matches / total_same_person_comparisons if total_same_person_comparisons > 0 else 0.0

    all_times = enrollment_times + verification_times
    avg_speed = np.mean(all_times) if all_times else 0.0

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total same-person comparisons: {total_same_person_comparisons}")
    print(f"Correct matches: {correct_matches}")
    print(f"False non-matches (should match but didn't): {false_non_matches}")
    print(f"False matches (wrong person matched): {false_matches}")
    print(f"\nAccuracy (True Accept Rate): {accuracy*100:.1f}%")
    print(f"False Match Rate (FAR): {fmr*100:.2f}%")
    print(f"False Non-Match Rate (FRR): {fnmr*100:.2f}%")
    print(f"Average speed: {avg_speed:.2f}ms per image")
    print(f"Enrollment speed: {np.mean(enrollment_times):.2f}ms avg")
    print(f"Verification speed: {np.mean(verification_times):.2f}ms avg")
    print("=" * 70)

    return BaselineTestResults(
        model_name="InsightFace_Buffalo_L_Recognition",
        accuracy=accuracy,
        avg_speed_ms=avg_speed,
        false_match_rate=fmr,
        false_non_match_rate=fnmr,
        total_comparisons=total_comparisons,
        correct_matches=correct_matches,
        gpu_enabled=gpu_enabled
    )


def main():
    """Run baseline test on InsightFace with GPU."""
    test_path = Path("./data/test")

    if not test_path.exists():
        print(f"ERROR: Test directory not found: {test_path}")
        print("Run 'python scripts/create_test_subset_frd.py' first")
        return 1

    print("\n" + "=" * 70)
    print("BASELINE TEST: InsightFace on Indian Celebrity Faces (GPU)")
    print("=" * 70)
    print(f"Test directory: {test_path.absolute()}")

    images_by_person = load_test_images(test_path)
    num_persons = len(images_by_person)
    total_images = sum(len(imgs) for imgs in images_by_person.values())

    print(f"\nDataset loaded:")
    print(f"  Individuals: {num_persons}")
    print(f"  Total images: {total_images}")
    print(f"  Enrollment images: {num_persons}")
    print(f"  Verification images: {total_images - num_persons}")

    # Test InsightFace with different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6]
    results = []

    for threshold in thresholds:
        print(f"\n{'='*70}")
        print(f"Testing with threshold: {threshold}")
        print(f"{'='*70}")
        result = test_insightface_recognition_model(images_by_person, threshold=threshold)
        results.append((threshold, result))

    # Print comparison
    print("\n" + "=" * 70)
    print("THRESHOLD COMPARISON")
    print("=" * 70)
    print(f"{'Threshold':<12} {'Accuracy':>10} {'FAR':>8} {'FRR':>8} {'Speed':>10}")
    print("-" * 70)

    for threshold, result in results:
        print(f"{threshold:<12.2f} {result.accuracy*100:>9.1f}% {result.false_match_rate*100:>7.2f}% "
              f"{result.false_non_match_rate*100:>7.1f}% {result.avg_speed_ms:>9.2f}ms")

    print("=" * 70)

    # Find best threshold
    best_result = max(results, key=lambda r: r[1].accuracy)
    print(f"\nBest threshold: {best_result[0]} with {best_result[1].accuracy*100:.1f}% accuracy")

    # Check if meets targets
    print("\n" + "=" * 70)
    print("TARGET COMPARISON")
    print("=" * 70)
    print(f"Target Accuracy: 95%+")
    print(f"Target Speed: <100ms")
    print(f"Target FAR: <0.1%")
    print()
    print(f"Achieved Accuracy: {best_result[1].accuracy*100:.1f}%")
    print(f"Achieved Speed: {best_result[1].avg_speed_ms:.2f}ms")
    print(f"Achieved FAR: {best_result[1].false_match_rate*100:.2f}%")
    print(f"GPU Enabled: {'Yes' if best_result[1].gpu_enabled else 'No'}")

    meets_acc = best_result[1].accuracy >= 0.95
    meets_speed = best_result[1].avg_speed_ms < 100
    meets_far = best_result[1].false_match_rate < 0.001

    print()
    if meets_acc and meets_speed and meets_far:
        print("[SUCCESS] All targets met!")
    else:
        print("[PARTIAL] Some targets not met - fine-tuning recommended")
        if not meets_acc:
            print("  - Accuracy below 95%")
        if not meets_speed:
            print("  - Speed above 100ms")
        if not meets_far:
            print("  - FAR above 0.1%")

    print("=" * 70)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
