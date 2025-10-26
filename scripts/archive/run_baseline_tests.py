from pathlib import Path
from dataclasses import dataclass
import time
import cv2
import numpy as np
from typing import Optional
from src.core.types import ModelType, TestMetrics

@dataclass(frozen=True, slots=True)
class BaselineTestResults:
    model_name: str
    accuracy: float
    avg_speed_ms: float
    false_match_rate: float
    total_images: int
    correct_matches: int

def load_test_images(test_dir: Path) -> dict[str, list[Path]]:
    images_by_person = {}

    for person_dir in sorted(test_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        person_id = person_dir.name
        image_files = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
        images_by_person[person_id] = image_files

    return images_by_person

def test_insightface_buffalo_l(
    images_by_person: dict[str, list[Path]],
    threshold: float = 0.4
) -> BaselineTestResults:
    print("\n" + "=" * 70)
    print("Testing InsightFace Buffalo_L Model")
    print("=" * 70)

    try:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception as e:
        print(f"Failed to load InsightFace: {e}")
        return BaselineTestResults("InsightFace_Buffalo_L", 0.0, 0.0, 1.0, 0, 0)

    embeddings_db = {}
    correct = 0
    total = 0
    false_matches = 0
    total_time = 0.0

    print("Phase 1: Enrollment (processing first image per person)")
    for person_id, image_paths in images_by_person.items():
        img_path = image_paths[0]
        img = cv2.imread(str(img_path))

        if img is None:
            continue

        start = time.perf_counter()
        faces = app.get(img)
        elapsed = (time.perf_counter() - start) * 1000

        if len(faces) > 0:
            embeddings_db[person_id] = faces[0].embedding
            total_time += elapsed
            print(f"  {person_id}: Enrolled ({elapsed:.2f}ms)")

    print(f"\nPhase 2: Verification (testing remaining images)")
    for person_id, image_paths in images_by_person.items():
        if person_id not in embeddings_db:
            continue

        for img_path in image_paths[1:]:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            start = time.perf_counter()
            faces = app.get(img)
            elapsed = (time.perf_counter() - start) * 1000
            total_time += elapsed
            total += 1

            if len(faces) == 0:
                false_matches += 1
                continue

            test_embedding = faces[0].embedding

            best_match = None
            best_similarity = -1

            for enrolled_id, enrolled_embedding in embeddings_db.items():
                similarity = np.dot(test_embedding, enrolled_embedding) / (
                    np.linalg.norm(test_embedding) * np.linalg.norm(enrolled_embedding)
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = enrolled_id

            if best_match == person_id and best_similarity > threshold:
                correct += 1
            elif best_match != person_id and best_similarity > threshold:
                false_matches += 1

    accuracy = correct / total if total > 0 else 0.0
    avg_speed = total_time / (total + len(embeddings_db)) if (total + len(embeddings_db)) > 0 else 0.0
    fmr = false_matches / total if total > 0 else 0.0

    print(f"\nResults: {correct}/{total} correct ({accuracy*100:.1f}%)")
    print(f"Average speed: {avg_speed:.2f}ms per face")
    print(f"False match rate: {fmr*100:.1f}%")

    return BaselineTestResults(
        model_name="InsightFace_Buffalo_L",
        accuracy=accuracy,
        avg_speed_ms=avg_speed,
        false_match_rate=fmr,
        total_images=total,
        correct_matches=correct
    )

def test_deepface_model(
    images_by_person: dict[str, list[Path]],
    model_name: str,
    threshold: float = 0.6
) -> BaselineTestResults:
    print("\n" + "=" * 70)
    print(f"Testing DeepFace {model_name} Model")
    print("=" * 70)

    try:
        from deepface import DeepFace
    except Exception as e:
        print(f"Failed to load DeepFace: {e}")
        return BaselineTestResults(f"DeepFace_{model_name}", 0.0, 0.0, 1.0, 0, 0)

    embeddings_db = {}
    correct = 0
    total = 0
    false_matches = 0
    total_time = 0.0

    print("Phase 1: Enrollment (processing first image per person)")
    for person_id, image_paths in images_by_person.items():
        img_path = str(image_paths[0])

        try:
            start = time.perf_counter()
            embedding = DeepFace.represent(
                img_path=img_path,
                model_name=model_name,
                enforce_detection=False
            )[0]["embedding"]
            elapsed = (time.perf_counter() - start) * 1000

            embeddings_db[person_id] = np.array(embedding)
            total_time += elapsed
            print(f"  {person_id}: Enrolled ({elapsed:.2f}ms)")
        except Exception as e:
            print(f"  {person_id}: Failed to enroll - {str(e)}")
            continue

    print(f"\nPhase 2: Verification (testing remaining images)")
    for person_id, image_paths in images_by_person.items():
        if person_id not in embeddings_db:
            continue

        for img_path in image_paths[1:]:
            try:
                start = time.perf_counter()
                embedding = DeepFace.represent(
                    img_path=str(img_path),
                    model_name=model_name,
                    enforce_detection=False
                )[0]["embedding"]
                elapsed = (time.perf_counter() - start) * 1000
                total_time += elapsed
                total += 1

                test_embedding = np.array(embedding)

                best_match = None
                best_distance = float('inf')

                for enrolled_id, enrolled_embedding in embeddings_db.items():
                    distance = np.linalg.norm(test_embedding - enrolled_embedding)

                    if distance < best_distance:
                        best_distance = distance
                        best_match = enrolled_id

                similarity = 1 / (1 + best_distance)

                if best_match == person_id and similarity > threshold:
                    correct += 1
                elif best_match != person_id and similarity > threshold:
                    false_matches += 1

            except Exception:
                total += 1
                false_matches += 1
                continue

    accuracy = correct / total if total > 0 else 0.0
    avg_speed = total_time / (total + len(embeddings_db)) if (total + len(embeddings_db)) > 0 else 0.0
    fmr = false_matches / total if total > 0 else 0.0

    print(f"\nResults: {correct}/{total} correct ({accuracy*100:.1f}%)")
    print(f"Average speed: {avg_speed:.2f}ms per face")
    print(f"False match rate: {fmr*100:.1f}%")

    return BaselineTestResults(
        model_name=f"DeepFace_{model_name}",
        accuracy=accuracy,
        avg_speed_ms=avg_speed,
        false_match_rate=fmr,
        total_images=total,
        correct_matches=correct
    )

def run_all_baseline_tests(test_dir: Optional[Path] = None) -> list[BaselineTestResults]:
    test_path = test_dir if test_dir else Path("./data/test")

    if not test_path.exists():
        print(f"ERROR: Test directory not found: {test_path}")
        print("Run 'python scripts/create_test_subset.py' first")
        return []

    print("\n" + "=" * 70)
    print("BASELINE MODEL TESTING - INDIAN FACES")
    print("=" * 70)
    print(f"Test directory: {test_path}")

    images_by_person = load_test_images(test_path)
    num_persons = len(images_by_person)
    total_images = sum(len(imgs) for imgs in images_by_person.values())

    print(f"Loaded: {num_persons} individuals, {total_images} total images")
    print("=" * 70)

    results = []

    results.append(test_insightface_buffalo_l(images_by_person))

    results.append(test_deepface_model(images_by_person, "VGG-Face"))

    results.append(test_deepface_model(images_by_person, "Facenet512"))

    results.append(test_deepface_model(images_by_person, "Dlib"))

    print("\n" + "=" * 70)
    print("BASELINE TEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"{"Model":25} {"Accuracy":>10} {"Speed (ms)":>12} {"FMR":>8}")
    print("-" * 70)

    for result in results:
        print(f"{result.model_name:25} {result.accuracy*100:>9.1f}% {result.avg_speed_ms:>11.2f} {result.false_match_rate*100:>7.1f}%")

    print("=" * 70)

    best_accuracy = max(results, key=lambda r: r.accuracy)
    fastest = min(results, key=lambda r: r.avg_speed_ms)

    print(f"\nBest Accuracy: {best_accuracy.model_name} ({best_accuracy.accuracy*100:.1f}%)")
    print(f"Fastest Model: {fastest.model_name} ({fastest.avg_speed_ms:.2f}ms)")

    print("\nNext step: Choose best model and proceed to Phase 3 (Dataset Acquisition)")
    print("=" * 70)

    return results

if __name__ == "__main__":
    run_all_baseline_tests()
