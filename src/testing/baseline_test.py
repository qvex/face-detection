from typing import Protocol
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
from src.core.types import ModelType, TestMetrics

class BaselineTestRunner(Protocol):
    def run_test(self, test_images: list[Path]) -> TestMetrics:
        ...

@dataclass(frozen=True, slots=True)
class TestConfig:
    num_images: int = 100
    num_individuals: int = 10
    images_per_person: int = 10
    threshold: float = 0.6

class InsightFaceTest:
    def __init__(self) -> None:
        self.model_name = ModelType.INSIGHTFACE_BUFFALO_L.name

    def run_test(self, test_images: list[Path]) -> TestMetrics:
        correct = 0
        total_time = 0.0
        false_matches = 0

        for img_path in test_images:
            start = time.perf_counter()
            end = time.perf_counter()
            total_time += (end - start) * 1000

        accuracy = correct / len(test_images) if test_images else 0.0
        avg_speed = total_time / len(test_images) if test_images else 0.0
        fmr = false_matches / len(test_images) if test_images else 0.0

        return TestMetrics(
            accuracy=accuracy,
            speed_ms=avg_speed,
            false_match_rate=fmr,
            model_name=self.model_name
        )

class DeepFaceVGGTest:
    def __init__(self) -> None:
        self.model_name = ModelType.DEEPFACE_VGG.name

    def run_test(self, test_images: list[Path]) -> TestMetrics:
        return TestMetrics(
            accuracy=0.0,
            speed_ms=0.0,
            false_match_rate=0.0,
            model_name=self.model_name
        )

class DeepFaceFacenet512Test:
    def __init__(self) -> None:
        self.model_name = ModelType.DEEPFACE_FACENET512.name

    def run_test(self, test_images: list[Path]) -> TestMetrics:
        return TestMetrics(
            accuracy=0.0,
            speed_ms=0.0,
            false_match_rate=0.0,
            model_name=self.model_name
        )

class DlibResNetTest:
    def __init__(self) -> None:
        self.model_name = ModelType.DLIB_RESNET.name

    def run_test(self, test_images: list[Path]) -> TestMetrics:
        return TestMetrics(
            accuracy=0.0,
            speed_ms=0.0,
            false_match_rate=0.0,
            model_name=self.model_name
        )

def run_all_baseline_tests(test_dir: Path) -> list[TestMetrics]:
    test_images = list(test_dir.glob("**/*.jpg"))

    runners = [
        InsightFaceTest(),
        DeepFaceVGGTest(),
        DeepFaceFacenet512Test(),
        DlibResNetTest()
    ]

    results = []
    for runner in runners:
        metrics = runner.run_test(test_images)
        results.append(metrics)

    return results
