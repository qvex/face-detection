from typing import Protocol, TypeVar
from dataclasses import dataclass
import numpy as np

T = TypeVar('T')

@dataclass(frozen=True, slots=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

@dataclass(frozen=True, slots=True)
class FaceDetection:
    bbox: BoundingBox
    landmarks: np.ndarray
    aligned_face: np.ndarray

class FaceDetector(Protocol):
    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        ...

class FaceRecognizer(Protocol):
    def extract_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        ...
