from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2

from src.models.detector import FaceDetection
from src.core.types import Result, Success, Failure
from src.core.errors import DetectionError, EmbeddingError, DetectionErrorKind

@dataclass(frozen=True, slots=True)
class ReferenceImage:
    embedding: np.ndarray
    original_image: np.ndarray
    face_detection: FaceDetection
    is_admit_card: bool

class ReferenceImageProcessor:
    def __init__(self, detector, recognizer):
        self._detector = detector
        self._recognizer = recognizer

    def _load_image(self, image_path: Path) -> Result[np.ndarray, DetectionError]:
        if not image_path.exists():
            return Failure(DetectionError(
                kind=DetectionErrorKind.INVALID_IMAGE,
                details=f"file not found: {image_path}"
            ))

        image = cv2.imread(str(image_path))

        if image is None:
            return Failure(DetectionError(
                kind=DetectionErrorKind.INVALID_IMAGE,
                details=f"failed to load image: {image_path}"
            ))

        return Success(image)

    def _is_admit_card_format(
        self,
        face_detection: FaceDetection,
        image_shape: tuple
    ) -> bool:
        image_height, image_width = image_shape[:2]
        image_area = image_height * image_width

        bbox = face_detection.bbox
        face_width = bbox.x2 - bbox.x1
        face_height = bbox.y2 - bbox.y1
        face_area = face_width * face_height

        face_ratio = face_area / image_area

        return face_ratio < 0.5

    def process_reference(
        self,
        image_path: Path
    ) -> Result[ReferenceImage, DetectionError | EmbeddingError]:
        load_result = self._load_image(image_path)
        if isinstance(load_result, Failure):
            return load_result

        image = load_result.value

        detection_result = self._detector.detect(image)
        if isinstance(detection_result, Failure):
            return detection_result

        face_detection = detection_result.value

        embedding_result = self._recognizer.extract_embedding(
            face_detection.aligned_face
        )
        if isinstance(embedding_result, Failure):
            return embedding_result

        embedding = embedding_result.value

        is_admit_card = self._is_admit_card_format(face_detection, image.shape)

        reference = ReferenceImage(
            embedding=embedding,
            original_image=image,
            face_detection=face_detection,
            is_admit_card=is_admit_card
        )

        return Success(reference)

def create_reference_processor(detector, recognizer) -> ReferenceImageProcessor:
    return ReferenceImageProcessor(detector, recognizer)
