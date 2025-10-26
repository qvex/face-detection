from typing import Optional
import numpy as np
import cv2

from src.models.detector import FaceDetector, FaceDetection, BoundingBox
from src.core.types import Result, Success, Failure
from src.core.errors import DetectionError, DetectionErrorKind

class InsightFaceDetector:
    def __init__(self, model_name: str = 'buffalo_l', confidence_threshold: float = 0.5):
        self._model_name = model_name
        self._confidence_threshold = confidence_threshold
        self._app: Optional[object] = None
        self._init_result = self._initialize_model()

    def _initialize_model(self) -> Result[None, DetectionError]:
        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(
            name=self._model_name,
            providers=['CPUExecutionProvider']
        )
        self._app.prepare(ctx_id=-1, det_size=(640, 640))

        return Success(None)

    def _validate_image(self, image: np.ndarray) -> Result[None, DetectionError]:
        if image is None or image.size == 0:
            return Failure(DetectionError(
                kind=DetectionErrorKind.INVALID_IMAGE,
                details="empty or null image provided"
            ))

        if len(image.shape) != 3 or image.shape[2] != 3:
            return Failure(DetectionError(
                kind=DetectionErrorKind.INVALID_IMAGE,
                details=f"expected 3-channel image, got shape {image.shape}"
            ))

        return Success(None)

    def _extract_largest_face(
        self,
        faces: list
    ) -> Result[object, DetectionError]:
        if len(faces) == 0:
            return Failure(DetectionError(
                kind=DetectionErrorKind.NO_FACE_DETECTED,
                details="no faces found in image"
            ))

        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return Success(largest)

    def _create_face_detection(
        self,
        face: object,
        image: np.ndarray
    ) -> FaceDetection:
        x1, y1, x2, y2 = face.bbox
        det_score = face.det_score

        bbox = BoundingBox(
            x1=int(max(0, x1)),
            y1=int(max(0, y1)),
            x2=int(min(image.shape[1], x2)),
            y2=int(min(image.shape[0], y2)),
            confidence=float(det_score)
        )

        landmarks = face.kps.astype(np.float32)

        return FaceDetection(
            bbox=bbox,
            landmarks=landmarks,
            aligned_face=image
        )

    def detect(self, image: np.ndarray) -> Result[FaceDetection, DetectionError]:
        validation = self._validate_image(image)
        if isinstance(validation, Failure):
            return validation

        faces = self._app.get(image)

        largest_face_result = self._extract_largest_face(faces)
        if isinstance(largest_face_result, Failure):
            return largest_face_result

        face = largest_face_result.value

        if face.det_score < self._confidence_threshold:
            return Failure(DetectionError(
                kind=DetectionErrorKind.LOW_CONFIDENCE,
                details=f"confidence {face.det_score:.2f} below threshold {self._confidence_threshold}"
            ))

        face_detection = self._create_face_detection(face, image)
        return Success(face_detection)

def create_insightface_detector(
    model_name: str = 'buffalo_l',
    confidence_threshold: float = 0.5
) -> InsightFaceDetector:
    return InsightFaceDetector(model_name, confidence_threshold)
