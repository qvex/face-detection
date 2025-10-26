from typing import Optional
import numpy as np
import cv2

from src.models.detector import FaceDetector, FaceDetection, BoundingBox
from src.core.types import Result, Success, Failure
from src.core.errors import DetectionError, DetectionErrorKind

class MTCNNDetector:
    def __init__(self, min_face_size: int = 40, confidence_threshold: float = 0.9):
        self._min_face_size = min_face_size
        self._confidence_threshold = confidence_threshold
        self._detector: Optional[object] = None
        self._init_result = self._initialize_model()

    def _initialize_model(self) -> Result[None, DetectionError]:
        from mtcnn import MTCNN
        self._detector = MTCNN(
            min_face_size=self._min_face_size,
            scale_factor=0.709,
            steps_threshold=[0.6, 0.7, 0.7]
        )
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

    def _convert_to_rgb(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _extract_largest_face(
        self,
        detections: list[dict]
    ) -> Result[dict, DetectionError]:
        if len(detections) == 0:
            return Failure(DetectionError(
                kind=DetectionErrorKind.NO_FACE_DETECTED,
                details="no faces found in image"
            ))

        largest = max(detections, key=lambda d: d['box'][2] * d['box'][3])
        return Success(largest)

    def _create_face_detection(
        self,
        detection: dict,
        image: np.ndarray
    ) -> FaceDetection:
        x, y, w, h = detection['box']
        confidence = detection['confidence']

        bbox = BoundingBox(
            x1=max(0, x),
            y1=max(0, y),
            x2=min(image.shape[1], x + w),
            y2=min(image.shape[0], y + h),
            confidence=confidence
        )

        landmarks = np.array([
            detection['keypoints']['left_eye'],
            detection['keypoints']['right_eye'],
            detection['keypoints']['nose'],
            detection['keypoints']['mouth_left'],
            detection['keypoints']['mouth_right']
        ], dtype=np.float32)

        face_region = image[bbox.y1:bbox.y2, bbox.x1:bbox.x2]

        return FaceDetection(
            bbox=bbox,
            landmarks=landmarks,
            aligned_face=face_region
        )

    def detect(self, image: np.ndarray) -> Result[FaceDetection, DetectionError]:
        validation = self._validate_image(image)
        if isinstance(validation, Failure):
            return validation

        rgb_image = self._convert_to_rgb(image)
        detections = self._detector.detect_faces(rgb_image)

        largest_face_result = self._extract_largest_face(detections)
        if isinstance(largest_face_result, Failure):
            return largest_face_result

        detection_dict = largest_face_result.value

        if detection_dict['confidence'] < self._confidence_threshold:
            return Failure(DetectionError(
                kind=DetectionErrorKind.LOW_CONFIDENCE,
                details=f"confidence {detection_dict['confidence']:.2f} below threshold {self._confidence_threshold}"
            ))

        face_detection = self._create_face_detection(detection_dict, image)
        return Success(face_detection)

def create_mtcnn_detector(
    min_face_size: int = 40,
    confidence_threshold: float = 0.9
) -> MTCNNDetector:
    return MTCNNDetector(min_face_size, confidence_threshold)
