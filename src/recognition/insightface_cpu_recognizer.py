from typing import Optional
import numpy as np
import cv2

from src.models.detector import FaceRecognizer
from src.core.types import Result, Success, Failure
from src.core.errors import EmbeddingError, EmbeddingErrorKind

class InsightFaceCPURecognizer:
    def __init__(self, model_name: str = 'buffalo_l'):
        self._model_name = model_name
        self._app: Optional[object] = None
        self._init_result = self._initialize_model()

    def _initialize_model(self) -> Result[None, EmbeddingError]:
        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(
            name=self._model_name,
            providers=['CPUExecutionProvider']
        )
        self._app.prepare(ctx_id=-1, det_size=(640, 640))

        return Success(None)

    def _validate_face_region(self, face: np.ndarray) -> Result[None, EmbeddingError]:
        if face is None or face.size == 0:
            return Failure(EmbeddingError(
                kind=EmbeddingErrorKind.INVALID_FACE_REGION,
                details="empty or null face region provided"
            ))

        if len(face.shape) != 3:
            return Failure(EmbeddingError(
                kind=EmbeddingErrorKind.INVALID_FACE_REGION,
                details=f"expected 3D face region, got shape {face.shape}"
            ))

        min_size = 20
        if face.shape[0] < min_size or face.shape[1] < min_size:
            return Failure(EmbeddingError(
                kind=EmbeddingErrorKind.INVALID_FACE_REGION,
                details=f"face region too small: {face.shape[0]}x{face.shape[1]}, minimum {min_size}x{min_size}"
            ))

        return Success(None)

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding

    def extract_embedding(
        self,
        aligned_face: np.ndarray
    ) -> Result[np.ndarray, EmbeddingError]:
        validation = self._validate_face_region(aligned_face)
        if isinstance(validation, Failure):
            return validation

        faces = self._app.get(aligned_face)

        if len(faces) == 0:
            return Failure(EmbeddingError(
                kind=EmbeddingErrorKind.EXTRACTION_FAILED,
                details="no face detected in provided region"
            ))

        raw_embedding = faces[0].embedding

        if raw_embedding.ndim == 2 and raw_embedding.shape[0] == 1:
            raw_embedding = raw_embedding.flatten()

        normalized_embedding = self._normalize_embedding(raw_embedding)

        return Success(normalized_embedding)

def create_insightface_cpu_recognizer(
    model_name: str = 'buffalo_l'
) -> InsightFaceCPURecognizer:
    return InsightFaceCPURecognizer(model_name)
