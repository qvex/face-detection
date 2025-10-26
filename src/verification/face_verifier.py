from dataclasses import dataclass
import numpy as np

from src.core.types import Result, Success, Failure
from src.core.errors import VerificationError, VerificationErrorKind

@dataclass(frozen=True, slots=True)
class VerificationResult:
    is_match: bool
    similarity_score: float
    threshold: float

class FaceVerifier:
    def __init__(self, threshold: float = 0.4):
        self._threshold = threshold

    def _validate_embedding_dimensions(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> Result[None, VerificationError]:
        if embedding1.shape != embedding2.shape:
            return Failure(VerificationError(
                kind=VerificationErrorKind.INVALID_EMBEDDING_DIMENSIONS,
                details=f"embedding dimension mismatch: {embedding1.shape} vs {embedding2.shape}"
            ))

        if embedding1.ndim != 1:
            return Failure(VerificationError(
                kind=VerificationErrorKind.INVALID_EMBEDDING_DIMENSIONS,
                details=f"expected 1D embeddings, got {embedding1.ndim}D"
            ))

        return Success(None)

    def _compute_cosine_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        return float(np.dot(embedding1, embedding2))

    def verify(
        self,
        reference_embedding: np.ndarray,
        test_embedding: np.ndarray
    ) -> Result[VerificationResult, VerificationError]:
        validation = self._validate_embedding_dimensions(
            reference_embedding,
            test_embedding
        )
        if isinstance(validation, Failure):
            return validation

        similarity = self._compute_cosine_similarity(
            reference_embedding,
            test_embedding
        )

        is_match = similarity >= self._threshold

        result = VerificationResult(
            is_match=is_match,
            similarity_score=similarity,
            threshold=self._threshold
        )

        return Success(result)

def create_face_verifier(threshold: float = 0.4) -> FaceVerifier:
    return FaceVerifier(threshold)
