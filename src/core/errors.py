from dataclasses import dataclass
from enum import Enum, auto
from typing import Union

class DetectionErrorKind(Enum):
    NO_FACE_DETECTED = auto()
    MULTIPLE_FACES = auto()
    INVALID_IMAGE = auto()
    LOW_CONFIDENCE = auto()

class EmbeddingErrorKind(Enum):
    EXTRACTION_FAILED = auto()
    INVALID_FACE_REGION = auto()
    MODEL_LOAD_FAILED = auto()

class VerificationErrorKind(Enum):
    SIMILARITY_COMPUTATION_FAILED = auto()
    INVALID_EMBEDDING_DIMENSIONS = auto()

class SessionErrorKind(Enum):
    INVALID_STATE_TRANSITION = auto()
    MISSING_REFERENCE_EMBEDDING = auto()
    SESSION_EXPIRED = auto()

@dataclass(frozen=True, slots=True)
class DetectionError:
    kind: DetectionErrorKind
    details: str

@dataclass(frozen=True, slots=True)
class EmbeddingError:
    kind: EmbeddingErrorKind
    details: str

@dataclass(frozen=True, slots=True)
class VerificationError:
    kind: VerificationErrorKind
    details: str

@dataclass(frozen=True, slots=True)
class SessionError:
    kind: SessionErrorKind
    details: str

AppError = Union[DetectionError, EmbeddingError, VerificationError, SessionError]
