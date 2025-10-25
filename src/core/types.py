from typing import TypeVar, Generic, Protocol, Union
from dataclasses import dataclass
from enum import Enum, auto

A = TypeVar('A')
E = TypeVar('E')

@dataclass(frozen=True, slots=True)
class Success(Generic[A]):
    value: A

@dataclass(frozen=True, slots=True)
class Failure(Generic[E]):
    error: E

Result = Union[Success[A], Failure[E]]

class ModelType(Enum):
    INSIGHTFACE_BUFFALO_L = auto()
    DEEPFACE_VGG = auto()
    DEEPFACE_FACENET512 = auto()
    DLIB_RESNET = auto()

@dataclass(frozen=True, slots=True)
class DetectionConfig:
    det_size: int = 640
    align_size: int = 112
    confidence_threshold: float = 0.9

@dataclass(frozen=True, slots=True)
class TrainingConfig:
    batch_size: int = 128
    epochs: int = 30
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0005
    arcface_margin: float = 0.5
    arcface_scale: int = 64
    freeze_layers: int = 30

@dataclass(frozen=True, slots=True)
class DataSplit:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15

@dataclass(frozen=True, slots=True)
class TestMetrics:
    accuracy: float
    speed_ms: float
    false_match_rate: float
    model_name: str
