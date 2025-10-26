from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os

from src.core.types import TrainingConfig, DetectionConfig, DataSplit

@dataclass(frozen=True, slots=True)
class DatasetPaths:
    root: Path
    imfdb: Path
    ifexd: Path
    face_indian: Path
    processed: Path
    test: Path

@dataclass(frozen=True, slots=True)
class PerformanceTargets:
    detection_latency_ms: float = 50.0
    recognition_latency_ms: float = 20.0
    search_latency_ms: float = 10.0
    total_latency_ms: float = 100.0
    target_fps: int = 15
    min_accuracy: float = 0.95
    max_false_accept_rate: float = 0.001

@dataclass(frozen=True, slots=True)
class SecurityConfig:
    encryption_algorithm: str = "AES-256"
    hash_algorithm: str = "SHA-256"
    tls_version: str = "1.3"
    api_rate_limit: int = 100
    jwt_expiry_hours: int = 24

@dataclass(frozen=True, slots=True)
class ModelPaths:
    checkpoints: Path
    pretrained: Path
    exported: Path
    best_model: Path

@dataclass(frozen=True, slots=True)
class ProjectConfig:
    training: TrainingConfig
    detection: DetectionConfig
    data_split: DataSplit
    dataset_paths: DatasetPaths
    model_paths: ModelPaths
    performance: PerformanceTargets
    security: SecurityConfig
    device: str
    num_workers: int

def get_env_or_default(key: str, default: str) -> str:
    return os.environ.get(key, default)

def get_env_int(key: str, default: int) -> int:
    value = os.environ.get(key)
    return int(value) if value else default

def get_env_float(key: str, default: float) -> float:
    value = os.environ.get(key)
    return float(value) if value else default

def create_default_dataset_paths(root: Optional[Path] = None) -> DatasetPaths:
    base_root = root if root else Path(get_env_or_default("DATA_ROOT", "./data"))

    return DatasetPaths(
        root=base_root,
        imfdb=base_root / "raw" / "IMFDB",
        ifexd=base_root / "raw" / "IFExD",
        face_indian=base_root / "raw" / "Face-Indian",
        processed=base_root / "processed",
        test=base_root / "test"
    )

def create_default_model_paths(root: Optional[Path] = None) -> ModelPaths:
    base_root = root if root else Path(get_env_or_default("MODEL_ROOT", "./models"))

    return ModelPaths(
        checkpoints=base_root / "checkpoints",
        pretrained=base_root / "pretrained",
        exported=base_root / "exported",
        best_model=base_root / "best_model.pth"
    )

def load_training_config_from_env() -> TrainingConfig:
    return TrainingConfig(
        batch_size=get_env_int("BATCH_SIZE", 128),
        epochs=get_env_int("EPOCHS", 30),
        learning_rate=get_env_float("LEARNING_RATE", 0.1),
        momentum=get_env_float("MOMENTUM", 0.9),
        weight_decay=get_env_float("WEIGHT_DECAY", 0.0005),
        arcface_margin=get_env_float("ARCFACE_MARGIN", 0.5),
        arcface_scale=get_env_int("ARCFACE_SCALE", 64),
        freeze_layers=get_env_int("FREEZE_LAYERS", 30)
    )

def load_detection_config_from_env() -> DetectionConfig:
    return DetectionConfig(
        det_size=get_env_int("DET_SIZE", 640),
        align_size=get_env_int("ALIGN_SIZE", 112),
        confidence_threshold=get_env_float("CONFIDENCE_THRESHOLD", 0.9)
    )

def load_data_split_from_env() -> DataSplit:
    return DataSplit(
        train=get_env_float("TRAIN_SPLIT", 0.70),
        val=get_env_float("VAL_SPLIT", 0.15),
        test=get_env_float("TEST_SPLIT", 0.15)
    )

def create_project_config(
    data_root: Optional[Path] = None,
    model_root: Optional[Path] = None,
    device: Optional[str] = None
) -> ProjectConfig:
    actual_device = device if device else get_env_or_default("DEVICE", "cuda")
    num_workers = get_env_int("NUM_WORKERS", 4)

    dataset_paths = create_default_dataset_paths(data_root)
    model_paths = create_default_model_paths(model_root)

    return ProjectConfig(
        training=load_training_config_from_env(),
        detection=load_detection_config_from_env(),
        data_split=load_data_split_from_env(),
        dataset_paths=dataset_paths,
        model_paths=model_paths,
        performance=PerformanceTargets(),
        security=SecurityConfig(),
        device=actual_device,
        num_workers=num_workers
    )

def ensure_directories(config: ProjectConfig) -> None:
    dirs_to_create = [
        config.dataset_paths.root,
        config.dataset_paths.imfdb,
        config.dataset_paths.ifexd,
        config.dataset_paths.face_indian,
        config.dataset_paths.processed,
        config.dataset_paths.test,
        config.model_paths.checkpoints,
        config.model_paths.pretrained,
        config.model_paths.exported
    ]

    for directory in dirs_to_create:
        directory.mkdir(parents=True, exist_ok=True)

default_config = create_project_config()
