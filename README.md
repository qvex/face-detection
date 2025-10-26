# Face Detection & Recognition System

Real-time face verification system optimized for Indian faces with interactive testing interface.

## Quick Start (Fresh Machine)

```bash
# 1. Install Python 3.11+ (add to PATH)
# 2. Clone repository
git clone <repository-url>
cd face-detection

# 3. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run with interactive file browser
python main.py verify-realtime
```

**First run downloads models (~900MB, takes 5-10 minutes)**

See [SETUP.md](SETUP.md) for detailed installation guide.

## Prerequisites

### Minimum (Real-Time Verification)
- Python 3.11+
- Webcam (built-in or USB)
- 8GB RAM
- Internet (for first-time model download)

### Optional (GPU Baseline Testing)
- NVIDIA GPU (8GB+ VRAM)
- CUDA 12.4 + cuDNN 9.14

## Project Structure

```
face-detection/
  src/
    core/
      types.py          # Type definitions and algebraic data types
    models/
      detector.py       # Face detection and recognition protocols
    testing/
      baseline_test.py  # Baseline model testing framework
  data/
    raw/                # Raw dataset storage
  models/               # Model checkpoints and ONNX exports
```

## Step 2: Baseline Testing

Test data requirements:
- 100 images minimum
- 10 Indian individuals
- 10 images per person

Models tested:
1. InsightFace Buffalo_L
2. DeepFace VGG-Face
3. DeepFace Facenet512
4. Dlib ResNet

Metrics measured:
- Accuracy
- Speed (ms/face)
- False match rate at threshold 0.6

## Step 3: Dataset Acquisition

Public datasets:
- IMFDB: Kaggle dataset (34,512 images)
- IFExD: GitHub clone (BSD license)
- Face-Indian: Roboflow public

Custom collection requirements:
- 30-50 people minimum
- 50-100 images per person
- Variations: angles, lighting, expressions, distance

Target: 15,000+ images, 80+ identities

## Step 4: Preprocessing Pipeline

Process:
1. Face detection with RetinaFace
2. Align to 112x112
3. Filter single-face images, remove blur
4. Remove duplicates

Data split:
- Train: 70%
- Validation: 15%
- Test: 15%

Minimum requirements:
- 50+ identities
- 20+ images per identity

## Step 5: Training Configuration

Base model: InsightFace ArcFace ResNet-50 pre-trained on MS1M-V3

Architecture:
- Freeze first 30 layers (60% of network)
- Replace final FC layer with ArcFace loss head

Hyperparameters:
- Batch size: 128
- Learning rate: 0.1 → 0.01 (epoch 16) → 0.001 (epoch 24)
- Optimizer: SGD (momentum=0.9, weight_decay=5e-4)
- Epochs: 30
- ArcFace: margin=0.5, scale=64

Augmentation:
- Horizontal flip (p=0.5)
- Rotation (±20°)
- Color jitter (brightness=0.2, contrast=0.2)
- Random crop (scale 0.8-1.0)
- Normalize: mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]

## Development Standards

This project follows strict engineering standards defined in engineering-standards.md:

1. Zero assumption policy - all implementations must have 10/10 confidence
2. Advanced Python patterns for code efficiency
3. Type-safe architecture using Protocols and algebraic data types
4. Railway-oriented programming for error handling
5. Formal correctness verification required before implementation
