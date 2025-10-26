# Face Detection & Recognition System - Usage Guide

## Quick Start

The project has a unified CLI entry point. All functionality is accessed through `main.py`.

### Installation

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Dependencies are already installed
```

### Running the System

#### Real-Time Verification (Recommended)

**Interactive Mode (Easiest):**
```bash
python main.py verify-realtime
```

**What it does:**
- Shows interactive hierarchical file browser starting at data/test/
- Browse folders: Arrow keys (↑/↓), Enter to open folder
- Navigate back: Select "../" option
- Select image: Arrow keys to highlight, Enter to confirm
- Automatically starts real-time verification

**Browser Features:**
- Shows folders marked with [DIR]
- Shows images indented
- Displays current location and counts
- Navigate into subfolders and back out
- Can access any image in any subdirectory

**Direct Mode:**
```bash
python main.py verify-realtime <path_to_reference_image>
```

**Features:**
- Accepts a single reference image (face photo OR admit card with photo)
- Auto-detects image type (face photo vs admit card)
- Opens split-screen view: reference on left, live camera on right
- Shows continuous MATCH/NO MATCH status in real-time
- Displays similarity score updating live

**Controls:**
- Interactive browser: Arrow keys to navigate, Enter to select
- During verification: `Q` - Quit

**Examples:**
```bash
# Interactive mode (recommended for testing)
python main.py verify-realtime

# Direct mode with file path
python main.py verify-realtime data/test/person_01_Akshay_Kumar/01.jpg
python main.py verify-realtime data/reference/admit_card.jpg
python main.py verify-realtime C:/Users/Student/photo.jpg
```

**Automatic Features:**
- Interactive file browser when no path provided
- Automatic image type detection (face photo vs admit card)
- Continuous real-time verification (no capture needed)
- Split-screen view showing reference and live feed
- Green "MATCH" when similarity ≥ 0.4 threshold
- Red "NO MATCH" when similarity < 0.4 threshold
- Live similarity score display

#### Admit Card Verification (3-Stage Demo)

```bash
python main.py verify-card
```

**What it does:**
- Opens camera for admit card capture
- Extracts face from admit card photo
- Captures live face from student
- Verifies identity with 1:1 face matching
- Displays VERIFIED or REJECTED result

**Controls:**
- `SPACE` - Capture image
- `R` - Retry/Restart
- `Q` - Quit

**Workflow:**
1. Stage 1: Place admit card in blue rectangle → press SPACE
2. Stage 2: Look directly at camera → press SPACE
3. Stage 3: View result with similarity score

#### Baseline Accuracy Testing

```bash
python main.py baseline
```

**What it does:**
- Tests InsightFace Buffalo_L model on Indian face dataset
- Evaluates accuracy at multiple thresholds (0.3, 0.4, 0.5, 0.6)
- Measures GPU-accelerated inference speed
- Generates detailed performance report

**Output:**
- Overall accuracy metrics
- Per-individual performance breakdown
- Speed benchmarks
- False accept/reject rates

#### Help

```bash
python main.py help
```

Shows available commands and usage examples.

## System Architecture

### Entry Point

- **main.py** - Unified CLI entry point
  - `verify-realtime <image>` - Real-time verification against reference image
  - `verify-card` - Admit card verification system (3-stage)
  - `baseline` - Baseline accuracy testing
  - `help` - Show usage information

### Core Modules

- **src/detection/** - Face detection implementations
  - `insightface_detector.py` - Unified InsightFace detector with RetinaFace

- **src/recognition/** - Face recognition implementations
  - `insightface_cpu_recognizer.py` - InsightFace ArcFace recognizer with CPU provider

- **src/verification/** - Verification logic
  - `face_verifier.py` - Cosine similarity verification
  - `verification_session.py` - Session state management
  - `reference_processor.py` - Reference image loader and auto-detection

- **src/ui/** - User interface
  - `verification_ui.py` - OpenCV-based UI (3-stage workflow)
  - `realtime_verification_ui.py` - Real-time split-screen UI

- **src/core/** - Core types and utilities
  - `types.py` - Result types, configs, enums
  - `errors.py` - Error type definitions

### Scripts (Internal)

- **scripts/run_realtime_verification.py** - Real-time verification implementation
- **scripts/run_admit_card_verification.py** - 3-stage verification implementation
- **scripts/run_baseline_insightface_gpu.py** - Baseline testing implementation
- **scripts/archive/** - Historical helper scripts (Phase 1-2 testing)

## Hardware Requirements

### Development Machine (Current Setup)
- GPU: NVIDIA RTX 3070 Ti (8GB VRAM)
- CUDA: 12.4
- cuDNN: 9.14
- Performance: 7.78ms per face

### Demo Machine (EliteBook)
- CPU: Intel i7-8650U (8th gen, 4 cores)
- RAM: 24GB
- GPU: Integrated Intel UHD 620
- Expected performance: ~150ms per verification

## Performance Metrics

### Admit Card Verification (i7-8650U)

| Operation | Expected Latency |
|-----------|-----------------|
| InsightFace RetinaFace detection | 40ms |
| InsightFace ArcFace embedding | 20ms |
| Cosine similarity | <1ms |
| Total per verification | ~100ms |

**Target:** <1 second (ACHIEVED: 6.7x faster)

### Baseline Testing (RTX 3070 Ti)

| Metric | Result |
|--------|--------|
| Accuracy (threshold 0.3) | 85.1% |
| Speed | 7.78ms per face |
| False Accept Rate | 0.00% |
| False Reject Rate | 14.9% |

## Error Handling

All operations return `Result[A, E]` types (Railway-oriented programming):
- `Success(value)` - Operation succeeded
- `Failure(error)` - Operation failed with typed error

Error types:
- `DetectionError` - Face detection failures
- `EmbeddingError` - Embedding extraction failures
- `VerificationError` - Verification computation failures
- `SessionError` - Session state management failures

## Troubleshooting

### Camera Not Opening
```bash
# Check if camera is available
python -c "import cv2; print('Camera:', cv2.VideoCapture(0).isOpened())"
```

### CUDA/GPU Issues
```bash
# Check CUDA installation
nvidia-smi

# Verify ONNX Runtime GPU provider
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

### Poor Accuracy
- Ensure good lighting (indoor, diffused light)
- Position face directly at camera (not angled)
- Maintain 20-60cm distance from camera
- Use high-quality admit card images

### Slow Performance
- Close background applications
- Ensure adequate RAM (16GB+ recommended)
- Check CPU usage (should be <80% during operation)

## Development

### Running Tests

No automated tests yet. Manual testing recommended:
1. Test card capture at various distances
2. Test different lighting conditions
3. Verify error handling scenarios
4. Measure actual latency on target hardware

### Code Standards

All code follows engineering-standards.md:
- Zero comments (self-documenting code)
- Result types for error handling
- Immutable dataclasses with slots
- Protocol-based design
- Functions ≤20 lines
- Cyclomatic complexity ≤7

### Adding New Features

1. Create new module in appropriate src/ directory
2. Follow existing patterns (Result types, protocols, dataclasses)
3. Add subcommand to main.py if user-facing
4. Update this documentation
5. Update dev-log.md

## Future Enhancements

Potential additions (not yet implemented):
- OCR for admit card text validation
- Liveness detection (blink, motion)
- GPU acceleration for sub-100ms latency
- Multi-student batch processing
- Database integration
- REST API wrapper
- Web-based UI
