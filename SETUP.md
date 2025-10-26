# Setup Guide - Fresh Machine Installation

## Prerequisites

### Minimum Requirements (For Real-Time Verification Only)

1. **Python 3.11+** (tested on 3.13)
   - Download: https://www.python.org/downloads/
   - During installation: Check "Add Python to PATH"

2. **Webcam**
   - Built-in laptop camera OR
   - External USB webcam

3. **Operating System**
   - Windows 10/11 (tested)
   - Linux (Ubuntu 20.04+)
   - macOS (limited testing)

4. **RAM**: 8GB minimum, 16GB+ recommended

5. **Disk Space**: ~2GB
   - 500MB for dependencies
   - 1GB for InsightFace models (auto-downloaded)
   - 500MB for Python + venv

### Additional Requirements (For GPU-Accelerated Baseline Testing)

6. **NVIDIA GPU** (optional, only for baseline testing)
   - 8GB+ VRAM recommended
   - Compute Capability 6.0+ (Pascal architecture or newer)

7. **CUDA Toolkit 12.4** (optional, only for GPU baseline)
   - Download: https://developer.nvidia.com/cuda-12-4-0-download-archive
   - Follow NVIDIA installation guide for your OS

8. **cuDNN 9.14 for CUDA 12.x** (optional, only for GPU baseline)
   - Download: https://developer.nvidia.com/cudnn (requires NVIDIA account)
   - Extract to CUDA installation directory

9. **NVIDIA GPU Drivers** (latest)
   - Download: https://www.nvidia.com/Download/index.aspx

## Installation Steps

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd face-detection
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

**For CPU-only (Real-Time Verification):**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- numpy (>=2.0)
- opencv-python (>=4.12.0)
- insightface (>=0.7)
- onnxruntime (>=1.16)
- mtcnn (>=0.1)
- questionary (>=2.0)

Note: Version constraints are minimums. Latest compatible versions will be installed automatically.

**For GPU-accelerated (Baseline Testing):**
```bash
pip install --upgrade pip
pip install onnxruntime-gpu>=1.16  # Replace onnxruntime with GPU version
pip install -r requirements.txt
```

Note: For GPU, uninstall CPU version first if already installed:
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu>=1.16
```

### Step 4: Verify Installation

**Test Python:**
```bash
python --version
# Should show: Python 3.11+ or 3.13
```

**Test camera:**
```bash
python -c "import cv2; print('Camera available:', cv2.VideoCapture(0).isOpened())"
# Should show: Camera available: True
```

**Test imports:**
```bash
python -c "import cv2, numpy, insightface, mtcnn, questionary; print('All imports successful')"
```

### Step 5: First Run (Downloads Models)

**First run downloads InsightFace models (~900MB):**
```bash
python main.py verify-realtime data/test/person_01_Akshay_Kumar/01.jpg
```

This will:
1. Download InsightFace Buffalo_L models to `~/.insightface/models/`
2. Download MTCNN weights automatically
3. May take 5-10 minutes on first run
4. Subsequent runs are instant

### Step 6: Test Interactive Mode

```bash
python main.py verify-realtime
```

Should show:
- Found 80 images in data\test
- Interactive file browser
- Arrow key navigation works

## What Gets Downloaded Automatically

1. **InsightFace Buffalo_L models** (~900MB)
   - Location: `~/.insightface/models/buffalo_l/`
   - Downloaded on first run
   - Cached for future use

2. **MTCNN weights** (~5MB)
   - Downloaded by mtcnn package
   - Cached automatically

## Directory Structure After Installation

```
face-detection/
├── venv/                           # Virtual environment (created by you)
├── data/
│   └── test/                       # Test images (included in repo)
│       ├── person_01_Akshay_Kumar/
│       ├── person_02_Alia_Bhatt/
│       └── ...
├── src/                            # Source code (in repo)
├── scripts/                        # Scripts (in repo)
├── main.py                         # Entry point (in repo)
├── requirements.txt                # Dependencies (in repo)
└── README.md                       # Documentation (in repo)
```

## Common Issues & Solutions

### Issue 1: Camera Not Opening

**Error:** `Failed to initialize camera`

**Solutions:**
1. Check if camera is being used by another application
2. Try unplugging/replugging USB camera
3. Check camera permissions (Windows Settings → Privacy → Camera)
4. Try different camera index: modify code to use `VideoCapture(1)` instead of `VideoCapture(0)`

### Issue 2: CUDA Not Found (for GPU baseline)

**Error:** `CUDA not available` or `onnxruntime-gpu not working`

**Solutions:**
1. Verify CUDA installation: `nvcc --version`
2. Verify GPU: `nvidia-smi`
3. Check PATH includes CUDA bin directory
4. Reinstall onnxruntime-gpu: `pip uninstall onnxruntime onnxruntime-gpu && pip install onnxruntime-gpu==1.16.3`

### Issue 3: InsightFace Download Fails

**Error:** `Failed to download model`

**Solutions:**
1. Check internet connection
2. Try manual download from InsightFace GitHub
3. Models location: `~/.insightface/models/buffalo_l/`

### Issue 4: Import Errors

**Error:** `ModuleNotFoundError: No module named 'cv2'`

**Solutions:**
1. Ensure virtual environment is activated
2. Reinstall dependencies: `pip install -r requirements.txt`
3. Check Python version: `python --version` (needs 3.11+)

### Issue 5: NumPy Compatibility

**Error:** `numpy.ndarray size changed` or similar

**Solutions:**
1. Ensure numpy 2.x compatible versions
2. Update OpenCV: `pip install opencv-python==4.12.0.88`
3. Reinstall all dependencies: `pip install --force-reinstall -r requirements.txt`

## Minimal Quick Start (No GPU)

For testing real-time verification on a new machine **without GPU**:

```bash
# 1. Install Python 3.11+ (add to PATH)
# 2. Clone repository
git clone <repo-url>
cd face-detection

# 3. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# OR: source venv/bin/activate  # Linux/Mac

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run (downloads models on first run)
python main.py verify-realtime

# 6. Select image with arrow keys, press Enter
# 7. Camera opens with split-screen view
# 8. Press Q to quit
```

Total time: ~15 minutes (including model download)

## Full Setup (With GPU for Baseline Testing)

For complete setup including GPU-accelerated baseline testing:

```bash
# 1. Install Python 3.13
# 2. Install CUDA 12.4 + cuDNN 9.14
# 3. Install NVIDIA drivers
# 4. Clone repository
# 5. Create venv and activate
# 6. Install dependencies:
pip install onnxruntime-gpu==1.16.3
pip install -r requirements.txt

# 7. Verify CUDA:
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
# Should show: ['CUDAExecutionProvider', 'CPUExecutionProvider']

# 8. Run baseline test:
python main.py baseline

# 9. Run real-time verification:
python main.py verify-realtime
```

## Testing on EliteBook (No GPU)

Your demo laptop (HP EliteBook 840 G5) requirements:

1. **Python 3.11+** ✓
2. **Webcam** ✓ (built-in)
3. **8GB+ RAM** ✓ (has 24GB)
4. **No GPU needed** ✓ (uses CPU inference)

Expected performance:
- Real-time verification: ~150ms per frame (6-7 FPS)
- Sufficient for smooth real-time experience

## Production Checklist

Before deploying on a new machine:

- [ ] Python 3.11+ installed and in PATH
- [ ] Virtual environment created
- [ ] Dependencies installed from requirements.txt
- [ ] Camera accessible (test with OpenCV)
- [ ] InsightFace models downloaded (first run)
- [ ] Test images available in data/test/
- [ ] Interactive mode works (`python main.py verify-realtime`)
- [ ] Real-time verification works with camera
- [ ] (Optional) CUDA + cuDNN installed for GPU baseline

## Network/Firewall Requirements

The following outbound connections are needed:

1. **PyPI** (pypi.org) - For pip install
2. **GitHub** (github.com) - For InsightFace model download
3. **Google Drive** (may be used by InsightFace) - Model hosting

If behind corporate firewall:
- Ensure HTTPS (443) is allowed
- May need to configure pip proxy: `pip install --proxy http://proxy:port`

## Offline Installation

For machines without internet:

1. **Download wheels on internet-connected machine:**
   ```bash
   pip download -r requirements.txt -d ./wheels
   ```

2. **Copy wheels folder to offline machine**

3. **Install from local wheels:**
   ```bash
   pip install --no-index --find-links=./wheels -r requirements.txt
   ```

4. **Manually download InsightFace models:**
   - Download buffalo_l models from InsightFace GitHub
   - Place in `~/.insightface/models/buffalo_l/`

## Support

For issues:
1. Check this SETUP.md
2. Review [project_configs/usage-guide.md](project_configs/usage-guide.md)
3. Check [project_configs/dev-log.md](project_configs/dev-log.md)
4. Verify all prerequisites are met
