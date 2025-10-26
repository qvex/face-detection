#!/usr/bin/env python3
"""Test InsightFace with CUDA 12.4."""
import os

# Add CUDA 12.4 and cuDNN to PATH BEFORE importing any CUDA-dependent libraries
cuda_12_4_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
cudnn_bin = r"C:\Program Files\NVIDIA\CUDNN\v9.14\bin\12.9"

current_path = os.environ.get('PATH', '')
os.environ['PATH'] = f"{cuda_12_4_bin};{cudnn_bin};{current_path}"
os.environ['CUDA_PATH'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"

print("=" * 70)
print("InsightFace CUDA 12.4 Test")
print("=" * 70)
print(f"Updated PATH to include:")
print(f"  - {cuda_12_4_bin}")
print(f"  - {cudnn_bin}")
print("=" * 70)

import cv2
import numpy as np
from pathlib import Path

# Test image path
test_img_path = Path("data/test/person_01_Akshay_Kumar/01.jpg")

if not test_img_path.exists():
    print(f"ERROR: Test image not found: {test_img_path}")
    exit(1)

# Load image
img = cv2.imread(str(test_img_path))
print(f"\nLoaded test image: {test_img_path}")
print(f"Image shape: {img.shape}")

# Initialize InsightFace
from insightface.app import FaceAnalysis

print("\nInitializing InsightFace with CUDA...")
try:
    app = FaceAnalysis(
        name='buffalo_l',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("InsightFace initialized successfully")

    # Check which providers are being used
    print(f"\nModel providers in use:")
    for model_name, model in app.models.items():
        if hasattr(model, 'session'):
            providers = model.session.get_providers()
            provider_str = providers[0] if providers else "Unknown"
            status = "[GPU]" if "CUDA" in provider_str else "[CPU]"
            print(f"  {status} {model_name}: {provider_str}")

    # Detect faces
    print(f"\nDetecting faces...")
    faces = app.get(img)

    if len(faces) > 0:
        print(f"\n[SUCCESS] Detected {len(faces)} face(s)")
        for idx, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            print(f"  Face {idx+1}:")
            print(f"    BBox: {bbox}")
            print(f"    Age: {face.age:.0f}")
            print(f"    Gender: {'Male' if face.gender==1 else 'Female'}")
            print(f"    Embedding shape: {face.embedding.shape}")
    else:
        print(f"\n[WARNING] No faces detected")
        print(f"Note: Images are 160x160 pre-cropped faces, might be too small for detection")
        print(f"This is expected - we'll use embeddings directly without detection")

    # Check if CUDA is actually being used
    cuda_in_use = any(
        'CUDAExecutionProvider' in model.session.get_providers()[0]
        for model_name, model in app.models.items()
        if hasattr(model, 'session')
    )

    if cuda_in_use:
        print(f"\n" + "=" * 70)
        print("[SUCCESS] InsightFace is using CUDA 12.4 GPU acceleration!")
        print("=" * 70)
    else:
        print(f"\n" + "=" * 70)
        print("[WARNING] InsightFace is running on CPU")
        print("=" * 70)

except Exception as e:
    print(f"\n[ERROR] InsightFace initialization failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
