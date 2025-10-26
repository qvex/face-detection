#!/usr/bin/env python3
"""Test InsightFace GPU support."""
import cv2
import numpy as np
from pathlib import Path

print("=" * 70)
print("InsightFace GPU Test")
print("=" * 70)

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
    print(f"\nChecking model providers...")
    for model_name, model in app.models.items():
        if hasattr(model, 'session'):
            providers = model.session.get_providers()
            print(f"  {model_name}: {providers[0]}")

    # Detect faces
    print(f"\nDetecting faces...")
    faces = app.get(img)

    if len(faces) > 0:
        print(f"[SUCCESS] Detected {len(faces)} face(s)")
        for idx, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            print(f"  Face {idx+1}: bbox={bbox}, age={face.age:.0f}, gender={'M' if face.gender==1 else 'F'}")
    else:
        print(f"[WARNING] No faces detected")
        print(f"This might indicate an issue with face detection parameters")

except Exception as e:
    print(f"[ERROR] InsightFace initialization failed: {e}")
    import traceback
    traceback.print_exc()

print("=" * 70)
