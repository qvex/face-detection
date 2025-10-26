#!/usr/bin/env python3
"""Verify CUDA 12.x and cuDNN installation."""
import os
import sys
from pathlib import Path

print("=" * 70)
print("CUDA 12.x & cuDNN Installation Verification")
print("=" * 70)

# Check environment variables
print("\n1. Environment Variables:")
cuda_path = os.environ.get('CUDA_PATH', 'Not set')
print(f"   CUDA_PATH: {cuda_path}")

# Check PATH for CUDA
path_env = os.environ.get('PATH', '')
cuda_paths = [p for p in path_env.split(';') if 'CUDA' in p.upper()]
print(f"\n   CUDA-related paths in PATH:")
for p in cuda_paths[:5]:
    print(f"     - {p}")

# Check CUDA 12.4 installation
print("\n2. CUDA 12.4 Installation:")
cuda_12_4_path = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4")
if cuda_12_4_path.exists():
    print(f"   [OK] CUDA 12.4 directory exists: {cuda_12_4_path}")

    # Check critical DLLs
    bin_path = cuda_12_4_path / "bin"
    critical_dlls = ['cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll']
    for dll in critical_dlls:
        dll_path = bin_path / dll
        if dll_path.exists():
            print(f"   [OK] {dll} found")
        else:
            print(f"   [MISSING] {dll} NOT found")
else:
    print(f"   [ERROR] CUDA 12.4 directory NOT found")

# Check cuDNN installation
print("\n3. cuDNN Installation:")
cudnn_path = Path("C:/Program Files/NVIDIA/CUDNN/v9.14/bin/12.9")
if cudnn_path.exists():
    print(f"   [OK] cuDNN directory exists: {cudnn_path}")

    # Check cuDNN DLLs
    cudnn_dlls = list(cudnn_path.glob("cudnn*.dll"))
    print(f"   [OK] Found {len(cudnn_dlls)} cuDNN DLLs")
    for dll in cudnn_dlls[:3]:
        print(f"     - {dll.name}")
else:
    print(f"   [WARNING] cuDNN directory NOT found at {cudnn_path}")

# Test PyTorch CUDA detection
print("\n4. PyTorch CUDA Detection:")
try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version (PyTorch): {torch.version.cuda}")
        print(f"   cuDNN version: {torch.backends.cudnn.version()}")
        print(f"   Device count: {torch.cuda.device_count()}")
        print(f"   Current device: {torch.cuda.current_device()}")
        print(f"   Device name: {torch.cuda.get_device_name(0)}")
    else:
        print(f"   [WARNING] PyTorch cannot detect CUDA")
except Exception as e:
    print(f"   [ERROR] {e}")

# Test ONNX Runtime CUDA detection
print("\n5. ONNX Runtime CUDA Detection:")
try:
    import onnxruntime as ort
    print(f"   ONNX Runtime version: {ort.__version__}")
    providers = ort.get_available_providers()
    print(f"   Available providers: {', '.join(providers)}")

    if 'CUDAExecutionProvider' in providers:
        print(f"   [OK] CUDAExecutionProvider is available")
    else:
        print(f"   [WARNING] CUDAExecutionProvider NOT available")
except Exception as e:
    print(f"   [ERROR] {e}")

print("\n" + "=" * 70)
print("Verification Complete")
print("=" * 70)

# Provide recommendations
print("\nRecommendations:")
if cuda_path != str(cuda_12_4_path):
    print("  1. Update CUDA_PATH environment variable to:")
    print(f"     {cuda_12_4_path}")

if not any('v12.4' in p for p in cuda_paths):
    print("  2. Add to PATH (at the beginning):")
    print(f"     {cuda_12_4_path / 'bin'}")
    print(f"     {cuda_12_4_path / 'libnvvp'}")

if cudnn_path.exists() and not any('CUDNN' in p for p in cuda_paths):
    print("  3. Add cuDNN to PATH:")
    print(f"     {cudnn_path}")

print("\n  After updating environment variables:")
print("  - Restart your terminal/IDE")
print("  - Re-run this script to verify")
print("=" * 70)
