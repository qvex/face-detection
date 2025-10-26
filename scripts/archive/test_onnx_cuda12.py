#!/usr/bin/env python3
"""Test ONNX Runtime with CUDA 12.x DLLs."""
import os
import sys

# Add CUDA 12.4 and cuDNN to PATH for this process
cuda_12_4_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
cudnn_bin = r"C:\Program Files\NVIDIA\CUDNN\v9.14\bin\12.9"

current_path = os.environ.get('PATH', '')
os.environ['PATH'] = f"{cuda_12_4_bin};{cudnn_bin};{current_path}"
os.environ['CUDA_PATH'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"

print("=" * 70)
print("ONNX Runtime CUDA 12.x Test (with runtime PATH update)")
print("=" * 70)

try:
    import onnxruntime as ort
    print(f"\nONNX Runtime version: {ort.__version__}")
    print(f"Available providers: {', '.join(ort.get_available_providers())}")

    # Try to create a session with CUDA
    print("\nTesting CUDA session creation...")

    # Create a minimal ONNX model in memory
    import numpy as np

    # Use a pre-existing model file to test
    print("Creating test inference session with CUDAExecutionProvider...")

    # Simple identity model
    from onnx import helper, TensorProto, save
    from pathlib import Path

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 224, 224])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3, 224, 224])
    node = helper.make_node('Identity', ['X'], ['Y'])
    graph = helper.make_graph([node], 'test_graph', [X], [Y])
    model = helper.make_model(graph, producer_name='test')

    # Save to temp file
    temp_model = Path("temp_test_model.onnx")
    save(model, str(temp_model))

    # Create session with CUDA
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
        }),
        'CPUExecutionProvider'
    ]

    session = ort.InferenceSession(
        str(temp_model),
        sess_options,
        providers=providers
    )

    print(f"\n[SUCCESS] Session created with providers: {session.get_providers()}")

    # Test inference
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    outputs = session.run(None, {'X': x})

    print(f"[SUCCESS] Inference completed successfully")
    print(f"Output shape: {outputs[0].shape}")

    # Check which provider was actually used
    if 'CUDAExecutionProvider' in session.get_providers():
        print(f"\n[SUCCESS] CUDA 12.4 is working with ONNX Runtime!")
        print(f"GPU acceleration is enabled.")
    else:
        print(f"\n[WARNING] Session fell back to CPU")
        print(f"Active providers: {session.get_providers()}")

    # Cleanup
    temp_model.unlink()

except Exception as e:
    print(f"\n[ERROR] Test failed: {e}")
    import traceback
    traceback.print_exc()

print("=" * 70)
