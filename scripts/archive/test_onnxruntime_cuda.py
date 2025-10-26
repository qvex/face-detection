#!/usr/bin/env python3
"""Test ONNX Runtime CUDA detection."""
import onnxruntime as ort

print("=" * 70)
print("ONNX Runtime CUDA Detection Test")
print("=" * 70)
print(f"ONNX Runtime version: {ort.__version__}")
print(f"\nAvailable providers:")
for provider in ort.get_available_providers():
    print(f"  - {provider}")

print(f"\nAll providers:")
for provider in ort.get_all_providers():
    print(f"  - {provider}")

# Try to create a session with CUDA
try:
    print("\nTesting CUDAExecutionProvider...")
    import numpy as np
    from onnx import helper, TensorProto

    # Create a simple ONNX model
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])
    node = helper.make_node('Identity', ['X'], ['Y'])
    graph = helper.make_graph([node], 'test', [X], [Y])
    model = helper.make_model(graph)

    # Try to run with CUDA
    sess = ort.InferenceSession(
        model.SerializeToString(),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    print(f"Session providers: {sess.get_providers()}")

    # Run inference
    x = np.array([[1.0, 2.0]], dtype=np.float32)
    result = sess.run(None, {'X': x})
    print(f"Test inference successful: {result}")

    if 'CUDAExecutionProvider' in sess.get_providers():
        print("\n[SUCCESS] CUDA is working!")
    else:
        print("\n[WARNING] Session fell back to CPU")

except Exception as e:
    print(f"\n[ERROR] CUDA test failed: {e}")

print("=" * 70)
