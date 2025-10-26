from dataclasses import dataclass
from typing import Optional
import sys

@dataclass(frozen=True, slots=True)
class PackageInfo:
    name: str
    version: str
    import_name: str
    gpu_support: bool

def get_package_version(package_name: str, import_name: Optional[str] = None) -> Optional[str]:
    actual_import = import_name if import_name else package_name
    try:
        module = __import__(actual_import)
        return getattr(module, '__version__', 'Unknown')
    except ImportError:
        return None

def verify_package(name: str, import_name: Optional[str] = None, check_gpu: bool = False) -> tuple[bool, PackageInfo]:
    actual_import = import_name if import_name else name
    version = get_package_version(name, actual_import)

    if version is None:
        return False, PackageInfo(name=name, version="NOT INSTALLED", import_name=actual_import, gpu_support=False)

    gpu_available = False
    if check_gpu and actual_import == "torch":
        import torch
        gpu_available = torch.cuda.is_available()
    elif check_gpu and actual_import == "faiss":
        try:
            import faiss
            gpu_available = hasattr(faiss, 'StandardGpuResources')
        except:
            gpu_available = False

    return True, PackageInfo(
        name=name,
        version=version,
        import_name=actual_import,
        gpu_support=gpu_available
    )

def print_package_status(info: PackageInfo, success: bool) -> None:
    status = "OK" if success else "MISSING"
    gpu_indicator = " [GPU]" if info.gpu_support else ""
    print(f"{info.name:20} {info.version:15} {status:10} {gpu_indicator}")

def main() -> int:
    print("\nDependency Verification for Face Detection System")
    print("=" * 70)
    print(f"{"Package":20} {"Version":15} {"Status":10} {"GPU Support":15}")
    print("-" * 70)

    packages = [
        ("torch", "torch", True),
        ("torchvision", "torchvision", False),
        ("insightface", "insightface", False),
        ("deepface", "deepface", False),
        ("opencv-python", "cv2", False),
        ("faiss-gpu", "faiss", True),
        ("numpy", "numpy", False),
        ("pandas", "pandas", False),
        ("scikit-learn", "sklearn", False),
        ("albumentations", "albumentations", False),
        ("pillow", "PIL", False),
        ("matplotlib", "matplotlib", False),
        ("onnx", "onnx", False),
        ("onnxruntime-gpu", "onnxruntime", False),
        ("tqdm", "tqdm", False),
    ]

    all_success = True
    results = []

    for pkg_name, import_name, check_gpu in packages:
        success, info = verify_package(pkg_name, import_name, check_gpu)
        results.append((info, success))
        print_package_status(info, success)
        if not success:
            all_success = False

    print("=" * 70)

    if not all_success:
        print("\nFAILED: Some packages are missing")
        print("\nTo install missing packages:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("  pip install insightface deepface opencv-python faiss-gpu")
        print("  pip install numpy pandas scikit-learn albumentations pillow matplotlib")
        print("  pip install onnx onnxruntime-gpu tqdm")
        return 1

    print("\nSUCCESS: All dependencies installed")

    torch_gpu = any(info.gpu_support for info, _ in results if info.import_name == "torch")
    faiss_gpu = any(info.gpu_support for info, _ in results if info.import_name == "faiss")

    if torch_gpu and faiss_gpu:
        print("GPU support: ENABLED for PyTorch and FAISS")
    elif torch_gpu:
        print("GPU support: ENABLED for PyTorch, MISSING for FAISS")
        print("  Consider installing faiss-gpu instead of faiss-cpu")
    elif faiss_gpu:
        print("GPU support: MISSING for PyTorch, ENABLED for FAISS")
        print("  PyTorch may be CPU-only version")
    else:
        print("GPU support: DISABLED for both PyTorch and FAISS")
        print("  Training will be significantly slower on CPU")

    print("\nNext step: Run 'python scripts/verify_gpu.py' to verify GPU environment")
    return 0

if __name__ == "__main__":
    sys.exit(main())
