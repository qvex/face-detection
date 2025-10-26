from dataclasses import dataclass
from typing import Optional
import sys
import time

@dataclass(frozen=True, slots=True)
class GPUInfo:
    name: str
    total_memory_gb: float
    cuda_version: str
    device_count: int
    compute_capability: tuple[int, int]

@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    operation: str
    time_ms: float
    throughput_gflops: Optional[float]

def check_pytorch_available() -> bool:
    try:
        import torch
        return True
    except ImportError:
        print("FAILED: PyTorch not installed")
        print("Install with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return False

def get_gpu_info() -> Optional[GPUInfo]:
    import torch

    if not torch.cuda.is_available():
        return None

    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    cuda_version = torch.version.cuda if torch.version.cuda else "Unknown"
    compute_cap = torch.cuda.get_device_capability(0)

    return GPUInfo(
        name=device_name,
        total_memory_gb=total_memory,
        cuda_version=cuda_version,
        device_count=device_count,
        compute_capability=compute_cap
    )

def benchmark_matrix_multiply(size: int = 4096) -> BenchmarkResult:
    import torch

    device = torch.device("cuda")
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    torch.cuda.synchronize()
    start = time.perf_counter()

    c = torch.matmul(a, b)

    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    flops = 2 * size * size * size
    gflops = (flops / (elapsed_ms / 1000)) / 1e9

    return BenchmarkResult(
        operation=f"Matrix Multiply ({size}x{size})",
        time_ms=elapsed_ms,
        throughput_gflops=gflops
    )

def benchmark_memory_bandwidth() -> BenchmarkResult:
    import torch

    device = torch.device("cuda")
    size_mb = 1024
    elements = (size_mb * 1024 * 1024) // 4

    data = torch.randn(elements, device=device)

    torch.cuda.synchronize()
    start = time.perf_counter()

    result = data * 2.0

    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    bandwidth_gbps = (size_mb / 1024) / (elapsed_ms / 1000)

    return BenchmarkResult(
        operation=f"Memory Bandwidth ({size_mb}MB)",
        time_ms=elapsed_ms,
        throughput_gflops=bandwidth_gbps
    )

def print_gpu_info(info: GPUInfo) -> None:
    print("=" * 60)
    print("GPU INFORMATION")
    print("=" * 60)
    print(f"Device Name:          {info.name}")
    print(f"Device Count:         {info.device_count}")
    print(f"Total Memory:         {info.total_memory_gb:.2f} GB")
    print(f"CUDA Version:         {info.cuda_version}")
    print(f"Compute Capability:   {info.compute_capability[0]}.{info.compute_capability[1]}")
    print("=" * 60)

def print_benchmark(result: BenchmarkResult) -> None:
    throughput_str = f"{result.throughput_gflops:.2f} GFLOPS" if result.throughput_gflops else "N/A"
    print(f"{result.operation:30} {result.time_ms:8.2f} ms    {throughput_str}")

def verify_minimum_requirements(info: GPUInfo) -> tuple[bool, list[str]]:
    issues = []

    if info.total_memory_gb < 8.0:
        issues.append(f"Insufficient VRAM: {info.total_memory_gb:.2f}GB (minimum: 8GB)")

    cuda_major = int(info.cuda_version.split('.')[0]) if info.cuda_version != "Unknown" else 0
    if cuda_major < 11:
        issues.append(f"CUDA version too old: {info.cuda_version} (minimum: 11.8)")

    compute_major, compute_minor = info.compute_capability
    if compute_major < 6:
        issues.append(f"Compute capability too low: {compute_major}.{compute_minor} (minimum: 6.0)")

    return len(issues) == 0, issues

def main() -> int:
    print("\nGPU Environment Verification for Face Detection System")
    print("=" * 60)

    if not check_pytorch_available():
        return 1

    import torch
    print(f"PyTorch Version: {torch.__version__}")

    if not torch.cuda.is_available():
        print("\nFAILED: CUDA not available")
        print("Possible causes:")
        print("  - NVIDIA GPU not present")
        print("  - CUDA drivers not installed")
        print("  - PyTorch CPU-only version installed")
        return 1

    gpu_info = get_gpu_info()
    if not gpu_info:
        print("\nFAILED: Could not retrieve GPU information")
        return 1

    print_gpu_info(gpu_info)

    meets_requirements, issues = verify_minimum_requirements(gpu_info)

    if not meets_requirements:
        print("\nWARNINGS:")
        for issue in issues:
            print(f"  - {issue}")
        print()

    print("\nBENCHMARKS")
    print("=" * 60)
    print(f"{"Operation":30} {"Time":>12}    {"Throughput":>15}")
    print("-" * 60)

    matmul_result = benchmark_matrix_multiply(4096)
    print_benchmark(matmul_result)

    bandwidth_result = benchmark_memory_bandwidth()
    print_benchmark(bandwidth_result)

    print("=" * 60)

    if meets_requirements:
        print("\nRESULT: GO - Environment suitable for training")
        print(f"Expected training time: ~{24 * (8.0 / gpu_info.total_memory_gb):.1f} hours for 30 epochs")
        return 0
    else:
        print("\nRESULT: PROCEED WITH CAUTION")
        print("Training possible but may be slower or limited by hardware constraints")
        return 2

if __name__ == "__main__":
    sys.exit(main())
