import os
import time

import numpy as np
import torch

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


def count_parameters(model: torch.nn.Module) -> tuple[int, int, float]:
    """Returns (total_params, trainable_params, model_size_mb)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    return total, trainable, size_mb


def measure_latency(
    model: torch.nn.Module,
    input_shape: tuple[int, int, int, int] = (1, 18, 128, 128),
    device: str = "cpu",
    warmup: int = 20,
    iterations: int = 100,
) -> dict[str, float]:
    """Measures mean, median, p95 latency and FPS for a model."""
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    dummy_input = torch.randn(*input_shape, device=dev)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
            if dev.type == "cuda":
                torch.cuda.synchronize()

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(dummy_input)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000.0)  # ms

    latencies = np.array(latencies)
    mean_ms = float(np.mean(latencies))
    fps = float(1000.0 / mean_ms) * input_shape[0]

    return {
        "mean_ms": mean_ms,
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "fps": fps,
    }


def export_and_verify_onnx(
    model: torch.nn.Module,
    onnx_path: str = "outputs/weights/model.onnx",
    in_channels: int = 18,
    patch_size: int = 128,
    opset_version: int = 17,
) -> bool:
    """
    Exports PyTorch model to ONNX format with dynamic batch dimension
    and verifies numerical parity against ONNX Runtime.
    """
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    model.eval()
    model_cpu = model.to("cpu")

    dummy_input = torch.randn(1, in_channels, patch_size, patch_size, dtype=torch.float32)

    with torch.no_grad():
        torch_output = model_cpu(dummy_input)
        if hasattr(torch_output, "logits"):
            torch_output = torch_output.logits
        torch_np = torch_output.numpy()

    # Export to ONNX
    torch.onnx.export(
        model_cpu,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_raster"],
        output_names=["glacier_logits"],
        dynamic_axes={
            "input_raster": {0: "batch_size"},
            "glacier_logits": {0: "batch_size"},
        },
    )
    print(f"ONNX model successfully exported to: {onnx_path}")

    # Verify numerical parity
    if HAS_ORT:
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = session.run(None, ort_inputs)
        max_diff = float(np.max(np.abs(torch_np - ort_outs[0])))
        print(f"ONNX Runtime Parity Verification: Max Absolute Diff = {max_diff:.2e}")
        parity_passed = max_diff < 1e-4
        if parity_passed:
            print("Numerical parity check PASSED (< 1e-4)")
        else:
            print("Warning: Parity discrepancy detected between PyTorch and ONNX Runtime")
        return parity_passed
    else:
        print("onnxruntime not installed; skipping numerical parity validation.")
        return True


def run_benchmark_report(
    model: torch.nn.Module,
    model_name: str = "AdapterUNet (ResNet34)",
    in_channels: int = 18,
    patch_size: int = 128,
) -> dict:
    """Generates an end-to-end benchmark summary dictionary."""
    total, trainable, size_mb = count_parameters(model)
    cpu_latency = measure_latency(model, (1, in_channels, patch_size, patch_size), device="cpu")

    report = {
        "model_name": model_name,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "model_size_mb": round(size_mb, 2),
        "cpu_latency_mean_ms": round(cpu_latency["mean_ms"], 2),
        "cpu_fps": round(cpu_latency["fps"], 1),
    }

    if torch.cuda.is_available():
        gpu_latency = measure_latency(model, (16, in_channels, patch_size, patch_size), device="cuda")
        report["gpu_batch16_latency_mean_ms"] = round(gpu_latency["mean_ms"], 2)
        report["gpu_batch16_fps"] = round(gpu_latency["fps"], 1)

    print("\n" + "=" * 50)
    print(f"BENCHMARK REPORT: {model_name}")
    print("=" * 50)
    print(f"Total Parameters     : {total:,}")
    print(f"Trainable Parameters : {trainable:,}")
    print(f"Model Checkpoint Size: {size_mb:.2f} MB")
    print(f"CPU Latency (B=1)    : {report['cpu_latency_mean_ms']} ms ({report['cpu_fps']} FPS)")
    if torch.cuda.is_available():
        print(f"GPU Latency (B=16)   : {report['gpu_batch16_latency_mean_ms']} ms ({report['gpu_batch16_fps']} FPS)")
    print("=" * 50 + "\n")

    return report
