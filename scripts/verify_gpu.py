"""BrandPulse AI - Hardware & GPU Verification Utility.

Checks Python version, PyTorch installation, CUDA availability,
device name, compute capability, and allocated/free VRAM.
"""

from __future__ import annotations

import platform
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_gpu")


def get_gpu_info() -> dict[str, str | bool | float | int]:
    """Inspect and return dictionary of GPU/system hardware parameters."""
    info: dict[str, str | bool | float | int] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": False,
        "device_name": "CPU",
        "device_count": 0,
        "vram_total_gb": 0.0,
        "vram_allocated_gb": 0.0,
        "vram_reserved_gb": 0.0,
        "fp16_supported": False,
    }

    try:
        import torch

        info["torch_version"] = torch.__version__
        cuda_ok = torch.cuda.is_available()
        info["cuda_available"] = cuda_ok

        if cuda_ok:
            count = torch.cuda.device_count()
            info["device_count"] = count
            curr_dev = torch.cuda.current_device()
            dev_name = torch.cuda.get_device_name(curr_dev)
            info["device_name"] = dev_name

            props = torch.cuda.get_device_properties(curr_dev)
            total_mem = props.total_memory / (1024**3)
            info["vram_total_gb"] = round(total_mem, 2)

            allocated = torch.cuda.memory_allocated(curr_dev) / (1024**3)
            reserved = torch.cuda.memory_reserved(curr_dev) / (1024**3)
            info["vram_allocated_gb"] = round(allocated, 3)
            info["vram_reserved_gb"] = round(reserved, 3)

            # Check compute capability >= 7.0 for fp16 Tensor Cores
            major, minor = torch.cuda.get_device_capability(curr_dev)
            info["compute_capability"] = f"{major}.{minor}"
            info["fp16_supported"] = major >= 7

            logger.info("CUDA GPU detected successfully!")
            logger.info(f"  Device: {dev_name} (Device ID: {curr_dev})")
            logger.info(f"  Total VRAM: {info['vram_total_gb']} GB")
            logger.info(f"  Compute Capability: {info['compute_capability']}")
            logger.info(f"  FP16 Tensor Cores: {'Supported' if info['fp16_supported'] else 'Fallback required'}")
        else:
            logger.warning("CUDA is NOT available. Falling back to CPU mode.")
            logger.warning("To enable GPU acceleration, install PyTorch with CUDA wheels:")
            logger.warning("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")

    except ImportError:
        logger.error("PyTorch is not installed in the current environment.")
        info["error"] = "torch_not_installed"

    return info


def main() -> int:
    """Entry point for command line execution."""
    print("=" * 60)
    print(" BRANDPULSE AI: GPU & HARDWARE VERIFICATION")
    print("=" * 60)
    info = get_gpu_info()
    print("-" * 60)
    for k, v in info.items():
        print(f"{k:25s}: {v}")
    print("=" * 60)
    return 0 if info["cuda_available"] else 1


if __name__ == "__main__":
    sys.exit(main())
