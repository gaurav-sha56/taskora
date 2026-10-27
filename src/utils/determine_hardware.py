import os
import ctranslate2


def get_optimal_hardware_config() -> tuple[str, str, int]:
  """Dynamically determines the best (device, compute_type, cpu_threads) across NVIDIA GPUs, Intel/AMD CPUs, and Apple Silicon."""
  # 1. Determine Device (CUDA GPU vs CPU)
  device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"

  # 2. Query CTranslate2 for hardware-supported precisions on this machine
  supported = ctranslate2.get_supported_compute_types(device)

  # 3. Rank precisions from fastest/best to safest fallback
  if device == "cuda":
    # On GPU: int8_float16 uses Tensor Cores + half the VRAM of float16
    preference_order = ["int8_float16", "float16", "int8", "float32"]
  else:
    # On CPU: int8 uses Intel VNNI / AVX2 / ARM NEON
    preference_order = ["int8", "int8_float32", "float32"]

  compute_type = next(
      (t for t in preference_order if t in supported), "default"
  )

  # 4. Optimal CPU thread count (avoids E-core straggler bottleneck on hybrid CPUs)
  logical_cores = os.cpu_count() or 4
  cpu_threads = min(4, max(1, logical_cores // 2))

  return device, compute_type, cpu_threads


