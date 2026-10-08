import os


def is_directml(device):
    return str(device).startswith("privateuseone")


def get_device(torch, backend="auto"):
    backend = os.environ.get("CATVTON_BACKEND", backend).lower()
    if backend == "cpu":
        return "cpu"
    if backend == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("CUDA was requested but is not available.")
    if backend == "directml":
        import torch_directml

        return torch_directml.device()
    if backend == "auto" and torch.cuda.is_available():
        return "cuda"
    if backend == "auto":
        try:
            import torch_directml

            return torch_directml.device()
        except (ImportError, RuntimeError):
            return "cpu"
    raise ValueError(f"Unknown CatVTON backend: {backend}")


def get_dtype(torch, device):
    if str(device) == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def get_inference_profile(device):
    if str(device) == "cuda":
        return 768, 1024, 30
    return 384, 512, 10


def get_mask_device(device):
    return "cpu" if is_directml(device) else device


def make_generator(torch, device, seed):
    if seed is None or int(seed) < 0:
        return None
    generator_device = "cpu" if is_directml(device) else device
    return torch.Generator(device=generator_device).manual_seed(int(seed))


def configure_cpu_threads(torch, device):
    if str(device) == "cpu":
        thread_count = int(os.environ.get("CATVTON_CPU_THREADS", min(8, os.cpu_count() or 1)))
        torch.set_num_threads(max(1, thread_count))