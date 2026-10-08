import argparse
import sys
import time
import traceback
from pathlib import Path

import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from catvton_runtime import (
    configure_cpu_threads,
    get_device,
    get_dtype,
    get_inference_profile,
    get_mask_device,
    make_generator,
)

BASE_DIR = Path(__file__).resolve().parent
CATVTON_SOURCE = BASE_DIR / "CatVTON_source"
if str(CATVTON_SOURCE) not in sys.path:
    sys.path.insert(0, str(CATVTON_SOURCE))

CATVTON_MODEL = BASE_DIR / "CatVTON_model"
VITON_CHECKPOINT = CATVTON_MODEL / "vitonhd-16k-512"
DENSEPOSE_DIR = CATVTON_MODEL / "DensePose"
SCHP_DIR = CATVTON_MODEL / "SCHP"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

BASE_MODEL = "booksforcharlie/stable-diffusion-inpainting"


def ensure_required_files():
    missing = [
        str(path)
        for path in [CATVTON_MODEL, VITON_CHECKPOINT, DENSEPOSE_DIR, SCHP_DIR]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Required CatVTON model files were not found:\n" + "\n".join(missing)
        )


def load_runtime():
    from model.cloth_masker import AutoMasker
    from model.pipeline import CatVTONPipeline
    from utils import resize_and_crop, resize_and_padding

    return AutoMasker, CatVTONPipeline, resize_and_crop, resize_and_padding


def print_runtime_summary():
    print("=" * 60)
    print("CATVTON LOCAL TEST")
    print("=" * 60)
    print("\nGPU:")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  {gpu_name}")
        print(f"  VRAM: {vram:.2f} GB")
    else:
        print(f"  Using {get_device(torch)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Model: {VITON_CHECKPOINT}")
    print(f"Base model: {BASE_MODEL}")
    print(f"Output: {OUTPUT_DIR / 'test_result.png'}")


def generate_result(person_path, cloth_path, output_path, cloth_type="upper", steps=10, guidance_scale=2.5, seed=42, width=384, height=512):
    ensure_required_files()
    AutoMasker, CatVTONPipeline, resize_and_crop, resize_and_padding = load_runtime()

    device = get_device(torch)
    dtype = get_dtype(torch, device)
    if device != "cuda":
        configure_cpu_threads(torch, device)
        print(f"Running on {device} at {width}x{height}. Inference may be slow.")
    mask_processor = VaeImageProcessor(
        vae_scale_factor=8,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True,
    )

    automasker = AutoMasker(
        densepose_ckpt=str(DENSEPOSE_DIR),
        schp_ckpt=str(SCHP_DIR),
        device=get_mask_device(device),
    )

    pipeline = CatVTONPipeline(
        base_ckpt=BASE_MODEL,
        attn_ckpt=str(CATVTON_MODEL),
        attn_ckpt_version="vitonhd",
        weight_dtype=dtype,
        use_tf32=(device == "cuda"),
        device=device,
        skip_safety_check=True,
    )
    person_image = Image.open(person_path).convert("RGB")
    cloth_image = Image.open(cloth_path).convert("RGB")
    person_image = resize_and_crop(person_image, (width, height))
    cloth_image = resize_and_padding(cloth_image, (width, height))

    mask = automasker(person_image, cloth_type)["mask"]
    mask = mask_processor.blur(mask, blur_factor=9)

    generator = make_generator(torch, device, seed)

    start = time.time()
    result = pipeline(
        image=person_image,
        condition_image=cloth_image,
        mask=mask,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
        width=width,
        height=height,
    )[0]
    elapsed = time.time() - start

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)

    print("\nGPU:")
    if torch.cuda.is_available():
        print(f"  {torch.cuda.get_device_name(0)}")
    else:
        print("  CUDA GPU not available")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Model: {VITON_CHECKPOINT}")
    print(f"Input: {person_path} + {cloth_path}")
    print(f"Output: {output_path}")
    print(f"Inference time: {elapsed:.2f} seconds")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run a local CatVTON try-on inference.")
    parser.add_argument("--person", type=str, required=True, help="Path to person image.")
    parser.add_argument("--cloth", type=str, required=True, help="Path to clothing image.")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR / "test_result.png"), help="Output path for result image.")
    parser.add_argument("--cloth-type", type=str, default="upper", choices=["upper", "lower", "overall"], help="Type of garment.")
    parser.add_argument("--steps", type=int, default=None, help="Inference steps.")
    parser.add_argument("--width", type=int, default=None, help="Output width; use multiples of 8.")
    parser.add_argument("--height", type=int, default=None, help="Output height; use multiples of 8.")
    parser.add_argument("--guidance", type=float, default=2.5, help="Guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Set -1 for random.")
    return parser.parse_args()


def main():
    args = parse_args()
    profile_width, profile_height, profile_steps = get_inference_profile(get_device(torch))
    args.width = args.width or profile_width
    args.height = args.height or profile_height
    args.steps = args.steps or profile_steps
    print_runtime_summary()
    try:
        output_path = generate_result(
            person_path=args.person,
            cloth_path=args.cloth,
            output_path=args.output,
            cloth_type=args.cloth_type,
            steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            width=args.width,
            height=args.height,
        )
        print(f"\nSaved result to: {output_path}")
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
