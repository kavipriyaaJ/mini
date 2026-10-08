import os
import sys
import traceback
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from catvton_runtime import (
    configure_cpu_threads,
    get_device,
    get_dtype,
    //
    get_inference_profile,
    get_mask_device,
    is_directml,
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


def build_runtime():
    from model.cloth_masker import AutoMasker, vis_mask
    from model.pipeline import CatVTONPipeline
    from utils import resize_and_crop, resize_and_padding

    return AutoMasker, vis_mask, CatVTONPipeline, resize_and_crop, resize_and_padding


try:
    ensure_required_files()
    AutoMasker, vis_mask, CatVTONPipeline, resize_and_crop, resize_and_padding = build_runtime()
    RUNTIME_IMPORT_ERROR = None
except Exception as exc:
    AutoMasker = None
    vis_mask = None
    CatVTONPipeline = None
    resize_and_crop = None
    resize_and_padding = None
    RUNTIME_IMPORT_ERROR = exc


mask_processor = VaeImageProcessor(
    vae_scale_factor=8,
    do_normalize=False,
    do_binarize=True,
    do_convert_grayscale=True,
)


def gpu_status_text():
    device = get_device(torch)
    if str(device) == "cuda":
        return f"CUDA detected: {torch.cuda.get_device_name(0)}"
    if is_directml(device):
        return f"DirectML detected: {device}"
    return "CPU mode: low-memory settings are enabled. Generation will be slow."


def run_tryon(person_file, cloth_file, cloth_type, steps, guidance_scale, seed, show_mode, resolution):
    if person_file is None:
        raise gr.Error("Please upload a person image.")
    if cloth_file is None:
        raise gr.Error("Please upload a clothing image.")
    if AutoMasker is None or CatVTONPipeline is None:
        raise gr.Error(f"CatVTON runtime could not be loaded: {RUNTIME_IMPORT_ERROR}")

    device = get_device(torch)
    dtype = get_dtype(torch, device)
    width, height = (int(value) for value in resolution.split("x"))
    if device != "cuda":
        configure_cpu_threads(torch, device)
        print(f"Running CatVTON on {device} at {width}x{height}; this may take a long time.")

    try:
        person = Image.open(person_file).convert("RGB")
        cloth = Image.open(cloth_file).convert("RGB")

        person = resize_and_crop(person, (width, height))
        cloth = resize_and_padding(cloth, (width, height))

        automasker = AutoMasker(
            densepose_ckpt=str(DENSEPOSE_DIR),
            schp_ckpt=str(SCHP_DIR),
            device=get_mask_device(device),
        )

        mask = automasker(person, cloth_type)["mask"]
        mask = mask_processor.blur(mask, blur_factor=9)

        pipeline = CatVTONPipeline(
            base_ckpt=BASE_MODEL,
            attn_ckpt=str(CATVTON_MODEL),
            attn_ckpt_version="vitonhd",
            weight_dtype=dtype,
            use_tf32=(device == "cuda"),
            device=device,
            skip_safety_check=True,
        )
        generator = make_generator(torch, device, seed)

        result = pipeline(
            image=person,
            condition_image=cloth,
            mask=mask,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
            width=width,
            height=height,
        )[0]

        if show_mode == "result only":
            return result

        if show_mode == "input & result":
            canvas = Image.new("RGB", (person.width * 2 + 10, person.height))
            canvas.paste(person, (0, 0))
            canvas.paste(result, (person.width + 10, 0))
            return canvas

        masked_person = vis_mask(person, mask)
        canvas = Image.new("RGB", (person.width * 4 + 30, person.height))
        canvas.paste(person, (0, 0))
        canvas.paste(masked_person, (person.width + 10, 0))
        canvas.paste(cloth, (person.width * 2 + 20, 0))
        canvas.paste(result, (person.width * 3 + 30, 0))
        return canvas
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(f"An error occurred while running CatVTON: {exc}")


with gr.Blocks(title="CatVTON Virtual Try-On") as demo:
    default_width, default_height, default_steps = get_inference_profile(get_device(torch))
    default_resolution = f"{default_width}x{default_height}"
    gr.Markdown("## CatVTON Virtual Try-On")
    gr.Markdown(value=gpu_status_text())

    with gr.Row():
        person_input = gr.Image(label="Person image", type="filepath", sources=["upload"])
        cloth_input = gr.Image(label="Clothing image", type="filepath", sources=["upload"])

    cloth_type = gr.Radio(["upper", "lower", "overall"], value="upper", label="Clothing Type")
    steps = gr.Slider(minimum=4, maximum=50, value=default_steps, step=1, label="Inference Steps")
    guidance = gr.Slider(minimum=1.0, maximum=5.0, value=2.5, step=0.1, label="Guidance Scale")
    seed = gr.Number(value=42, precision=0, label="Seed")
    resolution = gr.Radio(
        choices=["384x512", "512x680", "768x1024"],
        value=default_resolution,
        label="Resolution",
    )
    show_mode = gr.Radio(["result only", "input & result", "all inputs + result"], value="result only", label="Display Mode")

    output = gr.Image(label="Generated Result", type="pil")
    generate_button = gr.Button("Generate", interactive=True)
    generate_button.click(
        fn=run_tryon,
        inputs=[person_input, cloth_input, cloth_type, steps, guidance, seed, show_mode, resolution],
        outputs=output,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=True,
    )
