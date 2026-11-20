"""Generate styled images and captioned marketing visuals with Stable Diffusion."""

import argparse
import io
import textwrap
from functools import lru_cache
from pathlib import Path

import torch
from PIL import ImageDraw, ImageFont

MODEL_ID = "CompVis/stable-diffusion-v1-4"
STYLES = {
    "none": "",
    "photo": "editorial photograph, natural lighting",
    "watercolor": "watercolor painting, soft colors",
    "illustration": "digital illustration, clean composition",
}


@lru_cache(maxsize=1)
def load_pipeline(model_id=MODEL_ID, lora=None):
    from diffusers import StableDiffusionPipeline

    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if target_device == "cuda" else torch.float32
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    if lora:
        pipeline.load_lora_weights(lora)
    pipeline.to(target_device)
    pipeline.enable_attention_slicing()
    return pipeline


def add_caption(image, caption):
    result = image.convert("RGB").copy()
    if not caption.strip():
        return result
    size = max(12, result.width // 12)
    font = ImageFont.load_default(size=size)
    lines = textwrap.wrap(
        caption.strip(), width=max(1, int((result.width - 24) / (size * 0.65)))
    )
    text = "\n".join(lines)
    draw = ImageDraw.Draw(result)
    bounds = draw.multiline_textbbox(
        (0, 0), text, font=font, stroke_width=2, align="center"
    )
    x = (result.width - (bounds[2] - bounds[0])) / 2 - bounds[0]
    y = max(4, result.height - (bounds[3] - bounds[1]) - 16) - bounds[1]
    draw.multiline_text(
        (x, y),
        text,
        font=font,
        fill="white",
        stroke_fill="black",
        stroke_width=2,
        align="center",
    )
    return result


def generate(
    prompt,
    count=1,
    seed=42,
    steps=30,
    guidance=7.5,
    style="none",
    caption="",
    model_id=MODEL_ID,
    lora=None,
    pipeline=None,
):
    if not prompt.strip() or not 1 <= count <= 4 or not 1 <= steps <= 150:
        raise ValueError("Provide a prompt, 1–4 images, and 1–150 inference steps.")
    if not 0 <= guidance <= 20 or seed < 0 or style not in STYLES:
        raise ValueError("Invalid guidance, seed, or style.")
    if len(caption) > 240:
        raise ValueError("Keep captions within 240 characters.")
    pipeline = pipeline if pipeline is not None else load_pipeline(model_id, lora)
    styled_prompt = ", ".join(part for part in (prompt.strip(), STYLES[style]) if part)
    result = pipeline(
        [styled_prompt] * count,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=torch.Generator().manual_seed(seed),
    )
    flags = result.nsfw_content_detected
    if flags is not None and any(flags):
        raise ValueError(
            "The model's safety checker rejected an image. Use another prompt."
        )
    return [add_caption(image, caption) for image in result.images]


def image_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--caption", default="")
    parser.add_argument("--style", choices=list(STYLES), default="none")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--lora")
    parser.add_argument("--output", type=Path, default=Path("outputs/generated"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty output directory.")
    try:
        images = generate(
            args.prompt,
            args.count,
            args.seed,
            args.steps,
            args.guidance,
            args.style,
            args.caption,
            args.model_id,
            args.lora,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(images):
            image.save(args.output / f"image_{index:03d}.png")
        print(f"Saved {len(images)} images in {args.output}")
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")
