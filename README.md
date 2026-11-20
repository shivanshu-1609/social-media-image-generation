# Social Media Marketing in the United States

Text-to-image generation and captioned marketing visuals using Stable Diffusion,
with a Streamlit application and command-line interface.

## Model

The default model is [Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4).
The pipeline uses pretrained text encoding, iterative latent denoising, and VAE
image decoding. This project performs inference; it does not claim to train the
base model or fine-tune a LoRA adapter.

## Setup and usage

Use Python 3.11 and a separate virtual environment.

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Command-line generation:

```bash
python generate.py --prompt "A coffee cup on a wooden table, morning sunlight" --style photo --caption "Start fresh" --output outputs/coffee
```

The app supports prompt, style, inference steps, guidance scale, seed, optional
caption, image count, and downloads. A compatible LoRA can be loaded using `--lora`
or the model options in the app. LoRA compatibility depends on its base model.

CUDA uses float16; CPU uses float32 and can be slow. Weights are downloaded on
first use and are not included in the repository. Model access may require
accepting its hosting terms. The pipeline retains its safety checker.

## Outputs and limitations

The CLI writes PNG images; the app offers a ZIP download. Captions are rendered
with Pillow after generation. Seeds improve repeatability, but exact outputs can
vary with hardware and library versions. Generated visuals can contain artifacts
and require review before use. No quantitative marketing-performance claim is made.

## References and license

[WorldQuant University](https://www.wqu.edu/) provided the original learning project.
[Diffusers documentation](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/text2img)
describes the generation pipeline. Standalone scripts and documentation use MIT;
the Stable Diffusion weights and any LoRA retain their respective model licenses.
