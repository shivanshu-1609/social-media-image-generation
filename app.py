"""Streamlit app for image and meme generation."""

import io
import zipfile

import streamlit as st
from generate import MODEL_ID, STYLES, generate, image_bytes
from PIL import Image


def main():
    st.title("Marketing Image Generator")
    prompt = st.text_area("Image prompt")
    caption = st.text_input("Caption (optional)", max_chars=240)
    style = st.sidebar.selectbox("Style", list(STYLES))
    count = st.sidebar.slider("Images", 1, 4, 1)
    steps = st.sidebar.slider("Inference steps", 1, 100, 30)
    guidance = st.sidebar.slider("Guidance scale", 0.0, 15.0, 7.5)
    seed = st.sidebar.number_input("Seed", min_value=0, max_value=2147483647, value=42)
    with st.sidebar.expander("Model"):
        model_id = st.text_input("Model ID or local path", MODEL_ID)
        lora = st.text_input("Compatible LoRA (optional)")
    if st.button("Generate"):
        try:
            with st.spinner("Generating images"):
                images = generate(
                    prompt,
                    count,
                    int(seed),
                    steps,
                    guidance,
                    style,
                    caption,
                    model_id,
                    lora or None,
                )
            st.session_state["generated_images"] = [
                image_bytes(image) for image in images
            ]
        except Exception as error:
            st.error(str(error))
    images = st.session_state.get("generated_images", [])
    if images:
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as handle:
            for index, content in enumerate(images):
                with Image.open(io.BytesIO(content)) as image:
                    st.image(image.copy(), caption=f"Image {index + 1}")
                handle.writestr(f"image_{index:03d}.png", content)
        st.download_button(
            "Download images",
            archive.getvalue(),
            "generated_images.zip",
            "application/zip",
        )


if __name__ == "__main__":
    main()
