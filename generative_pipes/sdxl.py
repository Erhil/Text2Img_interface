import streamlit as st

from diffusers import (AutoPipelineForText2Image, DiffusionPipeline)
import diffusers
import torch
from PIL import Image


def generate_form():
    prompt = st.text_area("Enter prompt")
    negative_prompt = st.text_area("Enter negative prompt",
                                   help='This is a negative prompt, basically type what' \
                                        'you don\'t want to see in the generated image') or None
    with st.expander("advanced settings"):
        device = st.selectbox("Select device", ["cpu", "cuda"])
        width = st.number_input("Width", value=1024)
        height = st.number_input("Height", value=1024)
        num_steps = st.slider("Num steps", value=50, min_value=1, max_value=500)
        num_outputs = st.slider("Num outputs", value=1, min_value=1, max_value=10)
    return {"prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_steps": num_steps,
            "num_outputs": num_outputs,
            "device": device}

def generate_image(prompt, negative_prompt, width, height, num_steps, num_outputs, device):
    pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    # pipeline_text2image.enable_model_cpu_offload()
    pipeline_text2image.to(device)

    image_gen = pipeline_text2image(prompt=prompt,
                                    negative_prompt=negative_prompt,
                                    num_inference_steps=num_steps,
                                    target_size=(width,height)).images[0]
    return image_gen