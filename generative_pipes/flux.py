import streamlit as st

from diffusers import DiffusionPipeline
import torch

from .base_model import BaseModel


class FLUXModel(BaseModel):
    models = {
        # "base": "black-forest-labs/FLUX.1-dev",
        "NSFW niji56": "John6666/niji56-style-v3-fp8-flux",
        # "NSFW horny" : "John6666/real-horny-v2-v2unet-fp8-flux",
        "NSFW hentai": "John6666/xe-hentai-flux-01-fp8-flux",
        "NSFW jib-mix" : "John6666/jib-mix-flux-v8accentueight-nsfw-bf16-flux",
    }

    def generate_form(self):
        prompt = st.text_area("Enter prompt")
        negative_prompt = st.text_area("Enter negative prompt",
                                       help='This is a negative prompt, basically type what' \
                                            'you don\'t want to see in the generated image') or None
        with st.expander("advanced settings"):
            model = st.selectbox("Model name", list(self.models.keys()))
            cpu_offload = st.selectbox("CPU_offload", [True, False])
            device = st.selectbox("Select device", ["cuda", "cpu"])
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
                "device": device,
                "cpu_offload": cpu_offload,
                "model": model,
                }

    def load_model(self, model_name, cpu_offload, device):
        self.pipeline_text2image = DiffusionPipeline.from_pretrained(
            model_name
        )
        if cpu_offload:
            self.pipeline_text2image.enable_model_cpu_offload()
        else:
            self.pipeline_text2image.to(device)

        self.parameters = {"model_name": model_name, "cpu_offload": cpu_offload, "device": device}

    def offload_model(self):
        del self.pipeline_text2image
        torch.cuda.empty_cache()

    def generate_image(self, prompt, negative_prompt, width, height,
                       num_steps, num_outputs, device,
                       cpu_offload, model):
        model_name = self.models[model]
        parameters = {"model_name": model_name, "cpu_offload": cpu_offload, "device": device}

        if self.parameters is None:
            self.load_model(**parameters)
        elif self.parameters != parameters:
            self.offload_model()
            self.load_model(**parameters)

        image_gen = self.pipeline_text2image(prompt=prompt,
                                        negative_prompt=negative_prompt,
                                        num_inference_steps=num_steps,
                                        height=height,
                                        width=width,
                                        num_images_per_prompt=num_outputs).images
        torch.cuda.empty_cache()
        return image_gen
