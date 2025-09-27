import streamlit as st

from diffusers import DiffusionPipeline
import torch
from diffusers import SanaPipeline
from .base_model import BaseModel


class SanaModel(BaseModel):
    models = {
        "base": "Efficient-Large-Model/Sana_1600M_1024px",
        "twistedreality" : "frutiemax/twistedreality-sana-1600m-1024px"


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
        self.pipeline_text2image = SanaPipeline("configs/sana_config/1024ms/Sana_1600M_img1024.yaml")
        self.pipeline_text2image.from_pretrained(f"hf://{model_name}")
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
