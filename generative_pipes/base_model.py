import streamlit as st

import torch

class BaseModel:
    instance = None
    parameters = None

    def __new__(cls, *args, **kwargs):
        if not cls.instance:
            cls.instance = super().__new__(cls)
        return cls.instance

    def load_model(self):
        raise NotImplementedError

    def offload_model(self):
        raise NotImplementedError

    def generate_form(self):
        raise NotImplementedError

    def generate_image(self):
        raise NotImplementedError

