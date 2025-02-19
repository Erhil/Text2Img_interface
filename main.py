import streamlit as st

from PIL import Image
import numpy as np
import torch

from generative_pipes import sdxl

st.set_page_config(page_title="T2I Generator",
                   layout="wide")

def main():
    """
    Main function to run the Streamlit application.

    This function initializes the sidebar configuration and the main page layout.
    It retrieves the user inputs from the sidebar, and passes them to the main page function.
    The main page function then generates images based on these inputs.
    """
    gen = False

    st.header("Select generation parameters")
    mode = st.selectbox('Select mode', ('generator', "refiner"))
    #
    if mode == "generator":
        model = st.selectbox('Select model', ('SDXL', "SD3.5_l", "SD3.5_m", "FLUX.1-dev"))

        if model == "SDXL":
            res = sdxl.generate_form()
        elif model == "SD3.5_l":
            st.write("SD3.5_l")
        elif model == "SD3.5_m":
            st.write("SD3.5_m")
        elif model == "FLUX.1-dev":
            st.write("FLUX.1-dev")

        gen = st.button("Generate")
        if gen:
            st.write("generating...")
            st.image(sdxl.generate_image(**res), use_container_width=False)
    elif mode == "refiner":
        st.write("refiner")


if __name__ == "__main__":
    main()