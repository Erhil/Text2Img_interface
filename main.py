import streamlit as st
import torch

import generative_pipes

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
    pipeline = None
    prev_model = None

    st.header("Select generation parameters")
    mode = st.selectbox('Select mode', ('generator', "refiner"))
    gen_models = generative_pipes.__all__

    #
    if mode == "generator":
        model = st.selectbox('Select model', gen_models)

        if model != prev_model:
            del pipeline
            pipeline = getattr(generative_pipes, model)()

        res = pipeline.generate_form()
        gen = st.button("Generate")
        if gen:
            st.write("generating...")
            images = pipeline.generate_image(**res)
            torch.cuda.empty_cache()

            for idx, image in enumerate(images):
                st.image(image, use_container_width=False)
            gen = False
    elif mode == "refiner":
        st.write("refiner")


if __name__ == "__main__":
    main()