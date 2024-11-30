import streamlit as st
import diffusion
import numpy as np
import tensorflow as tf


LATENT_DIMS = 100
st.set_page_config("Image AI")
st.title("Generative AI for Images")

if not 'model' in st.session_state:
    st.session_state['model'] = tf.keras.models.load_model('model/generator.keras')


col_diff, col_gan = st.columns(2)

with col_diff:
    st.subheader("Dall-e 3 Diffusion Model")
    with st.form("Diffusion", clear_on_submit=True):
        prompt = st.text_input("Describe the image")
        key = st.text_input("OpenAI API Key. Get one [here](https://platform.openai.com/api-keys)")
        submit = st.form_submit_button("Enter")
    if prompt and key and submit:
        with st.spinner("Generating image..."):
            image_path = diffusion.get_image(prompt, key)
        st.image(image_path)

with col_gan:
    st.subheader("Generates a Pokemon Image using GAN")
    generate = st.button("Generate")
    if generate:
        noise = np.random.normal(0, 1, (1, LATENT_DIMS))
        image = st.session_state['model'].predict(noise)[0]
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        st.image(image, width=2*image.shape[0])