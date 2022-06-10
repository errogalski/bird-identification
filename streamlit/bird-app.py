#imports
import streamlit as st
import pandas as pd
from streamlit_cropper import st_cropper
import numpy as np
from PIL import Image
import time
import tensorflow as tf
from tensorflow import keras
import os

print(os.listdir(os.getcwd()))
bird_names = pd.read_csv(os.getcwd()+'/BirdLabels.csv')

def generate_prediction(img):
    img = img.resize((224,224))
    img_array = np.array(img)
    img_array = img_array.reshape(1,224,224,3)
    st.image(img_array)
    prediction = model.predict(img_array)
    df = pd.DataFrame(prediction.T)

    df = df.rename(columns={0:'Confidence'})
    df2 = pd.merge(bird_names, df, left_index=True, right_index=True)

    df2 = df2.sort_values(by='Confidence', ascending=False)
    st.write(df2.head(5))


st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("North American Bird Identifier")
st.write("1) Upload an image of a bird")
st.write("2) Place a box around the bird you wish to identify")


model = keras.models.load_model('../final-model/EFSave5.h5')

img_file = st.sidebar.file_uploader(label='Upload an image here', type=['jpg', 'jpeg'])
realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "Free"])
aspect_dict = {
    "1:1": (1, 1),
    "Free": None
}
aspect_ratio = aspect_dict[aspect_choice]

if img_file:
    img = Image.open(img_file)
    if not realtime_update:
        st.write("Double click to save crop")
    # Get a cropped image from the frontend
    cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                                aspect_ratio=aspect_ratio)
    
    # Manipulate cropped image at will
    st.write("Preview")
    _ = cropped_img.thumbnail((150,150))

    if st.button('Predict!', on_click=generate_prediction(cropped_img)):
        with st.spinner("Predicting..."):
            time.sleep(3)

