import streamlit as st
import pytesseract
import cv2
from PIL import Image
import torch
import pandas as pd
import numpy as np
from functions import get_bounding_box, get_box_coordinates, get_coordinates


st.text("Access the Axes")
uploaded_file = st.file_uploader("Upload an image of a graph - PDF and CDF", type=['png', 'jpeg', 'jpg'])
axis_selection = st.radio("Select axis to process", options=["x-axis", "y-axis"])
delta = st.slider("Bounding Box Padding", min_value=0, max_value=10, value=1)

if uploaded_file:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Detecting elements...")
    bbox_data, result = get_bounding_box(image)

    st.write("Objects detected.....")
    result.render()  # Updates the image with bounding boxes
    rendered_image = result.ims[0]
    st.image(rendered_image)

    box_coordinates = get_box_coordinates(bbox_data)

    data = get_coordinates(image, box_coordinates, axis_selection, delta)
    json_data = {"Extracted Numbers": data}
    st.markdown("### Extracted Data")
    st.json(json_data)

    st.write("Detection and extraction completed!")
