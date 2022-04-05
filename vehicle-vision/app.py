# from cProfile import label
# from msilib.schema import CheckBox
# import os
# import json
# # from tkinter import Button
# from xml.etree.ElementTree import VERSION
# from nbformat import write
# import requests
import SessionState
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

st.title("Welcome to Vehicle Vision 🚗📸")
st.header("Identify what type of vehicle is in the picture!")

base_classes = ['Ambulance',
    'Barge',
    'Bicycle',
    'Boat',
    'Bus',
    'Car',
    'Cart',
    'Caterpillar',
    'Helicopter',
    'Limousine',
    'Motorcycle',
    'Segway',
    'Snowmobile',
    'Tank',
    'Taxi',
    'Truck',
    'Van']

# Model Choice
choose_model = st.sidebar.selectbox(
    "Pick model you'd like to use",
    ("Model 1: EffecientNetB3","Model 2: EfficientNetB5")
)

if choose_model == "Model 1: EffecientNetB3":
    CLASSES = base_classes
    MODEL = tf.keras.models.load_model("model_EfficientNetB3.h5")

if choose_model == "Model 2: EfficientNetB5":
    CLASSES = base_classes
    MODEL = tf.keras.models.load_model("model_EfficientNetB5.h5")

# @st.cache
def make_prediction(image, model, class_names):
    """
    Takes an image and uses model (a trained TensorFlow model) to make a
    prediction.
    Returns:
     image (preproccessed)
     pred_class (prediction class from class_names)
     pred_conf (model confidence)
    """
    img = tf.image.decode_jpeg(image)
    img = tf.image.resize(img, [224,224])
    img=preprocess_input(img)
    img= tf.cast(tf.expand_dims(img, axis=0), tf.int16)
    prediction=model.predict(img)

    pred_class=class_names[tf.argmax(prediction[0])]
    pred_prob=(tf.reduce_max(prediction)*100).numpy()
    return image,pred_class,pred_prob

#Upload File
uploaded_file=st.file_uploader(label="Upload an Image of a Vehicle", type=["png","jpeg","jpg"])
#Start session State so that site does not load on every interaction
session_state = SessionState.get(pred_button=False)

# Create logic for app flow
if not uploaded_file:
    st.warning("Please upload an image.")
    st.stop()
else:
    session_state.uploaded_image = uploaded_file.read()
    st.image(session_state.uploaded_image, use_column_width=True)
    pred_button = st.button("Predict")

# Did the user press the predict button?
if pred_button:
    session_state.pred_button = True 
# Make Prediction
if session_state.pred_button:
    session_state.image, session_state.pred_class, session_state.pred_conf = make_prediction(session_state.uploaded_image, model=MODEL, class_names=CLASSES)
    st.write(f"Prediction: {session_state.pred_class}, \
               Confidence: {session_state.pred_conf:.2f}%")