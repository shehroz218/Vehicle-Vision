from cProfile import label
from msilib.schema import CheckBox
import os
import json
from tkinter import Button
from xml.etree.ElementTree import VERSION
from nbformat import write
import requests
import SessionState
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import preprocess_input
tf.keras.mixed_precision.set_global_policy('mixed_float16')
from utils import load_and_prep_image, classes_and_models, predict_json

# Environment credentials for accessing google cloud account
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:\Codes\Image-Classification\shehroz-dl-playground-218-4cc90568b002.json" # change for your GCP key
PROJECT = "shehroz-dl-playground-218" 
REGION = "us-central1" 





### Streamlit code (works as a straigtht-forward script) ###
st.title("Welcome to Vehicle Vision 🚗📸")
st.header("Identify what type of vehicle is in the picture!")




@st.cache
def make_prediction(image, model, class_names):
    """
    Takes an image and uses model (a trained TensorFlow model) to make a
    prediction.
    Returns:
     image (preproccessed)
     pred_class (prediction class from class_names)
     pred_conf (model confidence)
    """


    
    image = load_and_prep_image(image)
    
    prediction = predict_json(project=PROJECT,
                         region=REGION,
                         model=model,
                         instances=image)
    pred_class = class_names[tf.argmax(prediction[0])]
    pred_conf = tf.reduce_max(prediction[0])*100
    return image, pred_class, pred_conf


# Model Choice
choose_model = st.sidebar.selectbox(
    "Pick model you'd like to use",
    ("Model 1","Model 2")
)

if choose_model == "Model 1":
    CLASSES = classes_and_models["model_1"]["classes"]
    MODEL = classes_and_models["model_1"]["model_name"]

#Display info about chosen model
if st.checkbox('Show Model Classes'):
    st.write(f"You chose {MODEL}, these are the classes of vehicle it can identify:\n", CLASSES)

# Upload image to run model on
uploaded_file=st.file_uploader(label="Upload an Image of a Vehicle", type=["png","jpeg","jpg"])



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

# And if they did...
if session_state.pred_button:
    session_state.image, session_state.pred_class, session_state.pred_conf = make_prediction(session_state.uploaded_image, model=MODEL, class_names=CLASSES)
    st.write(f"Prediction: {session_state.pred_class}, \
               Confidence: {session_state.pred_conf:.3f}%")




