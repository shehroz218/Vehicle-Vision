# Utils for preprocessing data etc 
# from turtle import st
import tensorflow as tf
import googleapiclient.discovery
from google.api_core.client_options import ClientOptions
from tensorflow.keras.applications.mobilenet import preprocess_input
import numpy as np
import cv2
import base64
from tensorflow.keras.preprocessing import image

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



classes_and_models = {
    "model_1": {
        "classes": base_classes,
        "model_name": "vehicle_vision_218_model" 
    }
}

# Code snippet taken from Google resource
def predict_json(project, region, model, instances, version=None):
    """Send json data to a deployed model for prediction.
    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to Tensors.
        version (str): version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the 
            model.
    """
    # Create the ML Engine service object
    prefix = "{}-ml".format(region) if region else "ml"
    api_endpoint = "https://{}.googleapis.com".format(prefix)
    client_options = ClientOptions(api_endpoint=api_endpoint)

    # Setup model path
    model_path = "projects/{}/models/{}".format(project, model)
    if version is not None:
        model_path += "/versions/{}".format(version)

    # Create ML engine resource endpoint and input data
    ml_resource = googleapiclient.discovery.build(
        "ml", "v1", cache_discovery=False, client_options=client_options).projects()
        
    instances_list = instances.numpy().tolist() # turn input into list (ML Engine wants JSON)
    
    input_data_json = {"signature_name": "serving_default",
                       "instances": instances_list} 

    request = ml_resource.predict(name=model_path, body=input_data_json)
    response = request.execute()


    if "error" in response:
        raise RuntimeError(response["error"])

    return response["predictions"]

# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=224, rescale=False):
  """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (224, 224, 3).
  """
  #decode file into a tensor to feed into the model
  img = tf.image.decode_jpeg(filename)
  #Resixe Image to fir your mode
  img = tf.image.resize(img, [224,224])
  #expand dimensions and convert into int16 because google AI json input current size limit is 1.5mb
  img= tf.cast(tf.expand_dims(img, axis=0), tf.int16)


  #img=preprocess_input(img)
  # Rescale the image (get all values between 0 and 1)
  if rescale:
      return img/255.
  else:
      return img

# def update_logger(image, model_used, pred_class, pred_conf, correct=False, user_label=None):
#     """
#     Function for tracking feedback given in app, updates and reutrns 
#     logger dictionary.
#     """
#     logger = {
#         "image": image,
#         "model_used": model_used,
#         "pred_class": pred_class,
#         "pred_conf": pred_conf,
#         "correct": correct,
#         "user_label": user_label
#     }   
#     return logger
