[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/shehroz218/vehicle-vision/main/vehicle-vision/app.py)


# Vehicle-Vision
+ The Dataset was creating using Kaggle, Stanford vehicle Images datset and google images to reduce the bias found in the original datset.
+ Transfer learning was used on the MobileNet and EffecientNetB0 models. EfficientNetB0 performs better and ais able to bypass the google clouds 1.5 MB limit
+ The models are uploaded on google cloud and are not run locally. (The Json keys ar enot uploaded in the script for this to work on your mahine you will need your own google AI engine instance and json keys)
+ A combination of Session state and streamlit was used to produce a user friendly vehicle recognition application


<p align="center">
<img src="/dataset_bias.png"></img>
</p>
