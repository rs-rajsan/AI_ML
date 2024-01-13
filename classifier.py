# taipy has much more control componets for both frontend (including HTML & Markdown elements) and backend 
from taipy.gui import Gui

# Class names defined as dictionary
class_names = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}

# import os statements was added to ignore the below warning 
"""
2024-01-13 13:45:39.844912: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE SSE2 SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

'''
Once the data is downloaded and model is created and trained then we import the model from tensorflow
'''
# from tensorflow.keras import models
# Since above import statement didnt work had to break down the statement to following
# tensorflow is the library where the model is built in jupyter notebook
import tensorflow as tf             # importing tensorflow 
keras = tf.keras                    # creating keras object
models = tf.keras.models            # creating models object

# model = models.load_model("C:\\Users\\rajsa\\OneDrive\\Documents\\Python Projects\\AI_ML_Projects\\baseline_mariya.keras")
model = models.load_model("C:\\Users\\rajsa\\OneDrive\\Documents\\Python Projects\\AI_ML_Projects\\baseline.keras")

# Image object imported to process the image in the model
from PIL import Image

# importing numpy for math calculation.
# This is to normarlize the image to 0s and 1s since the color code is from 0 to 255
import numpy as np

def predict_img(model, path_to_image):
    # print(model.summary())
    # print(path_to_image)
    img = Image.open(path_to_image)     # assinging the uploaded image to img object
    img = img.convert("RGB")            # converting the loaded image to RGB color mode
    img = img.resize((32,32))           # resizing the image to 32X32 (as a tuple) pixel size 
    
    # by normalizing the image we convert the image to tensor (data) that will be fed to model for prediction
    data = np.asarray(img)
    
    # print("before: ",data[0][0])        # printing the color of first pixel before normalization
    
    data = data/255
    
    # print("after: ",data[0][0])         # printing the color of first pixel after normalization

    # actual prediction of image happening here
    # probs = model.predict(data)

    # since above probs variable is assigned to one image but model is expecting thousands of images
    # to overcome that we change the data into list of numpy array and process all images till index 1
    probs = model.predict(np.array([data][:1]))
    
    # this will print probability of 10 items of each class
    # Class is the classification of animal, ship, bird, plane etc. that was assinged in jupyter notebook
    # print(probs)

    # Printing the top probability
    # print(probs.max())
    top_prob = probs.max()

    # Printing the class
    # print(np.argmax(probs))
    top_pred = class_names[np.argmax(probs)]

    return top_prob, top_pred

# importing Path from pathlib to get folder path and file names. This is for *nix os only
# from pathlib import Path

# importing os will allow us to access paths and files
# import os

# Using HTML tags
# index = "<h1>Hello from Raja!!!</h1>"

# using Markdown elements.
# index = '# Hello from Raja !!! This is my first ML python program.'

# image path defined. Single slash (\) is escape character, so use double slashes (\\) when defining a folder path
img_path = 'D:\\MicroSD\\MyFolder\\Education\\Python Projects\\ML Projects\\ml_gui_app-main\\starterFiles\\'

# Concatenating the folder path to logo.png image file
logo = img_path + "logo.png"   
# logo = 'logo.png'
content = ""
chk_img = img_path + "placeholder_image.png"
# chk_img = 'placeholder_image.png'

prob = 0        # top_prob value will be assigned later
pred = ""       # class name of top_pred will be assinged here
# Actual index
index = """
<|text-center|
<|{logo}|image|width=25vw|>

Select an image file: 
<|{content}|file_selector|extensions = .png|> 

<|This is an image of {pred}|>

<|{chk_img}|image|>

<|{prob}|indicator|value={prob}|min=0|max=100|width=25vw|>
>
"""
app = Gui(page=index)

def on_change(state, var_name, var_val):
    if var_name == 'content':
        # assigning the loaded image file path and name to image display variable 
        # on state change 
        state.chk_img = var_val

        # unpacking the values of top prob and top pred into assgined variables
        # by calling predict_img function
        top_prob, top_pred = predict_img(model, var_val)

        # assigning the top_prob and top_pred values to 
        # display variables prob and pred when state changes
        state.prob = round(top_prob * 100)
        state.pred = top_pred
    # print('var_name: ', var_name, 'var_val: ', var_val)
    
if __name__ == "__main__":
    app.run(use_reloader=True)


