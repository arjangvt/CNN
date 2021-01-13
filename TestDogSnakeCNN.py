"""
This is a simple CNN network using Python and Keras.
The image set is downloaded from online public domain.

Written by: Arjang Fahim
Date: 5/10/2020

Version:1.0.0
"""

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras import regularizers
from tensorflow.keras import datasets, layers, models
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time

IMG_SIZE = 250
NB_CHANNELS = 3
BATCH_SIZE = 1
NB_TRAIN_IMG = 892
NB_VALID_IMG = 107
EPOCHS = 15
VERBOSE = 1

ROTATION_RANGE = 40
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
RESCALE = 1./255
SHEAR_RANGE = 0.2
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True

MODEL_NAME = "DogSnakeModel_all.h5"
category = {0:"Dog", 1:"Snake"}

# This function loads an image 
def LoadImage(image_path):

	# load an image from file
    image = load_img(image_path, target_size=(IMG_SIZE,IMG_SIZE))
	# convert the image pixels to a numpy array
    image = img_to_array(image)
	# reshape data for the model
    image = image.reshape((1,) +  image.shape)

    return image


def LoadModel(summary = True):

    cnn = keras.models.load_model(MODEL_NAME)
    if summary:
        print(cnn.summary())

    cnn.load_weights("DogSnakeModel.h5")
    return cnn

def Predict(model, img):

    eval = model.evaluate(img, verbose=VERBOSE)
    print(eval)
    preds = model.predict(img, verbose=VERBOSE)
            
    return category[preds[0][0]]


img = LoadImage('data/test/snake1.jpg')
model = LoadModel(False)
category = Predict(model, img)

print(category)
