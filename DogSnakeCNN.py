"""
This is a simple CNN network using Pythin and Keras.
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

def ReadData():

	# Defining generators
	train_datagen = ImageDataGenerator(
    	rotation_range = ROTATION_RANGE,                  
    	width_shift_range = WIDTH_SHIFT_RANGE,                  
    	height_shift_range = HEIGHT_SHIFT_RANGE,                  
    	rescale = RESCALE,
    	shear_range = SHEAR_RANGE,                  
    	zoom_range = ZOOM_RANGE,                     
    	horizontal_flip = HORIZONTAL_FLIP
    	)

	validation_datagen = ImageDataGenerator(rescale = 1./255)

	train_generator = train_datagen.flow_from_directory(
    	'data/train',
    	target_size=(IMG_SIZE,IMG_SIZE),
    	class_mode='binary',
    	batch_size = BATCH_SIZE)

	validation_generator = validation_datagen.flow_from_directory(
    	'data/validation',
    	target_size=(IMG_SIZE,IMG_SIZE),
    	class_mode='binary',
    	batch_size = BATCH_SIZE)

	return train_generator, validation_generator


def BuildModel(verbose=True):
	# CNN structure
	cnn = models.Sequential()

	# The same padding means zero
	cnn.add(layers.Conv2D(filters=32, 
               kernel_size=(3,3), 
               strides=(1,1),
               padding='same',
               input_shape=(IMG_SIZE, IMG_SIZE, NB_CHANNELS),
               data_format='channels_last', activation='relu'))
	cnn.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))


	cnn.add(layers.Conv2D(filters=64,
               kernel_size=(3,3),
               strides=(1,1),
               padding='valid', activation='relu'))
	cnn.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))

	cnn.add(layers.Conv2D(filters=64,
               kernel_size=(3,3),
               strides=(1,1),
               padding='valid', activation='relu'))
	cnn.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))
	cnn.add(layers.Dropout(0.2))


	cnn.add(layers.Flatten())        
	cnn.add(layers.Dense(64 , activation='relu'))
	cnn.add(layers.Dense(32 , activation='relu'))
	cnn.add(layers.Dense(1, activation='sigmoid'))

	opt = keras.optimizers.RMSprop(lr=0.001)

	cnn.compile(loss='binary_crossentropy', 
            optimizer=opt, metrics=['accuracy'])

	if verbose:
		cnn.summary()

	return cnn


def TrainModel(model=None, tg=None, vg=None, t=False, s=True):
	if t:
		start = time.time()

	history = model.fit(
    	tg,
    	verbose=VERBOSE,
    	steps_per_epoch=NB_TRAIN_IMG//BATCH_SIZE,
    	epochs=EPOCHS,
    	validation_data=vg,
    	validation_steps=NB_VALID_IMG//BATCH_SIZE)

	evaluate = model.evaluate(vg)
	
	if t:
		end = time.time()
		print('Processing time:',(end - start)/60)
	if s:	
		model.save_weights('DogSnakeModel.h5', overwrite=True)
		model.save('DogSnakeModel_all.h5', overwrite=True)

	accuracy = 	evaluate[1]
	loss = evaluate[0]
	
	return history, accuracy, loss


# Plotting training and validation loss
def TrainValidLossPlot(history):
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(1, len(loss) + 1)
	plt.plot(epochs, loss, color='red', label='Training loss')
	plt.plot(epochs, val_loss, color='green', label='Validation loss')
	plt.title('Training and validation loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.show()

# Plotting training and validation accuracy	
def TrainValidAccuracyPlot(history):
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']
	epochs = range(1, len(acc) + 1)
	plt.plot(epochs, acc, color='red', label='Training acc')
	plt.plot(epochs, val_acc, color='green', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.show()

train_generator, validation_generator = ReadData()
cnn = BuildModel()
history, accuracy, loss = TrainModel(cnn, train_generator, validation_generator)
print ("Evaluation accuracy:", accuracy, "Evaluation loss:", loss)

TrainValidLossPlot(history)
TrainValidAccuracyPlot(history)
