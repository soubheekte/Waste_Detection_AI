#Problem Statement:
# The importance of recycling is well known, either for environmental or economic reasons, 
# it is impossible to escape it and the industry demands efficiency. Manual labour and traditional 
# industrial sorting techniques are not capable of keeping up with the objectives demanded by the international community. 
# Solutions based on computer vision techniques have the potential to automate part of the waste handling tasks.




# import the libraries as shown below
# import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt



# re-size all the images to this
IMAGE_SIZE = [224, 224]
train_path = '/content/gdrive/MyDrive/Clasification_Data/dataset-resized/train'
valid_path = '/content/gdrive/MyDrive/Clasification_Data/dataset-resized/test'



# Import the InceptionV3 library as shown below and add preprocessing layer to the front of InceptionV3
# Here we will be using imagenet weights
inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights 
for layer in inception.layers:
    layer.trainable = False
# useful for getting number of output classes

folders = glob('/content/gdrive/MyDrive/Clasification_Data/dataset-resized/train/*')

# Adding our layers - you can add more if you want
x = Flatten()(inception.output)
prediction = Dense(len(folders), activation='softmax')(x)


# create a model object
model = Model(inputs=inception.input, outputs=prediction)

# view the structure of the model
model.summary()


# tell the model what cost and optimization method to use\
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# Use the Image Data Generator to import the images from the dataset
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)




# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory(directory='/content/gdrive/MyDrive/Clasification_Data/dataset-resized/train',
                                                 target_size = (224, 224),
                                                 batch_size = 16,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(directory='/content/gdrive/MyDrive/Clasification_Data/dataset-resized/test',
                                            target_size = (224, 224),
                                            batch_size = 16,
                                            class_mode = 'categorical')



# fit the model
# Run the cell. It will take some time to execute
# Setting the number of epochs
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=10,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)


# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')



    



