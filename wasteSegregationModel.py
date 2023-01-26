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



