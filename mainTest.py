import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

#here the load the model

model=load_model('BrainTumor10EpochsCategorical.h5')

#read the images 

image=cv2.imread('E:\\Brain tumor identification system\\pred\\pred0.jpg')

#convert this image into array

img=Image.fromarray(image)


#then resize them
img=img.resize((64,64))

#it convert image into numpy array
img=np.array(img)

#this function is to add an extra diamension to the numpy array
input_img=np.expand_dims(img, axis=0)

#predict the classes
result=model.predict_step(input_img)
print(result)
