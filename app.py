import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


# creates an instance of the flask class & initializes a flask application

app = Flask(__name__)

#model load (code for connect web browser/app )

model =load_model('BrainTumor10EpochsCategorical.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

# get the value from the maintest.py

# the values comes from the mainTest.py


def get_className(classNo):
	if classNo==0:
		return "Normal brain scan - No tumor Detected "
	elif classNo==1:
		return "Abnormal findings - Possible brain tumor identified"


def getResult(img):
    image=cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image=np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model.predict(input_img, verbose=0)
    predicted_class = np.argmax(result[0])
    return predicted_class


# execute with the index.html 
# browse the images  and upload the images 

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

#create the predict file
#if the mthod is post then it will upload the img file 

#newjs.js

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        # predict folder= dirname = basepath
        # we can upload the image in the datasets  ...all the image that we upload will add to the upload folder
        # here we use the secure file name beacause the request_method= POST

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        result=get_className(value) 
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)