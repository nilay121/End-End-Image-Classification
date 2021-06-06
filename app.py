# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 13:13:50 2021

@author: NILAY
"""
from flask import Flask,request, url_for, redirect, render_template
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

model=keras.models.load_model('Image_classification.h5')


@app.route('/')
def main_page():
    return render_template("index.htm")


@app.route('/predict',methods=['POST','GET'])
def predict():
    img_req=request.form['image']
    path='testing/'+img_req
    loaded_img=load_img(path,color_mode='grayscale')
    img_data=img_to_array(loaded_img)
    predictions=model.predict(img_data.reshape(1,150,150,1))
    if predictions==1:
       return render_template('index.htm',Make_Prediction="Building")
    elif predictions==0:
        return render_template('index.htm',Make_Prediction='Forest')



if __name__ == '__main__':
    app.run(debug=True)