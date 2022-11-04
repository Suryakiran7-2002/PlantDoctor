from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import sqlite3 as sq

from PIL import Image
from tensorflow import keras

model = tf.keras.models.load_model("2")
#model = keras.models.load_model('model.h5')
#ask pranav
def predict():
    class_names = ['Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato_Target_Spot', 'TomatoTomato_YellowLeafCurl_Virus','Tomato_Tomato_mosaic_virus','Tomato_healthy']

    
    img = np.array(Image.open('static/uploads/upload.jpg'))
    imgb = np.expand_dims(img,0)
    predictions = model.predict(imgb)
    index = np.argmax(predictions[0])
    conf = round(np.max(predictions[0]) * 100)
    dis = class_names[index]
    
    conn = sq.connect('database.db')
    c = conn.cursor()
    c.execute('select * from info where key = (?)',(dis,))
    data = c.fetchall()
    conn.close()
    return [data,conf]
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['POST','GET'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload.JPG'))
            
            flash('Image successfully uploaded and displayed below')
            
            data,conf = predict()
            #delete file
            
            return render_template('home.html',op = 1,data = data,conf = conf)
        else:
            flash('Allowed image types are - png, jpg, jpeg, gif')
            return redirect(request.url)
    else:
        return render_template('home.html',op = 0)


if __name__ == '__main__':
    
    app.run(debug=True)