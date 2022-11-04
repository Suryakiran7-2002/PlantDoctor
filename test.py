import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image


model=tf.keras.models.load_model('2')

class_names = ['Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato__Tomato_mosaic_virus','Tomato_healthy']


img = np.array(Image.open('static/uploads/upload.JPG'))
imgb = np.expand_dims(img,0)
predictions = model.predict(imgb)
index = np.argmax(predictions[0])
predicted_class = class_names[index]
print(predicted_class,np.max(predictions[0]))


