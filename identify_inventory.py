# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:45:15 2019

@author: JUAN
"""

import cv2
import numpy as np
from PIL import Image
from keras import models
#from keras.models import model_from_json
import time
from glob import glob
import os
from scipy import stats
from keras.preprocessing import image as image_utils
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


#Load the saved model
"""
"Loading previous model/classifier"
# load json and create model
json_file = open('model.json1','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model1.h5")
print("Loaded model from disk")
 
model = loaded_model
"""

model = models.load_model('model_saved.h5')
video = cv2.VideoCapture(0)

m=np.array([])
class_names = sorted([os.path.basename(os.path.dirname(f)) for f in glob('./resized/training_set/*/')])
label_dict = class_names
while True:
        _, original = video.read()

        #Convert the captured frame into RGB
       # im = Image.fromarray(frame,'RGB')
        frame = cv2.resize(original, (64, 64))
        im=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        #im /= 255
        #Resizing into 128x128 because we trained the model with this image size.
      #  im = im.resize((64,64))
      #  im = im.transpose((1,0, 2))
        #image = image.reshape((1,) + image.shape)
        img_array = np.array(im)

        #Our keras model used a 4D tensor, (images x height x width x channel)
        #So changing dimension 128x128x3 into 1x128x128x3 
        img_array = np.expand_dims(img_array, axis=0)

        #Calling the predict method on model to predict on the image
        img_array2 = img_array/255
        prediction = model.predict(img_array)
        prediction2 = model.predict(img_array2)
        
        label=label_dict[np.argmax(prediction)]
        label2=label_dict[np.argmax(prediction2)]
                
        #else:
                #print('Detecting')
        #original=frame
       # cv2.putText(original, "Label: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(original, "Label: {}".format(label2), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(original, "Certainty: {}".format(np.max(prediction2)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Classification", original)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release() 
cv2.destroyAllWindows()

a = stats.mode(m)
print(label_dict[int(a[0])])
import keras

from keras.preprocessing import image

#test_image = image.load_img('resized/single_prediction/what_class.jpg', target_size = (128,128))

test_image = image.load_img('resized/single_prediction/IMG_20190306_131215.jpg', target_size = (64,64))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)

result = model.predict(test_image)

"""
(c, d)=result.shape
for i in range(d):
    if int(prediction[0][i])==1:
        print('the item is '+label_dict[i])

one_hot_labels = keras.utils.to_categorical(labels, num_classes=32)

labels1 = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical one-hot encoding
one_hot_labels1 = keras.utils.to_categorical(labels1, num_classes=10)
"""
"""
(a, b)=prediction.shape
for i in range:
    if int(prediction[0][i])==1:
        print(label_dict[i])
        time.sleep(1)
    else:
        print('Detecting')
"""        
        