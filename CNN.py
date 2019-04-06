# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 19:03:46 2019

@author: JUAN
"""


# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

from keras.models import model_from_json
import os
# Initialising the CNN
classifier = Sequential()

image_size=(128, 128)
# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))  #Increase input_shape for increased accuraccy

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer (for increased accuracy)
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a third convolutional layer (for increased accuracy, with some changed parameters)
#classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())
#Add Dropout of 20% to reduce overfitting
classifier.add(Dropout(0.2))
# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
#Add Dropout of 20% to reduce overfitting
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 128, activation = 'relu'))

#For the OUTPUT layer, the number of units equals the number of classes
classifier.add(Dense(units = 32, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
#Data Augmentation - Create batches of shifted, rotated, etc images, therefore we have more images for training
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('resized/training_set',
                                                 target_size = image_size,            #Increase target size for increased accuraccy
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('resized/test_set',
                                            target_size = image_size,
                                            batch_size = 32,
                                            class_mode = 'categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch = 846,      #Number of images in training_set
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 201)      #Number of images in test_set


#Part 3: Make new predictions
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('resized/single_prediction/what_class.jpg', target_size = image_size)

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)

result = classifier.predict(test_image)
result

"""
def get_data(path='path',size=(64,64)):

    # Returns an array of images and labels from path
    # label_dict is a dictionary with value = 'class_name' and key = class_index
    # The folder structure of path must be: path/class1 path/class2, etc

    # For multiclass classification, you have to one-hot-encode np.array(labels) and return that

    list_images = []
    labels = []
    label_dict = {}

    for idx,j in enumerate(os.listdir(path)):

        if j == '.DS_Store':
            continue

        label_dict[idx]=j
        list_image_names = os.listdir(os.path.join(path,j))

        print ("LOADING "+str(j)+" IMAGES FROM :", os.path.join(path,j))
        for i in list_image_names:

            if i == '.DS_Store':
                continue

            labels.append(idx)

            #list_images.append(cv2.imread(os.path.join(path,j,i)),output_shape=size)


    return np.array(list_images), np.array(labels), label_dict

images, labels, label_dict = get_data(os.path.join('resized','test_set'))

test_generator = test_datagen.flow_from_directory(
    directory='resized/single_prediction/',
    target_size=(64, 64),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
)
test_generator.reset()
pred=classifier.predict_generator(test_generator,verbose=1, steps=1)

"""
"Saving the Model/classifier for later use"
# serialize model to JSON
model_json1 = classifier.to_json()
with open("model.json1", "w") as json_file:
    json_file.write(model_json1)
# serialize weights to HDF5
classifier.save_weights("model1.h5")
print("Saved model to disk")
 
# later...
 "Loading previous model/classifier"
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 

"""
#This is for knowing if result is equal to Cats or Dogs
training_set.class_indices 
if result[0][0] == 1:
    prediction = 'dog'
else: 
    prediction = 'cat'
"""    
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc,roc_auc_score
cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()    

