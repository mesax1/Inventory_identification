# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:50:38 2019

@author: JUAN
"""

import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from skimage.io import imread
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import itertools

from tqdm import tqdm

# This function saves and plots a confusion matrix. plt.show() must be called from outside
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          save_fig_name ='confusion_matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_fig_name+'.png',bbox_inches='tight')


# Returns images and labels. The data structure of the images must be: path/class1, path/class2
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
        for i in tqdm(list_image_names):
        #for i in range(len(list_image_names)):
            if i == '.DS_Store':
                continue

            labels.append(idx)

            list_images.append(resize(imread(os.path.join(path,j,i)),output_shape=size))
    # For multiclass classification, you have to one-hot-encode np.array(labels) and return that
    labels= to_categorical(labels, num_classes=32)
    
    return np.array(list_images), np.array(labels), label_dict


def return_CNN(input_shape=(64,64,3)):

    # create, compile and return a CNN from this function.
    classifier = Sequential()

    image_size=(128, 128)
    # Step 1 - Convolution
    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))  #Increase input_shape for increased accuraccy

    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Adding a second convolutional layer (for increased accuracy)
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Adding a third convolutional layer (for increased accuracy, with some changed parameters)
    classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Step 3 - Flattening
    classifier.add(Flatten())
    #Add Dropout of 20% to reduce overfitting
    #classifier.add(Dropout(0.2))
    # Step 4 - Full connection
    classifier.add(Dense(units = 128, activation = 'relu'))
    #Add Dropout of 20% to reduce overfitting
    #classifier.add(Dropout(0.5))
    classifier.add(Dense(units = 128, activation = 'relu'))
    
    #For the OUTPUT layer, the number of units equals the number of classes
    classifier.add(Dense(units = 32, activation = 'softmax'))
    
    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return classifier



print ("GET FULL DATASET")
# Get FULL Dataset
images, labels, label_dict = get_data(os.path.join('dataset','training_set'))

print ("SHUFFLE DATASET")
# Shuffle dataset
images, labels = shuffle(images,labels)

print ("SPLIT DATASET - 01")
# Split Dataset into Training and Test Set
images_train, images_test, labels_train, labels_test = train_test_split(images,labels,test_size=0.20)

print ("SPLIT DATASET - 02")
# Split Training set into Training and Validation Set
images_train, images_val, labels_train, labels_val = train_test_split(images_train,labels_train,test_size=0.20)

# Create ImageDataGenerator object for training and validation set

# No need for rescale here since skimage imread returns 0-1 values
datagen_train = ImageDataGenerator(
                               shear_range = 0.2,
                               zoom_range = 0.2,
                               horizontal_flip = True)

# No need for rescale here since skimage imread returns 0-1 values
datagen_val   = ImageDataGenerator(
                               shear_range = 0.2,
                               zoom_range = 0.2,
                               horizontal_flip = True)

# Fit training set and validation set to datagen

datagen_train.fit(images_train)
datagen_val.fit(images_val)

# Get CNN
model = return_CNN()

#Convert labels to categorical format
   # labels_train_one_hot = to_categorical(labels_train, num_classes=32)
   # labels_val_one_hot= to_categorical(labels_val, num_classes=32)
   # labels_test_one_hot= to_categorical(labels_test, num_classes=32)
# Get Keras Callbacks
es = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=22, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1)
mcp = ModelCheckpoint(filepath='model_saved3.h5',monitor='val_loss',verbose=1,save_best_only=True)

print ("START MODEL FITTING")
# fits the model on batches with real-time data augmentation:
"""
model.fit_generator(datagen_train.flow(images_train, labels_train_one_hot, batch_size=32),
                    steps_per_epoch=len(images_train)*2 / 32, epochs=50, callbacks=[es, rlr,mcp],
                    validation_data=datagen_val.flow(images_val,labels_val_one_hot,batch_size=32),
                    validation_steps=len(images_val)*2/32)

model.fit_generator(datagen_train.flow(images_train, labels_train, batch_size=32),
                    steps_per_epoch=len(images_train)*2/32, epochs=50, callbacks=[es, rlr, mcp],
                    validation_data=datagen_val.flow(images_val, labels_val, batch_size=32),
                    validation_steps=len(images_val)*2/32)

"""
image_size=(64,64)
train_datagen = ImageDataGenerator(#rescale = 1./255,
                               shear_range = 0.2,
                               zoom_range = 0.2,
                               horizontal_flip = True)

test_datagen = train_datagen

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                             target_size = image_size,            #Increase target size for increased accuraccy
                                             batch_size = 32,
                                             class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/training_set',
                                        target_size = image_size,
                                        batch_size = 32,
                                        class_mode = 'categorical')

model.fit_generator(training_set,
                     steps_per_epoch = 40,      #Number of images in training_set
                     epochs = 3,
                     validation_data = test_set,
                     validation_steps = 40)      #Number of images in test_set

model.fit_generator(datagen_train.flow_from_directory('dataset'+'/'+'training_set',images_train, batch_size=32),
                    steps_per_epoch=1, epochs=32, callbacks=[es, rlr, mcp],
                    validation_data=datagen_val.flow(images_val, batch_size=32),
                    validation_steps=1)

np.save('CLASS_INDICES', training_set.class_indices)

# Get accuracy of model on test_data.

# If you use a sequential model, you can just use predictions_test = model.predict_classes(images_test)
predictions_test = (model.predict(images_test)>0.5)

model.evaluate(images_test, labels_test)

model.evaluate(images_train, labels_train)

print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("ACCURACY ON TEST SET SPLIT")
print (accuracy_score(labels_test, predictions_test)*100.0)
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

# Compute confusion matrix
print ("CONFUSION MATRIX FOR TEST SET SPLIT")
#cnf_matrix = confusion_matrix(y_true=labels_test,y_pred=predictions_test)
cnf_matrix = confusion_matrix(y_true=labels_test.argmax(axis=1), y_pred=predictions_test.argmax(axis=1))

print ("PLOT CONFUSION MATRIX FOR TEST SET SPLIT")
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=label_dict.values(),save_fig_name='confusion_matrix_test_set_split')
plt.show()


#Remove variables we dont need anymore to make room in the RAM
del images_test,images_val,images_train,labels_test,labels_val,labels_train,predictions_test,cnf_matrix

print ("LOAD NEW TEST SET")
images_new_test, labels_new_test,label_dict = get_data(os.path.join('dataset','test_set'))

print ("PREDICT ON NEW TEST SET")
predictions_new_test = (model.predict(images_new_test) > 0.5)

print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("ACCURACY ON NEW TEST SET")
print (accuracy_score(labels_new_test,predictions_new_test)*100.0)
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

# Compute confusion matrix
print ("CONFUSION MATRIX FOR TEST SET")
cnf_matrix = confusion_matrix(y_true=labels_new_test,y_pred=predictions_new_test)



print ("PLOT CONFUSION MATRIX FOR TEST SET")
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=label_dict.values(),save_fig_name='confusion_matrix_new_test_set')
plt.show()

#if __name__ == "__main__":
#main()