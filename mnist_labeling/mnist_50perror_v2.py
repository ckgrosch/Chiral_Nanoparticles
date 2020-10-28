import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential, load_model
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.utils import to_categorical
from sklearn import metrics
from sklearn.model_selection import train_test_split
import h5py
from sklearn import utils
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


wrong_frac = 0.5
version = 2
name_frac = str(wrong_frac).split('.')[-1]
if len(name_frac) == 1:
    name_frac = name_frac + '0'
save_weights = '/global/scratch/cgroschner/chiral_nanoparticles/mnist_'+ name_frac +'perror_weights_v'+ str(version) + '.h5'
save_history = '/global/scratch/cgroschner/chiral_nanoparticles/mnist_'+ name_frac +'perror_history_v'+ str(version) + '.h5'

data = keras.datasets.mnist.load_data()

seven_imgs_train = []
seven_labels_train = []
for idx,label in enumerate(data[0][1]):
    if label == 7:
        seven_imgs_train.append(data[0][0][idx])
        seven_labels_train.append(label)
seven_imgs_train = np.array(seven_imgs_train)
seven_labels_train = np.array(seven_labels_train)


new_left_images = []
new_left_labels = []
new_right_images = []
new_right_labels = []

for img in seven_imgs_train:
    new_left_images.append(np.fliplr(img))
    new_left_labels.append(0)
    new_right_images.append(img)
    new_right_labels.append(1)
new_left_images = np.array(new_left_images)
new_right_images = np.array(new_right_images)

split = int(seven_imgs_train.shape[0]*wrong_frac)

for idx in np.arange(0,split):
    new_right_images[idx] = np.fliplr(new_right_images[idx])

for idx in np.arange(0,split):
    new_left_images[idx] = np.fliplr(new_left_images[idx])


X = utils.shuffle(np.concatenate((new_right_images,new_left_images),axis =0),random_state=0)
Y = utils.shuffle(np.concatenate((new_right_labels,new_left_labels),axis = 0),random_state=0)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,shuffle=False)

X_train_norm = X_train/X_train.max()
X_test_norm = X_test/X_test.max()
X_train_norm = np.expand_dims(X_train_norm,axis=3)
X_test_norm = np.expand_dims(X_test_norm,axis=3)


batch_size = 25
seed = 42
train_datagen = ImageDataGenerator(
        rotation_range = 10,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=False,
        vertical_flip = False)

test_datagen = ImageDataGenerator(rescale=1.)

train_generator = train_datagen.flow(X_train_norm, y=Y_train, batch_size=batch_size,seed=seed)
val_generator = test_datagen.flow(X_test_norm,y=Y_test,batch_size=batch_size,seed=seed)


modelE = keras.models.Sequential()
modelE.add(Conv2D(64, (3, 3), input_shape=(28, 28,1)))
modelE.add(Activation('relu'))
modelE.add(MaxPooling2D(pool_size=(2, 2)))

modelE.add(Conv2D(64, (3, 3)))
modelE.add(Activation('relu'))
modelE.add(MaxPooling2D(pool_size=(2, 2)))

modelE.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
modelE.add(Dense(64))
modelE.add(Activation('relu'))
modelE.add(Dropout(0.5))
modelE.add(Dense(1))
modelE.add(Activation('sigmoid'))

modelE.compile(loss='binary_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])

earlyStopping = EarlyStopping(monitor='val_loss',
                              patience=2,
                              verbose=2,
                              min_delta = 0.001,
                              mode='min',)

modelCheckpoint = ModelCheckpoint(save_weights,
                                  monitor = 'val_loss',
                                  save_best_only = True,
                                  mode = 'min',
                                  verbose = 2,
                                  save_weights_only = True)
callbacks_list = [modelCheckpoint,earlyStopping]
history = modelE.fit_generator(
        train_generator,
        steps_per_epoch=2500,
        epochs=30,
        validation_data=val_generator,
        validation_steps=100,
        verbose = 0,
        callbacks=callbacks_list)
modelE.save_weights(save_weights)
h = h5py.File(save_history,'w')
h_keys = history.history.keys()
for k in h_keys:
    h.create_dataset(k,data=history.history[k])
h.close()

seven_imgs_test = []
for idx,label in enumerate(data[1][1]):
    if label == 7:
        im_max =data[1][0][idx].max()
        seven_imgs_test.append(data[1][0][idx]/im_max)
seven_imgs_test = np.array(seven_imgs_test)

X_left = np.array([np.fliplr(img) for img in seven_imgs_test])
X = np.expand_dims(np.concatenate((seven_imgs_test,X_left),axis=0),axis=3)
Y = np.concatenate((np.ones((seven_imgs_test.shape[0],)),np.zeros((X_left.shape[0],))),axis=0)

print(modelE.evaluate(X,Y))
