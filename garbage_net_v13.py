import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Input, Average
from tensorflow.keras.models import Sequential, load_model, Model
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.utils import to_categorical
from sklearn import metrics
import h5py
from sklearn import utils
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn import utils


save_weights = '/global/scratch/cgroschner/chiral_nanoparticles/garbage_net_weights_v13.h5'
save_history = '/global/scratch/cgroschner/chiral_nanoparticles/garbage_net_history_v13.h5'

good = h5py.File('/global/scratch/cgroschner/chiral_nanoparticles/orientation_labels_20201008.h5')['good'][:225,:,:]
bad = h5py.File('/global/scratch/cgroschner/chiral_nanoparticles/orientation_labels_20201008.h5')['bad'][:,:,:]

good = np.expand_dims(good,axis=3)
bad = np.expand_dims(bad,axis=3)

trainX0 = h5py.File('/global/scratch/cgroschner/chiral_nanoparticles/20200514_garbage_detector_data.h5')['trainX'][:,:,:,:]
trainY0 = h5py.File('/global/scratch/cgroschner/chiral_nanoparticles/20200514_garbage_detector_data.h5')['trainY'][:,:]
testX0 = h5py.File('/global/scratch/cgroschner/chiral_nanoparticles/20200514_garbage_detector_data.h5')['testX'][:,:,:,:]
testY0 = h5py.File('/global/scratch/cgroschner/chiral_nanoparticles/20200514_garbage_detector_data.h5')['testY'][:,:]
valX0 = h5py.File('/global/scratch/cgroschner/chiral_nanoparticles/20200514_garbage_detector_data.h5')['valX'][:,:,:,:]
valY0 = h5py.File('/global/scratch/cgroschner/chiral_nanoparticles/20200514_garbage_detector_data.h5')['valY'][:,:]

for idx,img in enumerate(good):
    good[idx] = (img-img.min())/(img.max()-img.min())
for idx,img in enumerate(bad):
    bad[idx] = (img-img.min())/(img.max()-img.min())

good_labels = [[1,0] for i in good]
bad_labels = [[0,1] for i in bad]

trainX = np.concatenate((trainX0,good[:180],bad[:180]),axis=0)
trainY = np.concatenate((trainY0,good_labels[:180],bad_labels[:180]),axis=0)
trainX, trainY = utils.shuffle(trainX,trainY,random_state=0)
testX = np.concatenate((testX0,good[180:203],bad[180:203]))
testY = np.concatenate((testY0,good_labels[180:203],bad_labels[180:203]))
testX, testY = utils.shuffle(testX,testY,random_state=0)
valX = np.concatenate((valX0,good[203:],bad[203:]))
valY = np.concatenate((valY0,good_labels[203:],bad_labels[203:]))






batch_size = 25
seed = 42
train_datagen = ImageDataGenerator(
        rotation_range = 10,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=False,
        vertical_flip = False)

test_datagen = ImageDataGenerator(rescale=1.)

train_generator = train_datagen.flow(trainX, y=trainY, batch_size=batch_size,seed=seed)
val_generator = test_datagen.flow(testX,y=testY,batch_size=batch_size,seed=seed)

modelE = keras.models.Sequential()
modelE.add(Conv2D(32, (3, 3), input_shape=(128, 128,1)))
modelE.add(Activation('relu'))
modelE.add(MaxPooling2D(pool_size=(2, 2)))
modelE.add(Dropout(0.5))

modelE.add(Conv2D(32, (3, 3)))
modelE.add(Activation('relu'))
modelE.add(MaxPooling2D(pool_size=(2, 2)))
modelE.add(Dropout(0.5))

modelE.add(Conv2D(64, (3, 3)))
modelE.add(Activation('relu'))
modelE.add(MaxPooling2D(pool_size=(2, 2)))
modelE.add(Dropout(0.5))

modelE.add(Conv2D(64, (3, 3)))
modelE.add(Activation('relu'))
modelE.add(MaxPooling2D(pool_size=(2, 2)))
modelE.add(Dropout(0.5))

modelE.add(Conv2D(64, (3, 3)))
modelE.add(Activation('relu'))
modelE.add(MaxPooling2D(pool_size=(2, 2)))
modelE.add(Dropout(0.5))

modelE.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
modelE.add(Dense(64))
modelE.add(Activation('relu'))
modelE.add(Dropout(0.5))
modelE.add(Dense(2))
modelE.add(Activation('softmax'))

modelE.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-3),
              metrics=['accuracy'])
earlyStopping = EarlyStopping(monitor='val_loss',
                              patience=4,
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
        steps_per_epoch=1000,
        epochs=20,
        callbacks=callbacks_list,
        validation_data=val_generator,
        validation_steps=100,
        verbose = 0)
modelE.save_weights(save_weights)
h = h5py.File(save_history,'w')
h_keys = history.history.keys()
print(h_keys)
for k in h_keys:
    h.create_dataset(k,data=history.history[k])
h.close()

print(modelE.evaluate(valX,valY))