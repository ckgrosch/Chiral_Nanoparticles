import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential, load_model
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.utils import to_categorical
from sklearn import metrics
import h5py
from sklearn import utils
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

save_weights = '/global/scratch/cgroschner/chiral_nanoparticles/chirality_classification_10pwrongleftrightlabels_weights_v4_20200912.h5'
save_history = '/global/scratch/cgroschner/chiral_nanoparticles/chirality_classification_10pwrongleftrightlabels_history_v4_20200912.h5'

trainX = h5py.File('/global/scratch/cgroschner/chiral_nanoparticles/20200912_10perror_Chiral_D_Large_TIFF_Cropped_four_rows_sel_NPs_rotated.h5','r')['trainX'][:,:,:,:]
trainY = h5py.File('/global/scratch/cgroschner/chiral_nanoparticles/20200912_10perror_Chiral_D_Large_TIFF_Cropped_four_rows_sel_NPs_rotated.h5','r')['trainY'][:]
testX = h5py.File('/global/scratch/cgroschner/chiral_nanoparticles/20200912_10perror_Chiral_D_Large_TIFF_Cropped_four_rows_sel_NPs_rotated.h5','r')['testX'][:,:,:,:]
testY = h5py.File('/global/scratch/cgroschner/chiral_nanoparticles/20200912_10perror_Chiral_D_Large_TIFF_Cropped_four_rows_sel_NPs_rotated.h5','r')['testY'][:]
valX = h5py.File('/global/scratch/cgroschner/chiral_nanoparticles/20200912_10perror_Chiral_D_Large_TIFF_Cropped_four_rows_sel_NPs_rotated.h5','r')['valX'][:,:,:]
valY = h5py.File('/global/scratch/cgroschner/chiral_nanoparticles/20200912_10perror_Chiral_D_Large_TIFF_Cropped_four_rows_sel_NPs_rotated.h5','r')['valY'][:]
valX = np.expand_dims(valX,axis=3)


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

modelE.add(Conv2D(32, (3, 3)))
modelE.add(Activation('relu'))
modelE.add(MaxPooling2D(pool_size=(2, 2)))

modelE.add(Conv2D(64, (3, 3)))
modelE.add(Activation('relu'))
modelE.add(MaxPooling2D(pool_size=(2, 2)))

modelE.add(Conv2D(64, (3, 3)))
modelE.add(Activation('relu'))
modelE.add(MaxPooling2D(pool_size=(2, 2)))

modelE.add(Conv2D(64, (3, 3)))
modelE.add(Activation('relu'))
modelE.add(MaxPooling2D(pool_size=(2, 2)))

modelE.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
modelE.add(Dense(64))
modelE.add(Activation('relu'))
modelE.add(Dropout(0.5))
modelE.add(Dense(2))
modelE.add(Activation('softmax'))

modelE.compile(loss='categorical_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])


# modelCheckpoint = ModelCheckpoint(save_weights,
#                                   monitor = 'val_accuracy',
#                                   save_best_only = True,
#                                   mode = 'max',
#                                   verbose = 2,
#                                   save_weights_only = True)
# callbacks_list = [modelCheckpoint]

history = modelE.fit_generator(
        train_generator,
        steps_per_epoch=2500,
        epochs=30,
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
