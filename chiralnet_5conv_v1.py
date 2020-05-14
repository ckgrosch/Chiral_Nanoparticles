
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import h5py
import os
import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

#inport data
X_train_shuff = h5py.File('/global/scratch/cgroschner/chiral_data/Train_shuff_Chiral_D_Large_TIFF_Cropped_four_rows_sel_NPs_rotated.h5','r')['images'][:,:,:,:]
Y_train_shuff = h5py.File('/global/scratch/cgroschner/chiral_data/Train_shuff_Chiral_D_Large_TIFF_Cropped_four_rows_sel_NPs_rotated.h5','r')['labels'][:,:]
X_test_shuff = h5py.File('/global/scratch/cgroschner/chiral_data/Test_shuff_Chiral_D_Large_TIFF_Cropped_four_rows_sel_NPs_rotated.h5','r')['images'][:,:,:,:]
Y_test_shuff = h5py.File('/global/scratch/cgroschner/chiral_data/Test_shuff_Chiral_D_Large_TIFF_Cropped_four_rows_sel_NPs_rotated.h5','r')['labels'][:,:]

#define filenames to save model wights to
save_weights = '/global/scratch/cgroschner/chiral_data/chiralnet_5conv_v1.h5'
save_weights_final = '/global/scratch/cgroschner/chiral_data/chiralnet_5conv_v1.h5'
history_file = '/global/scratch/cgroschner/chiral_data/chiralnet_5conv_v1_history.pkl'
#constants to be defined
batch_size = 50
seed = 42

train_datagen = ImageDataGenerator(
        rotation_range = 90,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        vertical_flip = False)

test_datagen = ImageDataGenerator(rotation_range = 90)

train_generator = train_datagen.flow(X_train_shuff, y=Y_train_shuff, batch_size=batch_size,seed=seed)
val_generator = test_datagen.flow(X_test_shuff,y=Y_test_shuff,batch_size=batch_size,seed=seed)


if os.path.isfile(save_weights) == True:
    raise(RuntimeError('FILE ALREADY EXISTS RENAME WEIGHT FILE'))
if os.path.isfile(save_weights_final) == True:
    raise(RuntimeError('FILE ALREADY EXISTS RENAME FINAL WEIGHTS FILE'))

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

modelD = Sequential()
modelD.add(Conv2D(64, (3, 3), input_shape=(128, 128,1)))
modelD.add(Activation('relu'))
modelD.add(MaxPooling2D(pool_size=(2, 2)))

modelD.add(Conv2D(64, (3, 3)))
modelD.add(Activation('relu'))
modelD.add(MaxPooling2D(pool_size=(2, 2)))

modelD.add(Conv2D(64, (3, 3)))
modelD.add(Activation('relu'))
modelD.add(MaxPooling2D(pool_size=(2, 2)))

modelD.add(Conv2D(128, (3, 3)))
modelD.add(Activation('relu'))
modelD.add(MaxPooling2D(pool_size=(2, 2)))

modelD.add(Conv2D(128, (3, 3)))
modelD.add(Activation('relu'))
modelD.add(MaxPooling2D(pool_size=(2, 2)))

modelD.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
modelD.add(Dense(128))
modelD.add(Activation('relu'))
modelD.add(Dropout(0.5))
modelD.add(Dense(2))
modelD.add(Activation('softmax'))

modelD.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(modelD.summary())
history = modelD.fit_generator(
        train_generator,
        steps_per_epoch=1000 // batch_size,
        epochs=50,
        validation_data=val_generator,
        validation_steps=500 // batch_size)
modelD.save_weights(save_weights_final)
validation_score = modelD.evaluate_generator(val_generator,steps=5)
print(validation_score)
with open(history_file, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
