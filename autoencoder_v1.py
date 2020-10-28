from tensorflow import keras
from tensorflow.keras.layers import Lambda, Input, Dense,Conv2D, MaxPooling2D, Activation, Dropout, Flatten, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras import backend as K

import numpy as np
import argparse
import os
from sklearn import metrics
import h5py
from sklearn import utils

from tensorflow.keras.preprocessing.image import ImageDataGenerator

save_weights = '/global/scratch/cgroschner/chiral_nanoparticles/autoencoder_v1_weights.h5'
save_history = '/global/scratch/cgroschner/chiral_nanoparticles/autoencoder_v1_history.h5'

right_images = np.load('/global/scratch/cgroschner/chiral_nanoparticles/20200514_right__Chiral_D_Large_TIFF_Cropped_four_rows_sel_NPs_rotated.npy')


for idx,r in enumerate(right_images):
    right_images[idx] = (r-r.min())/(r.max()-r.min())


new_left_images = []
new_right_images = []



for img in right_images:
    new_left_images.append(np.fliplr(img))
    new_right_images.append(img)
new_left_images = np.array(new_left_images)
new_right_images = np.array(new_right_images)



right_img_shuff = utils.shuffle(new_right_images[:191],random_state=0)
left_img_shuff = utils.shuffle(new_left_images[:191],random_state=0)

X_train = np.concatenate((right_img_shuff[:191],left_img_shuff[:191]),axis =0)



right_img_shuff = utils.shuffle(new_right_images[191:286],random_state=0)
left_img_shuff = utils.shuffle(new_left_images[191:286],random_state=0)


X_test = np.concatenate((right_img_shuff,left_img_shuff),axis = 0)


X_train = np.expand_dims(X_train,axis=3)
X_test = np.expand_dims(X_test,axis=3)

X_train_shuff = utils.shuffle(X_train,random_state=0)
X_test_shuff = utils.shuffle(X_test,random_state=0)

batch_size = 25
seed = 42
train_datagen = ImageDataGenerator(
        rotation_range = 10,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=False,
        vertical_flip = False)

test_datagen = ImageDataGenerator(rescale=1.)

train_generator = train_datagen.flow(X_train_shuff, y=X_train_shuff, batch_size=batch_size,seed=seed)
val_generator = test_datagen.flow(X_test_shuff,y=X_test_shuff,batch_size=batch_size,seed=seed)

input_img = Input(shape=(128, 128, 1))  # adapt this if using `channels_first` image data format

x1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x2 = MaxPooling2D((2, 2), padding='same')(x1)
x3 = Conv2D(8, (3, 3), activation='relu', padding='same')(x2)
x4 = MaxPooling2D((2, 2), padding='same')(x3)
x5 = Conv2D(8, (3, 3), activation='relu', padding='same')(x4)
x6 = MaxPooling2D((2, 2), padding='same')(x5)
encoded = Conv2D(1, (3, 3), activation='relu', padding='same')(x6)
encoder = Model(input_img,encoded)

encode_input = Input(shape=(16,16,1))
x6 = Conv2D(8, (3, 3), activation='relu', padding='same')(encode_input)
x7 = UpSampling2D((2, 2))(x6)
x8 = Conv2D(8, (3, 3), activation='relu', padding='same')(x7)
x9 = UpSampling2D((2, 2))(x8)
x10 = Conv2D(16, (3, 3), activation='relu',padding='same')(x9)
x11 = UpSampling2D((2, 2))(x10)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x11)
decoder = Model(encode_input,decoded)

outputs = decoder(encoder(input_img))
autoencoder = Model(input_img,outputs)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

history = autoencoder.fit_generator(train_generator,
                                    steps_per_epoch=2500,
                                    epochs=30,
                                    validation_data=val_generator,
                                    validation_steps=100,
                                    verbose = 0)

autoencoder.save_weights(save_weights)
h = h5py.File(save_history,'w')
h_keys = history.history.keys()
print(h_keys)
for k in h_keys:
    h.create_dataset(k,data=history.history[k])
h.close()
X_val = np.concatenate((new_right_images[286:],new_left_images[286:]),axis = 0)
X_val = np.expand_dims(X_val,axis=3)
print(autoencoder.evaluate(X_val,X_val))
