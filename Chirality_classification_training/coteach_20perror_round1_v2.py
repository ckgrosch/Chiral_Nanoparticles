import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Input, concatenate
from tensorflow.keras.models import Model, load_model
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.utils import to_categorical
from sklearn import metrics
import h5py
from sklearn import utils
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

save_weightsA = '/global/scratch/cgroschner/chiral_nanoparticles/coteach_modelA_20perror_round1_weights_v2.h5'
save_historyA = '/global/scratch/cgroschner/chiral_nanoparticles/coteach_modelA_20perror_round1_history_v2.h5'
save_weightsB = '/global/scratch/cgroschner/chiral_nanoparticles/coteach_modelB_20perror_round1_weights_v2.h5'
save_historyB = '/global/scratch/cgroschner/chiral_nanoparticles/coteach_modelB_20perror_round1_history_v2.h5'

right_images = np.load('/global/scratch/cgroschner/chiral_nanoparticles/20200514_right__Chiral_D_Large_TIFF_Cropped_four_rows_sel_NPs_rotated.npy')


new_left_images = []
new_left_labels = []
new_right_images = []
new_right_labels = []


for img in right_images:
    new_left_images.append(np.fliplr(img))
    new_left_labels.append([0,1])
    new_right_images.append(img)
    new_right_labels.append([1,0])
new_left_images = np.array(new_left_images)
new_right_images = np.array(new_right_images)

split = int(191*0.2)


for idx in np.arange(0,split):
    new_right_images[idx] = np.fliplr(new_right_images[idx])

for idx in np.arange(0,split):
    new_left_images[idx] = np.fliplr(new_left_images[idx])

right_img_shuff, right_label_shuff = utils.shuffle(new_right_images[:191], new_right_labels[:191],random_state=0)
left_img_shuff, left_label_shuff = utils.shuffle(new_left_images[:191], new_left_labels[:191],random_state=0)

X_train = np.concatenate((right_img_shuff[:191],left_img_shuff[:191]),axis =0)
Y_train = np.concatenate((right_label_shuff[:191],left_label_shuff[:191]),axis = 0)

split = int(95*0.2)

flipped_right_indices = np.random.choice(np.arange(191,286),split,replace=False)
flipped_left_indices = np.random.choice(np.arange(191,286),split,replace=False)

for idx in np.arange(191,191+split):
    new_right_images[idx] = np.fliplr(new_right_images[idx])

for idx in np.arange(191,191+split):
    new_left_images[idx] = np.fliplr(new_left_images[idx])

right_img_shuff, right_label_shuff = utils.shuffle(new_right_images[191:286], new_right_labels[191:286],random_state=0)
left_img_shuff, left_label_shuff = utils.shuffle(new_left_images[191:286], new_left_labels[191:286],random_state=0)


X_test = np.concatenate((right_img_shuff,left_img_shuff),axis = 0)
Y_test = np.concatenate((right_label_shuff,left_label_shuff),axis = 0)

X_train_norm = X_train/X_train.max()
X_test_norm = X_test/X_test.max()
X_train_norm = np.expand_dims(X_train_norm,axis=3)
X_test_norm = np.expand_dims(X_test_norm,axis=3)

X_train_shuff, Y_train_shuff = utils.shuffle(X_train_norm, Y_train,random_state=0)
X_test_shuff, Y_test_shuff = utils.shuffle(X_test_norm, Y_test,random_state=0)

batch_size = 25
seed = 42
train_datagen = ImageDataGenerator(
        rotation_range = 10,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=False,
        vertical_flip = False)

test_datagen = ImageDataGenerator(rescale=1.)

train_generator = train_datagen.flow(X_train_shuff, y=Y_train_shuff, batch_size=batch_size,seed=seed)
val_generator = test_datagen.flow(X_test_shuff,y=Y_test_shuff,batch_size=batch_size,seed=seed)


def down(filters, input_):
    down_ = Conv2D(filters, (3, 3))(input_)
    down_res = Activation('relu')(down_)
    down_pool = MaxPooling2D((2, 2))(down_)
    return down_pool

def final_stack(input1_,factor):
    flat = Flatten()(input1_)
    dense1 = Dense(int(64*factor))(flat)
    act1 = Activation('relu')(dense1)
    drop = Dropout(0.5)(act1)
    dense2 = Dense(2)(drop)
    act2 = Activation('softmax')(dense2)
    return act2


def first_cnn(input_,factor):
    down1 = down(int(32*factor),input_)
    down2 = down(int(32*factor),down1)
    down3 = down(int(64*factor),down2)
    down4 = down(int(64*factor),down3)
    down5 = down(int(64*factor),down4)
    # final = first_final_stack(down5)
    return down5

def complete_model(input_shape,factor):
    input1 = Input(shape=input_shape)
    final1 = first_cnn(input1,factor)
    final = final_stack(final1,factor)
    model = Model(inputs=input1, outputs=final)
    return model

modelA = complete_model((128, 128,1),2)
modelB = complete_model((128,128,1),3)

modelA.compile(loss='categorical_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])
modelB.compile(loss='categorical_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])

historyA = modelA.fit_generator(
        train_generator,
        steps_per_epoch=2500,
        epochs=2,
        validation_data=val_generator,
        validation_steps=100,
        verbose = 0)
historyB = modelB.fit_generator(
        train_generator,
        steps_per_epoch=2500,
        epochs=2,
        validation_data=val_generator,
        validation_steps=100,
        verbose = 0)

modelA.save_weights(save_weightsA)
modelB.save_weights(save_weightsB)
h = h5py.File(save_historyA,'w')
h_keys = historyA.history.keys()
print(h_keys)
for k in h_keys:
    h.create_dataset(k,data=historyA.history[k])
h.close()
h = h5py.File(save_historyB,'w')
h_keys = historyB.history.keys()
print(h_keys)
for k in h_keys:
    h.create_dataset(k,data=historyB.history[k])
h.close()

X_val = np.concatenate((new_right_images[286:],new_left_images[286:]),axis = 0)
Y_val = np.concatenate((new_right_labels[286:],new_left_labels[286:]),axis = 0)
X_val = X_val/X_val.max()
X_val = np.expand_dims(X_val,axis=3)
X_val, Y_val = utils.shuffle(X_val, Y_val,random_state=0)
print(modelA.evaluate(X_val,Y_val),modelB.evaluate(X_val,Y_val))
