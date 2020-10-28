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

save_weightsA = '/global/scratch/cgroschner/chiral_nanoparticles/coteach_modelA_20perror_round1_weights_v1.h5'
save_historyA = '/global/scratch/cgroschner/chiral_nanoparticles/coteach_modelA_20perror_round1_history_v1.h5'
save_weightsB = '/global/scratch/cgroschner/chiral_nanoparticles/coteach_modelB_20perror_round1_weights_v1.h5'
save_historyB = '/global/scratch/cgroschner/chiral_nanoparticles/coteach_modelB_20perror_round1_history_v1.h5'

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

split = int(0.2*95)

right_img_shuff, right_label_shuff = utils.shuffle(new_right_images[:191], new_right_labels[:191],random_state=0)
left_img_shuff, left_label_shuff = utils.shuffle(new_left_images[:191], new_left_labels[:191],random_state=0)

X_trainA = np.concatenate((right_img_shuff[:95],left_img_shuff[:95]),axis =0)
Y_trainA = np.concatenate((left_label_shuff[:split],right_label_shuff[split:95],right_label_shuff[:split],left_label_shuff[split:95]),axis = 0)
X_trainB = np.concatenate((right_img_shuff[95:-1],left_img_shuff[95:-1]),axis =0)
Y_trainB = np.concatenate((left_label_shuff[:split],right_label_shuff[split:95],right_label_shuff[:split],left_label_shuff[split:95]),axis = 0)


right_img_shuff, right_label_shuff = utils.shuffle(new_right_images[191:286], new_right_labels[191:286],random_state=0)
left_img_shuff, left_label_shuff = utils.shuffle(new_left_images[191:286], new_left_labels[191:286],random_state=0)

split = int(0.2*47)

X_testA = np.concatenate((right_img_shuff[:47],left_img_shuff[:47]),axis = 0)
Y_testA = np.concatenate((left_label_shuff[:split],right_label_shuff[split:47],right_label_shuff[:split],left_label_shuff[split:47]),axis = 0)
correct_testA = np.concatenate((right_label_shuff[47:-1],left_label_shuff[47:-1]),axis = 0)
X_testB = np.concatenate((right_img_shuff[47:-1],left_img_shuff[47:-1]),axis = 0)
Y_testB = np.concatenate((left_label_shuff[:split],right_label_shuff[split:47],right_label_shuff[:split],left_label_shuff[split:47]),axis = 0)
correct_testB = np.concatenate((right_label_shuff[47:-1],left_label_shuff[47:-1]),axis = 0)

X_trainA = X_trainA/X_trainA.max()
X_testA = X_testA/X_testA.max()
X_trainB= X_trainB/X_trainB.max()
X_testB = X_testB/X_testB.max()

X_trainA = np.expand_dims(X_trainA,axis=3)
X_testA = np.expand_dims(X_testA,axis=3)
X_trainB = np.expand_dims(X_trainB,axis=3)
X_testB = np.expand_dims(X_testB,axis=3)

X_trainA, Y_trainA = utils.shuffle(X_trainA, Y_trainA,random_state=0)
X_testA, Y_testA, correct_testA = utils.shuffle(X_testA, Y_testA, correct_testA,random_state=0)
X_trainB,Y_trainB = utils.shuffle(X_trainB, Y_trainB,random_state=0)
X_testB, Y_testB, correct_testB = utils.shuffle(X_testB, Y_testB, correct_testB, random_state=0)

batch_size = 25
seed = 42
train_datagen = ImageDataGenerator(
        rotation_range = 10,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=False,
        vertical_flip = False)

test_datagen = ImageDataGenerator(rescale=1.)

train_generatorA = train_datagen.flow(X_trainA, y=Y_trainA, batch_size=batch_size,seed=seed)
val_generatorA = test_datagen.flow(X_testA,y=Y_testA,batch_size=batch_size,seed=seed)
train_generatorB = train_datagen.flow(X_trainB, y=Y_trainB, batch_size=batch_size,seed=seed)
val_generatorB = test_datagen.flow(X_testB,y=Y_testB,batch_size=batch_size,seed=seed)


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
        train_generatorA,
        steps_per_epoch=2500,
        epochs=2,
        validation_data=val_generatorA,
        validation_steps=100,
        verbose = 0)
historyB = modelB.fit_generator(
        train_generatorB,
        steps_per_epoch=2500,
        epochs=2,
        validation_data=val_generatorB,
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
