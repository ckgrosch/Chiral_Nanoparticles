import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential, load_model
import numpy as np
from tensorflow.keras.losses import categorical_crossentropy as cce
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.utils import to_categorical

import h5py
from sklearn import utils
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



save_weightsA = '/global/scratch/cgroschner/chiral_nanoparticles/coteach_00pwrongleftrightlabels_weights_v1A_20200914.h5'
save_historyA = '/global/scratch/cgroschner/chiral_nanoparticles/coteach_00pwrongleftrightlabels_history_v1A_20200914.h5'
save_weightsB = '/global/scratch/cgroschner/chiral_nanoparticles/coteach_00pwrongleftrightlabels_weights_v1B_20200914.h5'
save_historyB = '/global/scratch/cgroschner/chiral_nanoparticles/coteach_00pwrongleftrightlabels_history_v1B_20200914.h5'

right_images = np.load('/global/scratch/cgroschner/chiral_nanoparticles/20200514_right__Chiral_D_Large_TIFF_Cropped_four_rows_sel_NPs_rotated.npy')
# right_labels = [[1,0] for i in np.arange(0,len(right_images))]




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



right_img_shuff, right_label_shuff = utils.shuffle(new_right_images[:191], new_right_labels[:191],random_state=0)
left_img_shuff, left_label_shuff = utils.shuffle(new_left_images[:191], new_left_labels[:191],random_state=0)

X_trainA = np.concatenate((right_img_shuff[:95],left_img_shuff[:95]),axis =0)
Y_trainA = np.concatenate((right_label_shuff[:95],left_label_shuff[:95]),axis = 0)
X_trainB = np.concatenate((right_img_shuff[95:-1],left_img_shuff[95:-1]),axis =0)
Y_trainB = np.concatenate((right_label_shuff[95:-1],left_label_shuff[95:-1]),axis = 0)


right_img_shuff, right_label_shuff = utils.shuffle(new_right_images[191:286], new_right_labels[191:286],random_state=0)
left_img_shuff, left_label_shuff = utils.shuffle(new_left_images[191:286], new_left_labels[191:286],random_state=0)


X_testA = np.concatenate((right_img_shuff[:47],left_img_shuff[:47]),axis = 0)
Y_testA = np.concatenate((right_label_shuff[:47],left_label_shuff[:47]),axis = 0)
X_testB = np.concatenate((right_img_shuff[47:-1],left_img_shuff[47:-1]),axis = 0)
Y_testB = np.concatenate((right_label_shuff[47:-1],left_label_shuff[47:-1]),axis = 0)

X_trainA = X_trainA/X_trainA.max()
X_testA = X_testA/X_testA.max()
X_trainB= X_trainB/X_trainB.max()
X_testB = X_testB/X_testB.max()

X_trainA = np.expand_dims(X_trainA,axis=3)
X_testA = np.expand_dims(X_testA,axis=3)
X_trainB = np.expand_dims(X_trainB,axis=3)
X_testB = np.expand_dims(X_testB,axis=3)

X_trainA, Y_trainA = utils.shuffle(X_trainA, Y_trainA,random_state=0)
X_testA, Y_testA = utils.shuffle(X_testA, Y_testA,random_state=0)
X_trainB,Y_trainB = utils.shuffle(X_trainB, Y_trainB,random_state=0)
X_testB, Y_testB = utils.shuffle(X_testB, Y_testB,random_state=0)

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


modelA = keras.models.Sequential()
modelA.add(Conv2D(42, (3, 3), input_shape=(128, 128,1)))
modelA.add(Activation('relu'))
modelA.add(MaxPooling2D(pool_size=(2, 2)))
modelA.add(Conv2D(42, (3, 3)))
modelA.add(Activation('relu'))
modelA.add(MaxPooling2D(pool_size=(2, 2)))

modelA.add(Conv2D(74, (3, 3)))
modelA.add(Activation('relu'))
modelA.add(MaxPooling2D(pool_size=(2, 2)))

modelA.add(Conv2D(74, (3, 3)))
modelA.add(Activation('relu'))
modelA.add(MaxPooling2D(pool_size=(2, 2)))

modelA.add(Conv2D(74, (3, 3)))
modelA.add(Activation('relu'))
modelA.add(MaxPooling2D(pool_size=(2, 2)))

modelA.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
modelA.add(Dense(74))
modelA.add(Activation('relu'))
modelA.add(Dropout(0.5))
modelA.add(Dense(2))
modelA.add(Activation('softmax'))

modelA.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=0.001),metrics=['accuracy'])

modelB = keras.models.Sequential()
modelB.add(Conv2D(22, (3, 3), input_shape=(128, 128,1)))
modelB.add(Activation('relu'))
modelB.add(MaxPooling2D(pool_size=(2, 2)))

modelB.add(Conv2D(22, (3, 3)))
modelB.add(Activation('relu'))
modelB.add(MaxPooling2D(pool_size=(2, 2)))

modelB.add(Conv2D(54, (3, 3)))
modelB.add(Activation('relu'))
modelB.add(MaxPooling2D(pool_size=(2, 2)))

modelB.add(Conv2D(54, (3, 3)))
modelB.add(Activation('relu'))
modelB.add(MaxPooling2D(pool_size=(2, 2)))

modelB.add(Conv2D(54, (3, 3)))
modelB.add(Activation('relu'))
modelB.add(MaxPooling2D(pool_size=(2, 2)))

modelB.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
modelB.add(Dense(54))
modelB.add(Activation('relu'))
modelB.add(Dropout(0.5))
modelB.add(Dense(2))
modelB.add(Activation('softmax'))

modelB.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])



# modelCheckpoint = ModelCheckpoint(save_weights,
#                                   monitor = 'val_accuracy',
#                                   save_best_only = True,
#                                   mode = 'max',
#                                   verbose = 2,
#                                   save_weights_only = True)
# callbacks_list = [modelCheckpoint]

historyA = modelA.fit_generator(
        train_generatorA,
        steps_per_epoch=2500,
        epochs=1,
        validation_data=val_generatorA,
        validation_steps=100,
        verbose = 0)
historyB = modelB.fit_generator(
        train_generatorB,
        steps_per_epoch=2500,
        epochs=1,
        validation_data=val_generatorB,
        validation_steps=100,
        verbose = 0)

def new_labels_from_loss(model1,model2,X1,Y1,X2,Y2):
    pred1a = model1.predict(X1)
    pred2a = model2.predict(X2)
    loss1a = cce(Y1,pred1a)
    loss2a = cce(Y2,pred2a)
    pred1b = model1.predict(X2)
    pred2b = model2.predict(X1)
    loss1b = cce(Y2,pred1b)
    loss2b = cce(Y1,pred2b)
    for idx, l1b in enumerate(loss1b):
        if l1b < loss2a[idx]:
            Y2[idx] = pred1b[idx]
    for idx, l2b in enumerate(loss2b):
        if l2b < loss1a[idx]:
            Y1[idx] = pred2b[idx]
    return Y1, Y2

Y_trainA, Y_trainB = new_labels_from_loss(modelA,modelB,X_trainA,Y_trainA,X_trainB,Y_trainB)

train_generatorA = train_datagen.flow(X_trainA, y=Y_trainA, batch_size=batch_size,seed=seed)
train_generatorB = train_datagen.flow(X_trainB, y=Y_trainB, batch_size=batch_size,seed=seed)

historyA = modelA.fit_generator(
        train_generatorA,
        steps_per_epoch=2500,
        epochs=1,
        validation_data=val_generatorA,
        validation_steps=100,
        verbose = 0)
historyB = modelB.fit_generator(
        train_generatorB,
        steps_per_epoch=2500,
        epochs=1,
        validation_data=val_generatorB,
        validation_steps=100,
        verbose = 0)

Y_trainA, Y_trainB = new_labels_from_loss(modelA,modelB,X_trainA,Y_trainA,X_trainB,Y_trainB)

train_generatorA = train_datagen.flow(X_trainA, y=Y_trainA, batch_size=batch_size,seed=seed)
train_generatorB = train_datagen.flow(X_trainB, y=Y_trainB, batch_size=batch_size,seed=seed)

historyA = modelA.fit_generator(
        train_generatorA,
        steps_per_epoch=2500,
        epochs=1,
        validation_data=val_generatorA,
        validation_steps=100,
        verbose = 0)
historyB = modelB.fit_generator(
        train_generatorB,
        steps_per_epoch=2500,
        epochs=1,
        validation_data=val_generatorB,
        validation_steps=100,
        verbose = 0)

Y_trainA, Y_trainB = new_labels_from_loss(modelA,modelB,X_trainA,Y_trainA,X_trainB,Y_trainB)

train_generatorA = train_datagen.flow(X_trainA, y=Y_trainA, batch_size=batch_size,seed=seed)
train_generatorB = train_datagen.flow(X_trainB, y=Y_trainB, batch_size=batch_size,seed=seed)

historyA = modelA.fit_generator(
        train_generatorA,
        steps_per_epoch=2500,
        epochs=1,
        validation_data=val_generatorA,
        validation_steps=100,
        verbose = 0)
historyB = modelB.fit_generator(
        train_generatorB,
        steps_per_epoch=2500,
        epochs=1,
        validation_data=val_generatorB,
        validation_steps=100,
        verbose = 0)

Y_trainA, Y_trainB = new_labels_from_loss(modelA,modelB,X_trainA,Y_trainA,X_trainB,Y_trainB)

train_generatorA = train_datagen.flow(X_trainA, y=Y_trainA, batch_size=batch_size,seed=seed)
train_generatorB = train_datagen.flow(X_trainB, y=Y_trainB, batch_size=batch_size,seed=seed)


modelA.save_weights(save_weightsA)
h = h5py.File(save_historyA,'w')
h_keys = historyA.history.keys()
print(h_keys)
for k in h_keys:
    h.create_dataset(k,data=historyA.history[k])
h.close()
modelB.save_weights(save_weightsB)
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
print('modelA: ', modelA.evaluate(X_val,Y_val))
print('modelB: ', modelB.evaluate(X_val,Y_val))
