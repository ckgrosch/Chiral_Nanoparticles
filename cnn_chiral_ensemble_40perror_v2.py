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


save_weights = '/global/scratch/cgroschner/chiral_nanoparticles/cnn_ensemble_40pwrongleftrightlabels_v2_weights_20200916.h5'
save_history = '/global/scratch/cgroschner/chiral_nanoparticles/cnn_ensemble_40pwrongleftrightlabels_v2_history_20200916.h5'
base_name = '/global/scratch/cgroschner/chiral_nanoparticles/cnn_ensemble_40pwrongleftrightlabels_v2'

def base_cnn():
    modelE = keras.models.Sequential()
    modelE.add(Conv2D(32*2, (3, 3), input_shape=(128, 128,1)))
    modelE.add(Activation('relu'))
    modelE.add(MaxPooling2D(pool_size=(2, 2)))

    modelE.add(Conv2D(32*2, (3, 3)))
    modelE.add(Activation('relu'))
    modelE.add(MaxPooling2D(pool_size=(2, 2)))

    modelE.add(Conv2D(64*2, (3, 3)))
    modelE.add(Activation('relu'))
    modelE.add(MaxPooling2D(pool_size=(2, 2)))

    modelE.add(Conv2D(64*2, (3, 3)))
    modelE.add(Activation('relu'))
    modelE.add(MaxPooling2D(pool_size=(2, 2)))

    modelE.add(Conv2D(64*2, (3, 3)))
    modelE.add(Activation('relu'))
    modelE.add(MaxPooling2D(pool_size=(2, 2)))

    modelE.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    modelE.add(Dense(64*2))
    modelE.add(Activation('relu'))
    modelE.add(Dropout(0.5))
    modelE.add(Dense(2))
    modelE.add(Activation('softmax'))

    modelE.compile(loss='categorical_crossentropy',
                optimizer='Adadelta',
                metrics=['accuracy'])
    return modelE

def train_base_cnns(train_generator,val_generator,base_name):
    model1 = base_cnn()
    model1.fit_generator(
            train_generator,
            steps_per_epoch=2500,
            epochs=4,
            validation_data=val_generator,
            validation_steps=100,
            verbose = 0)
    model1.save_weights(base_name+'_model1.h5')
    model2 = base_cnn()
    model2.fit_generator(
            train_generator,
            steps_per_epoch=2500,
            epochs=4,
            validation_data=val_generator,
            validation_steps=100,
            verbose = 0)
    model2.save_weights(base_name+'_model2.h5')
    model3 = base_cnn()
    model3.fit_generator(
            train_generator,
            steps_per_epoch=2500,
            epochs=4,
            validation_data=val_generator,
            validation_steps=100,
            verbose = 0)
    model3.save_weights(base_name+'_model3.h5')
    model4 = base_cnn()
    model4.fit_generator(
            train_generator,
            steps_per_epoch=2500,
            epochs=10,
            validation_data=val_generator,
            validation_steps=100,
            verbose = 0)
    model4.save_weights(base_name+'_model4.h5')
    model5 = base_cnn()
    model5.fit_generator(
            train_generator,
            steps_per_epoch=2500,
            epochs=4,
            validation_data=val_generator,
            validation_steps=100,
            verbose = 0)
    model5.save_weights(base_name+'_model5.h5')
    model6 = base_cnn()
    model6.fit_generator(
            train_generator,
            steps_per_epoch=2500,
            epochs=4,
            validation_data=val_generator,
            validation_steps=100,
            verbose = 0)
    model6.save_weights(base_name+'_model6.h5')
    model7 = base_cnn()
    model7.fit_generator(
            train_generator,
            steps_per_epoch=2500,
            epochs=4,
            validation_data=val_generator,
            validation_steps=100,
            verbose = 0)
    model7.save_weights(base_name+'_model7.h5')
    model8 = base_cnn()
    model8.fit_generator(
            train_generator,
            steps_per_epoch=2500,
            epochs=4,
            validation_data=val_generator,
            validation_steps=100,
            verbose = 0)
    model8.save_weights(base_name+'_model8.h5')
    model9 = base_cnn()
    model9.fit_generator(
            train_generator,
            steps_per_epoch=2500,
            epochs=4,
            validation_data=val_generator,
            validation_steps=100,
            verbose = 0)
    model9.save_weights(base_name+'_model9.h5')
    model10 = base_cnn()
    model10.fit_generator(
            train_generator,
            steps_per_epoch=2500,
            epochs=4,
            validation_data=val_generator,
            validation_steps=100,
            verbose = 0)
    model10.save_weights(base_name+'_model10.h5')
    model1.trainable = False
    model2.trainable = False
    model3.trainable = False
    model4.trainable = False
    model5.trainable = False
    model6.trainable = False
    model7.trainable = False
    model8.trainable = False
    model9.trainable = False
    model10.trainable = False
    return model1,model2,model3,model4,model5,model6,model7,model8,model9,model10

def ensemble_net(model1,model2,model3,model4,model5,model6,model7,model8,model9,model10,input_shape=(128,128,1)):

    inputs = Input(shape=input_shape)
    m1 = model1(inputs,training=False)
    m2 = model2(inputs,training=False)
    m3 = model3(inputs,training=False)
    m4 = model4(inputs,training=False)
    m5 = model5(inputs,training=False)
    m6 = model6(inputs,training=False)
    m7 = model7(inputs,training=False)
    m8 = model8(inputs,training=False)
    m9 = model9(inputs,training=False)
    m10 = model10(inputs,training=False)
    avg = Average()([m1,m2,m3,m4,m5,m6,m7,m8,m9,m10])
    outputs = Dense(2)(avg)
    model = Model(inputs, outputs)

    return model



right_images = np.load('/global/scratch/cgroschner/chiral_nanoparticles/20200514_right__Chiral_D_Large_TIFF_Cropped_four_rows_sel_NPs_rotated.npy')



new_left_images = []
new_left_labels = []
new_right_images = []
new_right_labels = []


for img in right_images:
    new_left_images.append(np.fliplr(img))
    new_left_labels.append([1,0])
    new_right_images.append(img)
    new_right_labels.append([0,1])
new_left_images = np.array(new_left_images)
new_right_images = np.array(new_right_images)


split = int(191*0.40)


for idx in np.arange(0,split):
    new_right_images[idx] = np.fliplr(new_right_images[idx])

for idx in np.arange(0,split):
    new_left_images[idx] = np.fliplr(new_left_images[idx])

right_img_shuff, right_label_shuff = utils.shuffle(new_right_images[:191], new_right_labels[:191],random_state=0)
left_img_shuff, left_label_shuff = utils.shuffle(new_left_images[:191], new_left_labels[:191],random_state=0)

X_train = np.concatenate((right_img_shuff[:191],left_img_shuff[:191]),axis =0)
Y_train = np.concatenate((right_label_shuff[:191],left_label_shuff[:191]),axis = 0)

split = int(95*0.40)

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

model1,model2,model3,model4,model5,model6,model7,model8,model9,model10 = train_base_cnns(train_generator,val_generator,base_name)
ensemble_model = ensemble_net(model1,model2,model3,model4,model5,model6,model7,model8,model9,model10)
ensemble_model.compile(loss='mean_absolute_error',
              optimizer='Adadelta',
              metrics=['accuracy'])
history = ensemble_model.fit_generator(
        train_generator,
        steps_per_epoch=2500,
        epochs=5,
        validation_data=val_generator,
        validation_steps=100,
        verbose = 0)
ensemble_model.save_weights(save_weights)
h = h5py.File(save_history,'w')
h_keys = history.history.keys()
print(h_keys)
for k in h_keys:
    h.create_dataset(k,data=history.history[k])
h.close()
X_val = np.concatenate((new_right_images[286:],new_left_images[286:]),axis = 0)
Y_val = np.concatenate((new_right_labels[286:],new_left_labels[286:]),axis = 0)
X_val = X_val/X_val.max()
X_val = np.expand_dims(X_val,axis=3)
X_val, Y_val = utils.shuffle(X_val, Y_val,random_state=0)
print(ensemble_model.evaluate(X_val,Y_val))
