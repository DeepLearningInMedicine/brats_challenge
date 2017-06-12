import numpy as np
import math
import nibabel as nib
import os, sys, re, time
import keras
import tensorflow as tf
from format_data import get_images_numpy
import os.path
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Activation,Dropout
from keras import backend as K
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, UpSampling3D, ZeroPadding3D
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import Layer
import keras.metrics
import keras.losses
from keras_contrib.layers.convolutional import Deconvolution3D
from keras.callbacks import ModelCheckpoint

def get_samples():
    fnamei = "data/images.npy"
    fnamel = "data/labels.npy"
    if os.path.isfile(fnamei):
        print("Loading images")
        images = np.load(fnamei)
        labels = np.load(fnamel)

    else: 
        print("getting images")
        images, labels = get_images_numpy()
        print("editing labels")
        labels[labels > 0] = 1
        print("cropping images")
        # In[3]:

        # crop image down
        n,x,y,z = images.shape
        # low_x = 0;
        # high_x = x;
        # low_y = 0;
        # high_y = y
        # for i in range(n):
        #     for j in range(x):
        #         for k in range(y):
        #             for l in range(z):
        #                 if images[i,j,k,l] != 0:
        #                     if low_x < 
        # region = images != 0
        # reg = images[region]
        min_x = 25
        max_x = 215
        min_y = 25
        max_y = 215
        images = images[:, min_x:max_x, min_y:max_y, :154]
        labels = labels[:, min_x:max_x, min_y:max_y, :154]
        print("saving!")
        np.save(fnamei, images);
        np.save(fnamel, labels);
        #from numpy import array, argwhere
        print("Done getting images")

        #min_x = 240
        #max_x = 0
        #min_y = 240
        #max_y = 0
        #for i in range(n):
        #    B = argwhere(images[i])
        #    (ystart, xstart, zstart), (ystop, xstop, zstop) = B.min(0), B.max(0) + 1
        #    min_x = xstart if xstart < min_x else min_x
        #    min_y = ystart if ystart < min_y else min_y
        #    max_x = xstop if xstop > max_x else max_x
        #    max_y = ystop if ystop > max_y else max_y

        #print(min_x, max_x, min_y, max_y)
        #images = images[:, min_x:max_x, min_y:max_y, :154]
        #labels = labels[:, min_x:max_x, min_y:max_y, :154]
        #np.save(fnamei, images);
        #np.save(fnamel, labels);
    
    return images, labels

def reshape_images(images, labels):
    print("Reshaping images")
    if images.ndim != 4:
        images = np.squeeze(images)
    if labels.ndim != 4:
        labels = np.squeeze(labels)
    if images.ndim != 5:
        images = images[...,np.newaxis]
    if labels.ndim != 5:
        labels = labels[...,np.newaxis]
        
    n,x,y,z,c = images.shape
    print("Number of samples: ", n)
    return images, labels, images.shape

class Round(Layer):
    def __init__(self, **kwargs):
        super(Round, self).__init__(**kwargs)

    def get_output(self, train=False):
        X = self.get_input(train)
        return K.round(X)

    def get_config(self):
        config = {"name": self.__class__.__name__}
        base_config = super(Round, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_val(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_ff = K.flatten(y_pred)
    y_pred_f = K.round(y_pred_ff)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def build_model(shape):
    learning_rate = 1e-4
    dropout_rate = .2
    num_filters = [32, 64, 128, 256, 512]

    n,x,y,z,c = shape
    inputs = Input((x,y,z,c))

    conv1 = Conv3D(num_filters[0], (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(num_filters[0], (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(5, 5, 1))(conv1)
    pool1 = Dropout(dropout_rate)(pool1)

    conv2 = Conv3D(num_filters[1], (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(num_filters[1], (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)
    pool2 = Dropout(dropout_rate)(pool2)

    conv3 = Conv3D(num_filters[2], (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(num_filters[2], (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)
    pool3 = Dropout(dropout_rate)(pool3)

    conv4 = Conv3D(num_filters[3], (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(num_filters[3], (3, 3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 1))(conv4)
    pool4 = Dropout(dropout_rate)(pool4)

    conv5 = Conv3D(num_filters[4], (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Dropout(dropout_rate)(conv5)

    out_shape2 = (None, 8, 8, 154, num_filters[3])
    up6 = Deconvolution3D(num_filters[3], (3,3,3), out_shape2, padding='same', strides=(2,2,1), data_format="channels_last")(conv5)
    up6 = Dropout(dropout_rate)(up6)
    conv6 = Conv3D(num_filters[3], (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(num_filters[3], (3, 3, 3), activation='relu', padding='same')(conv6)

    out_shape3 = (None, 16, 16, 154, num_filters[2])
    up7 = Deconvolution3D(num_filters[2], (3,3,3), out_shape3, padding='same', strides=(2,2,1), data_format="channels_last")(conv6)
    up7 = Dropout(dropout_rate)(up7)
    conv7 = Conv3D(num_filters[2], (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(num_filters[2], (3, 3, 3), activation='relu', padding='same')(conv7)

    out_shape4 = (None, 32, 32, 154, num_filters[1])
    up8 = Deconvolution3D(num_filters[1], (3,3,3), out_shape4, padding='same', strides=(2,2,1), data_format="channels_last")(conv7)
    up8 = Dropout(dropout_rate)(up8)
    up8_pad = ZeroPadding3D(padding=(3, 3, 0), data_format=None)(up8)
    conv8 = Conv3D(num_filters[1], (3, 3, 3), activation='relu', padding='same')(up8_pad)
    conv8 = Conv3D(num_filters[1], (3, 3, 3), activation='relu', padding='same')(conv8)

    out_shape5 = (None, 190, 190, 154, num_filters[0])
    up9 = Deconvolution3D(num_filters[0], (3,3,3), out_shape5, padding='same', strides=(5,5,1), data_format="channels_last")(conv8)
    up9 = Dropout(dropout_rate)(up9)
    conv9 = Conv3D(num_filters[0], (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(num_filters[0], (3, 3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[dice_coef, dice_coef_val])

    return model

keras.losses.dice_coef_loss = dice_coef_loss 
keras.metrics.dice_coef = dice_coef
keras.metrics.dice_coef_val = dice_coef_val
keras.layers.Round = Round

def run():
    images, labels = get_samples()
    images, labels, shape = reshape_images(images, labels)

    model_path = 'models/brats_model7.h5'
    num_epochs = 15 
    batch_size = 1 
    num_training = 240 

    model = None
    if os.path.isfile(model_path):
        print("Loading Model!")
        model = load_model(model_path)
    else:
        print("Building model!")
        model = build_model(shape)
        model.summary()
    
    try:
        print("Testing model..")
        preds = model.predict(images[0:1])
        print("Does output shape match? ", labels[0:1].shape == preds.shape)
        print("Preds shape = ", preds.shape)
    except RuntimeError as e:
        print(e)
        print("Did not work! Exception raised")
        return

    print("Training...")
    num_test = 285 - num_training # test on rest 
    train = list(range(num_training))
    test = list(range(num_training, num_training+num_test))

    cp = ModelCheckpoint(model_path)
    model.fit(x=images[train], y=labels[train], batch_size=batch_size, verbose=1, epochs=num_epochs, validation_split=.2, callbacks=[cp])

    print("Saving Model!")
    model.save(model_path)

run()
