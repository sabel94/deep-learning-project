__author__ = "Johan Sabel, Felix Büttner, Joel Ekelöf"


from config import (image_height, image_width, model_input_path, batch_size,
                    l2_regularization_lambda, dropout_rate,
                    using_l2_regularization, using_dropout)
import os
from keras.models import Model, load_model
from keras.layers import (Input, Conv2D, BatchNormalization, Activation,
                          Dropout, UpSampling2D, Conv2DTranspose)
from keras.regularizers import l2
from keras.utils import plot_model
import scipy.misc


#Load existing model (if possible).
def load_existing_model():
    model = None
    model_directory = os.listdir(model_input_path)
    if len(model_directory) > 1:
        model_filename = "model.h5"
        model = load_model(model_input_path+model_filename)
    return model

#Instantiate model.
def get_model():
    #Load existing model (if possible).
    model = load_existing_model()
    if model != None:
        using_existing_model = True
    else:
        using_existing_model = False
        kernel_regularizer = None
        if using_l2_regularization:
            kernel_regularizer = l2(l=l2_regularization_lambda)

        #Input: Grayscale image (lab color space).
        visible = Input(shape=(image_height, image_width, 1))

        #Conv1.
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),
                       padding="same", use_bias=True,
                       kernel_initializer="he_normal", bias_initializer="zeros",
                       kernel_regularizer=kernel_regularizer)(visible)
        conv1 = Activation("relu")(conv1)
        if using_dropout:
            conv1 = Dropout(rate=dropout_rate)(conv1)
        conv1 = BatchNormalization(axis=3)(conv1)

        #Conv2.
        conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                       padding="same", use_bias=True,
                       kernel_initializer="he_normal", bias_initializer="zeros",
                       kernel_regularizer=kernel_regularizer)(conv1)
        conv2 = Activation("relu")(conv2)
        if using_dropout:
            conv2 = Dropout(rate=dropout_rate)(conv2)
        conv2 = BatchNormalization(axis=3)(conv2)

        #Conv3.
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2),
                       padding="same", use_bias=True,
                       kernel_initializer="he_normal", bias_initializer="zeros",
                       kernel_regularizer=kernel_regularizer)(conv2)
        conv3 = Activation("relu")(conv3)
        if using_dropout:
            conv3 = Dropout(rate=dropout_rate)(conv3)
        conv3 = BatchNormalization(axis=3)(conv3)

        #Conv4.
        conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                       padding="same", use_bias=True,
                       kernel_initializer="he_normal", bias_initializer="zeros",
                       kernel_regularizer=kernel_regularizer)(conv3)
        conv4 = Activation("relu")(conv4)
        if using_dropout:
            conv4 = Dropout(rate=dropout_rate)(conv4)
        conv4 = BatchNormalization(axis=3)(conv4)

        #Conv5.
        conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2),
                       padding="same", use_bias=True,
                       kernel_initializer="he_normal", bias_initializer="zeros",
                       kernel_regularizer=kernel_regularizer)(conv4)
        conv5 = Activation("relu")(conv5)
        if using_dropout:
            conv5 = Dropout(rate=dropout_rate)(conv5)
        conv5 = BatchNormalization(axis=3)(conv5)

        #Conv6.
        conv6 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                       padding="same", use_bias=True,
                       kernel_initializer="he_normal", bias_initializer="zeros",
                       kernel_regularizer=kernel_regularizer)(conv5)
        conv6 = Activation("relu")(conv6)
        if using_dropout:
            conv6 = Dropout(rate=dropout_rate)(conv6)
        conv6 = BatchNormalization(axis=3)(conv6)

        #Conv7.
        conv7 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                       padding="same", dilation_rate=(2, 2), use_bias=True,
                       kernel_initializer="he_normal", bias_initializer="zeros",
                       kernel_regularizer=kernel_regularizer)(conv6)
        conv7 = Activation("relu")(conv7)
        if using_dropout:
            conv7 = Dropout(rate=dropout_rate)(conv7)
        conv7 = BatchNormalization(axis=3)(conv7)

        #Conv8.
        conv8 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                       padding="same", dilation_rate=(2, 2), use_bias=True,
                       kernel_initializer="he_normal", bias_initializer="zeros",
                       kernel_regularizer=kernel_regularizer)(conv7)
        conv8 = Activation("relu")(conv8)
        if using_dropout:
            conv8 = Dropout(rate=dropout_rate)(conv8)
        conv8 = BatchNormalization(axis=3)(conv8)

        #Conv9.
        conv9 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                       padding="same", use_bias=True,
                       kernel_initializer="he_normal", bias_initializer="zeros",
                       kernel_regularizer=kernel_regularizer)(conv8)
        conv9 = Activation("relu")(conv9)
        if using_dropout:
            conv9 = Dropout(rate=dropout_rate)(conv9)
        conv9 = BatchNormalization(axis=3)(conv9)

        #Conv10.
        conv10 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                        padding="same", use_bias=True,
                        kernel_initializer="he_normal", bias_initializer="zeros",
                        kernel_regularizer=kernel_regularizer)(conv9)
        conv10 = Activation("relu")(conv10)
        if using_dropout:
            conv10 = Dropout(rate=dropout_rate)(conv10)
        conv10 = BatchNormalization(axis=3)(conv10)
        conv10 = UpSampling2D(size=(2, 2))(conv10)

        #Conv11.
        conv11 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                        padding="same", use_bias=True,
                        kernel_initializer="he_normal", bias_initializer="zeros",
                        kernel_regularizer=kernel_regularizer)(conv10)
        conv11 = Activation("relu")(conv11)
        if using_dropout:
            conv11 = Dropout(rate=dropout_rate)(conv11)
        conv11 = BatchNormalization(axis=3)(conv11)

        #Conv12.
        conv12 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                        padding="same", use_bias=True,
                        kernel_initializer="he_normal", bias_initializer="zeros",
                        kernel_regularizer=kernel_regularizer)(conv11)
        conv12 = Activation("relu")(conv12)
        if using_dropout:
            conv12 = Dropout(rate=dropout_rate)(conv12)
        conv12 = BatchNormalization(axis=3)(conv12)
        conv12 = UpSampling2D(size=(2, 2))(conv12)

        #Conv13.
        conv13 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                        padding="same", use_bias=True,
                        kernel_initializer="he_normal", bias_initializer="zeros",
                        kernel_regularizer=kernel_regularizer)(conv12)
        conv13 = Activation("relu")(conv13)
        if using_dropout:
            conv13 = Dropout(rate=dropout_rate)(conv13)
        conv13 = BatchNormalization(axis=3)(conv13)

        #Output.
        output = Conv2D(filters=2, kernel_size=(3, 3), strides=(1, 1),
                        padding="same", use_bias=True,
                        kernel_initializer="glorot_uniform",
                        bias_initializer="zeros")(conv13)
        output = Activation("tanh")(output)
        output = UpSampling2D(size=(2, 2))(output)

        model = Model(inputs=visible, outputs=output)
    
    #Summarize layers.
    print(model.summary())
    #plot_model(model, to_file='model_plot.png')
    return model, using_existing_model
