__author__ = "Johan Sabel, Felix Büttner, Joel Ekelöf"


from config import (image_height, image_width, model_input_path, batch_size,
                    l2_regularization_lambda, dropout_rate,
                    using_l2_regularization, using_dropout)
import os
from keras.models import Model, load_model
from keras.layers import (Input, Conv2D, BatchNormalization, Activation,
                          Dropout, UpSampling2D, Conv2DTranspose, MaxPooling2D,Add)
from keras.regularizers import l2
from keras.utils import plot_model
import scipy.misc

def normal_block(input,filterSize):
  layer = Conv2D(filters=filterSize, kernel_size=(3, 3), strides=(1, 1),
                  padding="same", use_bias=True,
                  kernel_initializer="he_normal", bias_initializer="zeros")(input)
  layer = Activation("relu")(layer)
  layer = BatchNormalization()(layer)

  return layer

def upsampling_block(input, filterSize):
  layer = Conv2D(filters=filterSize, kernel_size=(3, 3), strides=(1, 1),
                  padding="same", use_bias=True,
                  kernel_initializer="he_normal", bias_initializer="zeros")(input)
  layer = Activation("relu")(layer)
  layer = BatchNormalization()(layer)
  layer = Conv2DTranspose(filters = filterSize, kernel_size = (3,3), strides = (2,2),
          padding = "same", name = ("Transpose" + str(filterSize)))(layer)

  return layer

def downsample_block(input,filterSizes):

  #shortcut here need an extra layer, since it has the wrong dimensions and needs to be downsampled
  identity = input
  identity = Conv2D(filters = filterSizes[2], kernel_size = (1,1),
              strides = (2,2), padding = "same")(identity)

  #first block (notice stride = 2, to downsample.)
  layer = Conv2D(filters = filterSizes[0], kernel_size = (1,1),
                 strides = (2,2), padding = "same", use_bias = True,
                 kernel_initializer = "he_normal", bias_initializer = "zeros")(input)
  layer = Activation("relu")(layer)
  layer = BatchNormalization()(layer)

  #second block
  layer = Conv2D(filters = filterSizes[1], kernel_size = (3,3),
                 strides = (1,1), padding = "same", use_bias = True,
                 kernel_initializer = "he_normal", bias_initializer = "zeros")(layer)
  layer = Activation("relu")(layer)
  layer = BatchNormalization()(layer)

  #third block
  layer = Conv2D(filters = filterSizes[2], kernel_size = (1,1),
                 strides = (1,1), padding = "same", use_bias = True,
                 kernel_initializer = "he_normal", bias_initializer = "zeros")(layer)

  #shortcut
  layer = Add()([identity,layer])
  layer = Activation("relu")(layer)
  layer = BatchNormalization()(layer)

  return layer

  #code for the resnet inspired by https://arxiv.org/pdf/1512.03385.pdf and https://engmrk.com/residual-networks-resnets/

def same_size_block(input, filterSizes):

  #save identity to be added in the end
  identity = input

  #first block
  layer = Conv2D(filters = filterSizes[0], kernel_size = (1,1),
                 strides = (1,1), padding = "same", use_bias = True,
                 kernel_initializer = "he_normal", bias_initializer = "zeros")(input)
  layer = Activation("relu")(layer)
  layer = BatchNormalization()(layer)


  #second block
  layer = Conv2D(filters = filterSizes[1], kernel_size = (3,3),
                 strides = (1,1), padding = "same", use_bias = True,
                 kernel_initializer = "he_normal", bias_initializer = "zeros")(layer)
  layer = Activation("relu")(layer)
  layer = BatchNormalization()(layer)

  #third block
  layer = Conv2D(filters = filterSizes[2], kernel_size = (1,1),
                 strides = (1,1), padding = "same", use_bias = True,
                 kernel_initializer = "he_normal", bias_initializer = "zeros")(layer)

  #shortcut
  layer = Add()([layer,identity])
  layer = Activation("relu")(layer)
  layer = BatchNormalization()(layer)

  return layer

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
        inputLayer = Input(shape=(image_height,image_width,1), name = "Input")

        conv1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                       padding="same", use_bias=True,
                       kernel_initializer="he_normal", bias_initializer="zeros",
                       name = "1-7x7")(inputLayer)
        conv1 = Activation("relu", name = "1-relu")(conv1)
        conv1 = BatchNormalization(name = "1-batchNorm")(conv1)

        pool = MaxPooling2D((3,3),strides = 2, name = "2-pool")(conv1)

        L2 = downsample_block(pool,[64,64,256])
        L2 = same_size_block(L2,[64,64,256])
        L2 = same_size_block(L2,[64,64,256])

        L3 = downsample_block(L2,[128,128,512])
        L3 = same_size_block(L3,[128,128,512])
        L3 = same_size_block(L3,[128,128,512])
        L3 = same_size_block(L3,[128,128,512])

        L4 = downsample_block(L3,[256,256,1024])
        L4 = same_size_block(L4,[256,256,1024])
        L4 = same_size_block(L4,[256,256,1024])
        L4 = same_size_block(L4,[256,256,1024])
        L4 = same_size_block(L4,[256,256,1024])
        L4 = same_size_block(L4,[256,256,1024])

        L5 = downsample_block(L4, [512,512,2048])
        L5 = same_size_block(L5, [512,512,2048])
        L5 = same_size_block(L5, [512,512,2048])

        #we have 6 downsamples (first thing, maxpool, 4 downsamples (L2,L3,L4,L5))
        #so have to have 6 upsamples

        up = upsampling_block(L5,2048)#1
        up = normal_block(up,1024)
        up = upsampling_block(up,512)#2
        up = normal_block(up,256)
        up = upsampling_block(up,128)#3
        up = normal_block(up,64)
        up = upsampling_block(up,32)#4


        #Output.
        output = Conv2D(filters=2, kernel_size=(3, 3), strides=(1, 1),
                        padding="same", use_bias=True,
                        kernel_initializer="glorot_uniform",
                        bias_initializer="zeros")(up)
        output = Activation("tanh")(output)
        output = Conv2DTranspose(filters = 2, kernel_size = (3,3), strides = (2,2), padding = "same", name = "TransposeLast")(output)#6


        model = Model(input = inputLayer, output = output)

    #Summarize layers.
    # print(model.summary())
    #plot_model(model, to_file='model_plot.png')
    return model, using_existing_model
