__author__ = "Johan Sabel, Felix Büttner, Joel Ekelöf"


from config import image_height, image_width, batch_size
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from skimage import color


#Get training image generator (image augmentation).
def get_image_generator(dataset):
    if dataset == "training":
        datagen = ImageDataGenerator(rescale=1./255,
                                     rotation_range=20,
                                     shear_range=0.3,
                                     zoom_range=0.3,
                                     horizontal_flip=True)
    elif dataset == "validation" or dataset == "testing":
        datagen = ImageDataGenerator(rescale=1./255)
    return datagen

#Get batches of training samples.
def get_samples(datagen, data_path, shuffle):
    for batch in datagen.flow_from_directory(os.path.abspath(data_path),
                                             target_size=(image_height, image_width),
                                             batch_size=batch_size,
                                             color_mode="rgb",
                                             class_mode=None,
                                             shuffle=shuffle):
        #Convert to Lab color space, i.e. shape: (batch_size, height, width, 3).
        #Lab has 3 channels: grayscale, a and b.
        batch = color.rgb2lab(batch)
        #Input: grayscale images (channel 0).
        X_batch = batch[:, :, :, 0] #Shape: (batch_size, height, width).
        X_batch = X_batch[..., np.newaxis] #Shape: (batch_size, height, width, 1).
        #Targets: a and b color channels for the images (channel 1 and 2).
        Y_batch = batch[:, :, :, 1:]
        #Change Lab color space L channel interval from [0, 100] to [0, 1].
        X_batch = X_batch / 100
        #Change Lab color space a and b channel interval from [-128, 127] to [-1, 127/128].
        Y_batch = Y_batch / 128
        yield (X_batch, Y_batch)
