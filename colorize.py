__author__ = "Johan Sabel, Felix Büttner, Joel Ekelöf"


import os
import math
import numpy as np
import matplotlib.pyplot as plt
import json
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, UpSampling2D
from keras.regularizers import l2
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from skimage import color
from skimage.io import imsave
import scipy.misc


#Image dimensions.
image_height = 256
image_width = 256
#Input paths.
model_input_path = "input/model/"
history_input_path = "input/history/"
training_path = "input/datasets/training/"
validation_path = "input/datasets/validation/"
testing_path = "input/datasets/testing/"
#Output paths.
model_output_path = "output/model/"
history_output_path = "output/history/"
loss_plot_path = "output/loss_plot/"
ground_truth_test_data_path = "output/ground_truth_test_data/"
grayscale_test_data_path = "output/grayscale_test_data/"
colorized_test_data_path = "output/colorized_test_data/"
#Parameters.
batch_size = 40
num_epochs = 50
learning_rate = 0.005


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
        #Input: Grayscale image (lab color space).
        visible = Input(shape=(image_height, image_width, 1))

        #Conv1.
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),
                       padding="same", use_bias=True,
                       kernel_initializer="he_normal",
                       bias_initializer="zeros")(visible)
        conv1 = Activation("relu")(conv1)
        conv1 = BatchNormalization()(conv1)

        #Conv2.
        conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                       padding="same", use_bias=True,
                       kernel_initializer="he_normal",
                       bias_initializer="zeros")(conv1)
        conv2 = Activation("relu")(conv2)
        conv2 = BatchNormalization()(conv2)

        #Conv3.
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2),
                       padding="same", use_bias=True,
                       kernel_initializer="he_normal",
                       bias_initializer="zeros")(conv2)
        conv3 = Activation("relu")(conv3)
        conv3 = BatchNormalization()(conv3)

        #Conv4.
        conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                       padding="same", use_bias=True,
                       kernel_initializer="he_normal",
                       bias_initializer="zeros")(conv3)
        conv4 = Activation("relu")(conv4)
        conv4 = BatchNormalization()(conv4)

        #Conv5.
        conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2),
                       padding="same", use_bias=True,
                       kernel_initializer="he_normal",
                       bias_initializer="zeros")(conv4)
        conv5 = Activation("relu")(conv5)
        conv5 = BatchNormalization()(conv5)

        #Conv6.
        conv6 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                       padding="same", use_bias=True,
                       kernel_initializer="he_normal",
                       bias_initializer="zeros")(conv5)
        conv6 = Activation("relu")(conv6)
        conv6 = BatchNormalization()(conv6)

        #Conv7.
        conv7 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                       padding="same", dilation_rate=(2, 2), use_bias=True,
                       kernel_initializer="he_normal",
                       bias_initializer="zeros")(conv6)
        conv7 = Activation("relu")(conv7)
        conv7 = BatchNormalization()(conv7)

        #Conv8.
        conv8 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                       padding="same", dilation_rate=(2, 2), use_bias=True,
                       kernel_initializer="he_normal",
                       bias_initializer="zeros")(conv7)
        conv8 = Activation("relu")(conv8)
        conv8 = BatchNormalization()(conv8)

        #Conv9.
        conv9 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                       padding="same", use_bias=True,
                       kernel_initializer="he_normal",
                       bias_initializer="zeros")(conv8)
        conv9 = Activation("relu")(conv9)
        conv9 = BatchNormalization()(conv9)

        #Conv10.
        conv10 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                        padding="same", use_bias=True,
                        kernel_initializer="he_normal",
                        bias_initializer="zeros")(conv9)
        conv10 = Activation("relu")(conv10)
        conv10 = BatchNormalization()(conv10)
        conv10 = UpSampling2D(size=(2, 2))(conv10)

        #Conv11.
        conv11 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                        padding="same", use_bias=True,
                        kernel_initializer="he_normal",
                        bias_initializer="zeros")(conv10)
        conv11 = Activation("relu")(conv11)
        conv11 = BatchNormalization()(conv11)

        #Conv12.
        conv12 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                        padding="same", use_bias=True,
                        kernel_initializer="he_normal",
                        bias_initializer="zeros")(conv11)
        conv12 = Activation("relu")(conv12)
        conv12 = BatchNormalization()(conv12)
        conv12 = UpSampling2D(size=(2, 2))(conv12)

        #Conv13.
        conv13 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                        padding="same", use_bias=True,
                        kernel_initializer="he_normal",
                        bias_initializer="zeros")(conv12)
        conv13 = Activation("relu")(conv13)
        conv13 = BatchNormalization()(conv13)

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

#Get training image generator (image augmentation).
def get_image_generator(dataset):
    if dataset == "training":
        datagen = ImageDataGenerator(rescale=1/255,
                                     rotation_range=20,
                                     shear_range=0.3,
                                     zoom_range=0.3,
                                     horizontal_flip=True)
    elif dataset == "validation" or dataset == "testing":
        datagen = ImageDataGenerator(rescale=1/255)
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

#Train the network.
def train_model(model, using_existing_model,training_datagen, validation_datagen):
    if not using_existing_model:
        optimizer = Adam(lr=learning_rate)
        model.compile(loss="mse", optimizer=optimizer)
    num_training_samples = len(os.listdir(training_path+"training/"))
    num_steps_per_epoch = math.floor(num_training_samples / batch_size)
    num_validation_samples = len(os.listdir(validation_path+"validation/"))
    num_val_steps_per_epoch = math.floor(num_validation_samples / batch_size)
    history = model.fit_generator(generator=get_samples(training_datagen, training_path, True),
                                  steps_per_epoch=num_steps_per_epoch,
                                  epochs=num_epochs,
                                  validation_data=get_samples(validation_datagen, validation_path, False),
                                  validation_steps=num_val_steps_per_epoch)
    #Save the model (weights etc.).
    model.save(model_output_path+"model.h5")
    #Save the training history (loss, accuracy etc. for each epoch).
    save_history(history, using_existing_model)
    return model

#Save the training history (loss, accuracy etc. for each epoch) to file.
def save_history(history, using_existing_model):
    history_dict = history.history
    #Merge training history with existing training history if possilbe,
    #i.e. if training has been continued using an existing model.
    if using_existing_model:
        history_directory = os.listdir(history_input_path)
        if len(history_directory) > 1:
            prev_history_dict = json.load(open(history_input_path+
                                               "history.json", 'r'))
            for key in history_dict:
                history_dict[key] = prev_history_dict[key] + history_dict[key]
    json.dump(history_dict, open(history_output_path+"history.json", 'w'))

#Plot the training and validation loss for each training epoch.
def plot_loss():
    history_directory = os.listdir(history_output_path)
    if len(history_directory) > 1:
        history_dict = json.load(open(history_output_path+"history.json", 'r'))
        training_loss = history_dict["loss"]
        validation_loss = history_dict["val_loss"]
        epochs = list(range(1, len(training_loss)+1))
        fig, ax = plt.subplots()
        ax.plot(epochs, training_loss, linewidth=1, color="g", label="Training Loss")
        ax.plot(epochs, validation_loss, linewidth=1, color="r", label="Validation Loss")
        ax.set_xlim(1, len(epochs)+1)
        plt.xticks(plt.xticks()[0][1:])
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Graph")
        plt.legend()
        plt.savefig(loss_plot_path+'loss_plot.svg')

#Save the input images (grayscale), the corresponding output images (colorized)
#and the ground truth images to directory.
def save_images_to_directory(X_batch, Y_batch_predicted, Y_batch_true, batch_index):
    batch_size = X_batch.shape[0]
    for image_index in range(batch_size):
        lab_image_colorized = np.zeros((image_height, image_width, 3))
        #Set grayscale channel.
        lab_image_colorized[:, :, 0] = X_batch[image_index, :, :, 0]
        lab_image_grayscale = np.copy(lab_image_colorized)
        lab_image_true = np.copy(lab_image_colorized)
        #Set a and b channels.
        lab_image_colorized[:, :, 1:] = Y_batch_predicted[image_index, :, :, :]
        lab_image_true[:, :, 1:] = Y_batch_true[image_index, :, :, :]
        #Convert to rgb color space.
        rgb_image_grayscale = color.lab2rgb(lab_image_grayscale)
        rgb_image_colorized = color.lab2rgb(lab_image_colorized)
        rgb_image_true = color.lab2rgb(lab_image_true)
        #Save to directory.
        imsave(os.path.abspath(grayscale_test_data_path+str(batch_index)+
                               "_"+str(image_index)+".png"), rgb_image_grayscale)
        imsave(os.path.abspath(colorized_test_data_path+str(batch_index)+
                               "_"+str(image_index)+".png"), rgb_image_colorized)
        imsave(os.path.abspath(ground_truth_test_data_path+str(batch_index)+
                               "_"+str(image_index)+".png"), rgb_image_true)

#Make predictions for unseen grayscale images (test data set).
def test_model(model, testing_datagen):
    error = 0
    num_testing_samples = len(os.listdir(testing_path+"testing/"))
    testing_batch_size = min(num_testing_samples, 100)
    num_testing_batches = int(np.ceil(num_testing_samples / testing_batch_size))
    batch_index = 0
    for batch in testing_datagen.flow_from_directory(os.path.abspath(testing_path),
                                                     target_size=(image_height, image_width),
                                                     batch_size=testing_batch_size,
                                                     color_mode="rgb",
                                                     class_mode=None,
                                                     shuffle=False):
        batch = color.rgb2lab(batch)
        X_batch = batch[:, :, :, 0]
        X_batch = X_batch[..., np.newaxis]
        Y_batch_predicted = model.predict(X_batch / 100) * 128
        Y_batch_true = batch[:, :, :, 1:]
        save_images_to_directory(X_batch, Y_batch_predicted,
                                 Y_batch_true, batch_index)
        error = error + model.evaluate(X_batch / 100, Y_batch_true / 128) * len(X_batch)
        if (batch_index == num_testing_batches - 1):
            mse = error / num_testing_samples
            print("mse (test data): " + str(mse))
            return
        batch_index = batch_index + 1


def main():
    #Get a new model, or load an existing one if possible.
    model, using_existing_model = get_model()
    #Train the model.
    training_datagen = get_image_generator("training")
    validation_datagen = get_image_generator("validation")
    model = train_model(model, using_existing_model,
                        training_datagen, validation_datagen)
    #Plot the training and validation loss for each training epoch.
    plot_loss()
    #Test the model.
    testing_datagen = get_image_generator("testing")
    test_model(model, testing_datagen)


if __name__ == '__main__':
    main()
