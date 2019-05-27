__author__ = "Johan Sabel, Felix Büttner, Joel Ekelöf"


from config import (image_height, image_width, history_input_path,
                    history_output_path, loss_plot_path,
                    ground_truth_test_data_path, grayscale_test_data_path,
                    colorized_test_data_path, batch_size)
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from skimage import color
from skimage.io import imsave


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
