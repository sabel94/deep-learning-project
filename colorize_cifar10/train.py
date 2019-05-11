__author__ = "Johan Sabel, Felix Büttner, Joel Ekelöf"


from config import (image_height, image_width, model_output_path,
                    batch_size, num_epochs, learning_rate)
from data_generator import get_samples
from helpers import save_history
from keras.optimizers import Adam
import os
import numpy as np


#Train the network.
def train_model(model, using_existing_model,
                training_images, validation_images,
                training_datagen, validation_datagen):
    if not using_existing_model:
        optimizer = Adam(lr=learning_rate)
        model.compile(loss="mse", optimizer=optimizer)
    num_training_samples = training_images.shape[0]
    num_steps_per_epoch = int(np.ceil(num_training_samples / batch_size))
    num_validation_samples = validation_images.shape[0]
    num_val_steps_per_epoch = int(np.ceil(num_validation_samples / batch_size))
    history = model.fit_generator(generator=get_samples(training_datagen, training_images, True),
                                  steps_per_epoch=num_steps_per_epoch,
                                  epochs=num_epochs,
                                  validation_data=get_samples(validation_datagen, validation_images, False),
                                  validation_steps=num_val_steps_per_epoch)
    #Save the model (weights etc.).
    model.save(model_output_path+"model.h5")
    #Save the training history (loss, accuracy etc. for each epoch).
    save_history(history, using_existing_model)
    return model
