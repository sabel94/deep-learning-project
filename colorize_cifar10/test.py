__author__ = "Johan Sabel, Felix Büttner, Joel Ekelöf"


from config import image_height, image_width, batch_size
from helpers import save_images_to_directory
import os
import numpy as np
from skimage import color


#Make predictions for unseen grayscale images (test data set).
def test_model(model, images, datagen):
    error = 0
    num_testing_samples = images.shape[0]
    num_testing_batches = int(np.ceil(num_testing_samples / batch_size))
    batch_index = 0
    for batch in datagen.flow(images, batch_size=batch_size, shuffle=False):
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
            print("MSE (test data): " + str(mse))
            return
        batch_index = batch_index + 1
