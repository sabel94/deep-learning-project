__author__ = "Johan Sabel, Felix Büttner, Joel Ekelöf"


from config import image_height, image_width, testing_path, batch_size
from helpers import save_images_to_directory
import os
import numpy as np
from skimage import color


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
        error = error + model.evaluate(X_batch / 100, Y_batch_true / 128)[0] * len(X_batch)
        if (batch_index == num_testing_batches - 1):
            mse = error / num_testing_samples
            print("mse (test data): " + str(mse))
            return
        batch_index = batch_index + 1
