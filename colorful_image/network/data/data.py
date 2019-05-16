import os
import sys
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np
from skimage import color, transform, io
from sklearn.neighbors import NearestNeighbors


def data_generator(dataset, nn_learner, params):
    """Generate batches of tensor image data with real-time data augmentation. The data will be looped over (in batches).

    Input:
        dataset:    Defines the purpose of the generator (training, 
                    validation, or testing)
        nn_learner: Fitted unsupervised nearest neighbors model
        params:     Needed parameters stored like [c_data, c_training]

    Return:
        batch_input, batch_truth:   Input and ground truth images packed in 
                                    at least one batch
    """
    # Define generator and settings according to the specified data set
    if dataset == "training":
        datagen = ImageDataGenerator(rescale=1./255,
                                     rotation_range=20,
                                     shear_range=0.3,
                                     zoom_range=0.3,
                                     horizontal_flip=True
                                     )
        dir_data = os.path.abspath(params[0]['dir']['training_data'])
        shuffle = True
    elif dataset == "validation":
        datagen = ImageDataGenerator(rescale=1./255)
        dir_data = os.path.abspath(params[0]['dir']['validation_data'])
        shuffle = False
    else:
        datagen = ImageDataGenerator(rescale=1./255)
        dir_data = os.path.abspath(params[0]['dir']['testing_data'])
        shuffle = False

    # Check for availability of images
    n_images = len(os.listdir(dir_data + "/images/"))
    if n_images == 0:
        msg = "No images found in " + dir_data + "/images/"
        raise OSError(msg)

    # Get image height and width needed for the input of the network
    H = params[0]['input']['height']
    W = params[0]['input']['width']

    # Get batch size
    batch_size = params[1]['batch_size']

    # Define flow, which takes the path to a directory and generates batches
    flow = datagen.flow_from_directory(dir_data,
                                       target_size=(H, W),
                                       batch_size=batch_size,
                                       color_mode="rgb",
                                       class_mode=None,
                                       shuffle=shuffle)
    # Loop through each batch
    for batch_rgb in flow:
        # Convert RGB to LAB
        batch_lab = color.rgb2lab(batch_rgb)  # (batch_size, H, W, 3)

        # Extract the L-channel.
        batch_l = batch_lab[:, :, :, 0, np.newaxis]  # (batch_size, H, W, 1)

        # Normalize space from [0, 100] to [0, 1]
        batch_l_norm = batch_l / 100 # (batch_size, H, W, 1)

        # Extract the a and b channel
        batch_ab = batch_lab[:, :, :, 1:]  # (batch_size, H, W, 2)

        # Get probability distribution over possible colors
        # Note: z in [0, 1] where z is an element of batch_probs
        batch_probs = y2z(batch_ab, nn_learner, params[0]) # (batch_size, 
                                                           #  Ht, Wt, Q)
        # Define input batch
        batch_input = batch_l_norm

        # Define ground truth batch
        batch_truth = batch_probs

        yield (batch_input, batch_truth)


def y2z(batch_ab, nn_learner, params):
    """Calculates the probability distribution over possible colors (the Q ab 
    bins) for all pixels.

    Input:
        batch_ab:   The a and b channel of the ground truth images in the 
                    batch, shape (batch_size, H, W, 2)
        nn_learner: Fitted unsupervised nearest neighbors model
        params:     Needed parameters like image height and width of the ground
                    truth image

    Output:
        z_values:   Probability distribution over possible colors (the Q ab 
                    bins) for all pixels
    """
    # Parameters
    Ht = params['truth']['height'] # height of the ground truth image
    Wt = params['truth']['width']  # width of the ground truth image
    Q = params['Q']
    sigma = params['sigma']

    # Initialization
    z_values = list()

    # Loop through all images
    for image in batch_ab: # image has shape (H, W, 2)
        # Resize image to (Ht, Wt, 2) if necessary
        if image.shape[0] != Ht or image.shape[1] != Wt:
            image = transform.resize(image,
                                     (Ht, Wt),
                                     mode='constant',
                                     anti_aliasing=True
                                     )
        # Flatten image
        image = image.reshape(-1, 2)  # (Ht*Wt, 2)

        # Find the K closest ab bins (neighbors) for each pixel where 
        # K = n_neighbors
        distances, indices = nn_learner.kneighbors(image)  # (Ht*Wt, 
                                                           #  n_neighbors)

        # Use a Gaussian kernel to assign a high (low) probability p to ab 
        # bins (neighbors) with a small (large) distance to the respective 
        # pixel, p in [0, 1]
        pseudo_probas = np.exp(-distances**2./(2.*sigma**2.))  # (Ht*Wt, 
                                                               #  n_neighbors)

        # Normalize probabilities to express which ab bin (neighbor) is the 
        # most probable one for each pixel
        probas = pseudo_probas / np.sum(pseudo_probas, axis=1)[:, np.newaxis]

        # Initialization: Assign a probability of zero to all Q ab bins
        # for all pixels
        z = np.zeros((Ht * Wt, Q)) # (Ht*Wt, Q)

        # Only update the probability of occurence for the ab bins which were 
        # neighbors to one of the pixels 
        z[np.arange(0, Ht * Wt, dtype='int')[:, np.newaxis], indices] = probas

        # Reshape the matrix to match the network architecture
        z = z.reshape(Ht, Wt, Q) # (Ht, Wt, Q)
        
        # Save matrix
        z_values.append(z)

    # Convert to numpy array
    z_values = np.array(z_values) # (batch_size, Ht, Wt, Q)

    return z_values


def nearest_neighbors(samples, params):
    """Fit unsupervised nearest neighbors learner to training data (samples).

    Input:
        samples:    Training data
        params:     Needed parameters like number of neighbors

    Return:
        nn_learner: Fitted model
    """
    nn_learner = NearestNeighbors(n_neighbors=params['n_neighbors'],
                                  algorithm='ball_tree'
                                  )
    nn_learner.fit(samples)
    return nn_learner

