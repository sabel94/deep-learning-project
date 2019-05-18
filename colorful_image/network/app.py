import os
import numpy as np
from keras.models import load_model

from network.config import config
from network.data import data
from network.model import models
from network.model.loss import weighted_categorical_crossentropy_wrapper
from network.utils import helpers


def main():
    # Get configurations/settings
    c_data = get_configuration('data')
    c_training = get_configuration('training')
    c_testing = get_configuration('testing')

    """Prepare data."""
    # The file pts_in_hull contains the ab values for Q=313 bins
    pts_in_hull = np.load(c_data['dir']['pts_in_hull']) # (Q=313, 2)
    
    # Update dict with an entry for Q
    c_data.update({"Q": pts_in_hull.shape[0]}) # Q=313

    # Fit unsupervised nearest neighbors learner to pts_in_hull
    nn_learner = data.nearest_neighbors(pts_in_hull, c_data)

    # Train network
    #train([c_data, c_training], nn_learner)

    # Test network
    test([c_data, c_training, c_testing], nn_learner, pts_in_hull)


def train(params, nn_learner):
    """Train network.

    Input:
        params:     Needed parameters stored in [c_data, c_training]
        nn_learner: Fitted unsupervised nearest neighbors model
    """
    # Parameters
    c_data = params[0]
    c_training = params[1]

    """Continue with data preparation."""
    # Get data generator
    datagen_training = data.data_generator('training', nn_learner, params)
    datagen_validation = data.data_generator('validation', nn_learner, params)

    """Define Model."""
    # Get model
    model = models.get_model(c_data)

    # The file prior_probs contains the general probability of occurence for 
    # each of the Q ab bins. Note that the probabilities are already smoothed 
    # with a Gaussian kernel.
    prior_probs = np.load(c_data['dir']['prior_probs']) # (Q=313,)

    # Compile model
    model = models.compile_model(model, prior_probs, c_training)

    # Show summary representation of the model
    model.summary()

    """Train network."""
    # Load model weights if specified
    dir_model_weights = c_data['dir']['model_weights']
    if c_training['load_weights'] == 1 and os.path.isfile(dir_model_weights):
        print("Load model weights.")
        model.load_weights(dir_model_weights)

    # Get batch size
    batch_size = c_training['batch_size']

    # Get number of epochs
    n_epochs = c_training['n_epochs']

    # Calculate needed steps per epoch
    dir_training_data = c_data['dir']['training_data'] + "/images/"
    n_training_samples = len(os.listdir(dir_training_data))
    steps_per_epoch = np.ceil(n_training_samples / batch_size)

    # Calculate needed validation steps
    dir_validation_data = c_data['dir']['validation_data'] + "/images/"
    n_validation_samples = len(os.listdir(dir_training_data))
    n_validation_steps = np.ceil(n_validation_samples / batch_size)

    # Train model
    dir_model = str(c_data['dir']['model'])
    try:
        model.fit_generator(generator=datagen_training,
                            steps_per_epoch=steps_per_epoch,
                            epochs=n_epochs,
                            validation_data=datagen_validation,
                            validation_steps=n_validation_steps,
                            callbacks=None)

        model.save(dir_model)
        model.save_weights(dir_model_weights)

    except KeyboardInterrupt:
        model.save(dir_model)

    print("Success: Training completed")


def test(params, nn_learner, pts_in_hull):
    """Test network.
    
    Input:
        params:         Needed parameters stored in [c_data, c_training,
                                                     c_testing]
        nn_learner:     Fitted unsupervised nearest neighbors model
        pts_in_hull:    Contains the ab values for Q=313 bins, shape (Q=313, 2)
    """
    # Parameters
    c_data = params[0]
    c_training = params[1]
    c_testing = params[2]

    """Continue with data preparation."""
    # Get data generator
    datagen_testing = data.data_generator('testing', nn_learner, params)

    """Load Model."""
    dir_model = str(c_data['dir']['model'])
    prior_probs = np.load(c_data['dir']['prior_probs']) # (Q=313,)
    wccw = weighted_categorical_crossentropy_wrapper(prior_probs, c_training)
    co = {'weighted_categorical_crossentropy': wccw}
    model = load_model(dir_model, custom_objects=co)

    # Loop through batches
    dir_output = c_data['dir']['output']
    index = 0
    for batch in datagen_testing:
        # Extract batches
        batch_input = batch[0]      # Normalized L-channel, range [0, 1]
        t_batch_probs = batch[1]    # Ground truth

        # Predict distributions
        p_batch_probs = model.predict(batch_input) # (batch_size, Ht, Wt, Q)
        
        # Map the predicted distributions to point estimates in ab space
        batch_color_ab = data.z2y(p_batch_probs, 
                                  pts_in_hull, 
                                  [c_data, c_testing]
                                  ) # (batch_size, H, W, 2)

        # Denormalize input from tange [0, 1] to [0, 100] (L-channel)
        batch_input = batch_input*100       
        
        # Concatenate L-channel with the ab channel
        images_lab = np.concatenate((batch_input, batch_color_ab), axis=3)

        # Save images
        helpers.save_images(images_lab, dir_output, index)
        
        # Break loop when the number of total batches is reached
        n_batches = batch[2]
        if index == n_batches - 1:
            return
        else:
            index = index + 1
        
    print("Success: Testing completed")


def get_configuration(category):
    """Get all configurations for one category defined in config.json.

    Input:
        category: One of the first level entries in the json file 

    Return:
        c:  All configurations saved in a dict
    """
    c = config.var.data[category]

    return c

