import os
import numpy as np

from network.config import config
from network.data import data
from network.model import models, loss


def main():
    """Train model. Evaluation is following. :)"""
    # Get configurations/settings
    c_data = get_configuration('data')
    c_training = get_configuration('training')

    """Prepare data."""
    # The file pts_in_hull contains the ab values for Q=313 bins
    pts_in_hull = np.load(c_data['dir']['pts_in_hull']) # (Q=313, 2)
    
    # Update dict with an entry for Q
    c_data.update({"Q": pts_in_hull.shape[0]}) # Q=313

    # Fit unsupervised nearest neighbors learner to pts_in_hull
    nn_learner = data.nearest_neighbors(pts_in_hull, c_data)

    # Get data generator
    params = [c_data, c_training]
    datagen_training = data.data_generator('training', nn_learner, params)
    datagen_validation = data.data_generator('validation', nn_learner, params)

    """Define Model."""
    # Get model
    model = models.get_model(c_data)

    # The file prior_probs contains the general probability of occurence for 
    # each of the Q ab bins
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


def get_configuration(category):
    """Get all configurations for one category defined in config.json.

    Input:
        category: One of the first level entries in the json file 

    Return:
        c:  All configurations saved in a dict
    """
    c = config.var.data[category]

    return c
