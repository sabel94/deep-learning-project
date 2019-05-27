import numpy as np
from keras import backend as K
import tensorflow as tf


def accuracy_ab(y_true, y_pred, threshold):
    """The following accuracy metric gives the ratio between correctly
    predicted pixels and the total number of pixels. A pixel is considered as
    correctly predicted when the distance of its a and b value to the
    respective true value is within a threshold.

    Input:
        y_true:     True ab channels, shape (batch_size, H, W, 2)
        y_pred:     Predicted ab channels, shape (batch_size, H, W, 2)
        threshold:  Threshold value

    Return:
        accuracy:   The accuracy value
    """
    # Initialization
    m = np.zeros(y_true.shape)

    # If the distance between the true and the predicted
    # value is within the threshold, then account the prediction
    # as correct
    s = np.abs(np.subtract(y_true, y_pred))
    m[s <= threshold] = 1.

    # A prediction of a pixel is correct when both predictions for
    # the a and b channels are correct, otherwise not
    a_channel = m[:,:,:,0]
    b_channel = m[:,:,:,1]
    p = np.multiply(a_channel, b_channel) # (batch_size, H, W)

    # The mean value gives the percentage of correctly predicted pixels
    # to the total number of pixels
    accuracy = np.mean(p)

    return accuracy


def accuracy_probs_wrapper(threshold):
    """Wrapper around accuracy_probs to pass some inputs.

    Input:
        threshold:  Threshold value

    Return:
        accuracy_probs:   The accuracy value
    """

    def accuracy_probs(y_true, y_pred):
        """The following accuracy metric gives the ratio between correctly
        predicted pixels and the total number of pixels. A pixel is considered
        as correctly predicted when the difference between the predicted and
        the true probability for each ab bin is within a threshold.

        Input:
            y_true:     True probability distribution,
                        shape (batch_size, Ht, Wt, Q)
            y_pred:     Predicted probability distribution,
                        shape (batch_size, Ht, Wt, Q)
            threshold:  Threshold value

        Return:
            accuracy:   The accuracy value
        """
        # Initialization
        m = tf.zeros(K.shape(y_true))

        # If the difference between the true and the predicted probability
        # distributin is within the threshold, then account the prediction
        # as correct
        s = tf.abs(y_true - y_pred)
        m = tf.where(tf.less_equal(s, threshold), tf.ones_like(s), m)

        # A prediction of a pixel is correct when all predictions for
        # the Q ab bins are correct, otherwise not
        a_channel = m[:,:,:,0]
        b_channel = m[:,:,:,1]
        p = tf.multiply(a_channel, b_channel) # (batch_size, Ht, Wt)

        # The mean value gives the percentage of correctly predicted pixels
        # to the total number of pixels
        accuracy = tf.reduce_mean(p)

        return accuracy

    return accuracy_probs
