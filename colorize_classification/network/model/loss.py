from keras import backend as K
import tensorflow as tf


def get_weights(y_true, prior_probs, params):
    """Calculate all weights v.

    Input:
        y_true:         Converted ground truth color Z = H_gt(Y)^(-1) with
                        shape (b, H, W, Q)
        prior_probs:    Smoothed empirical distribution of colors
        params:         Needed parameters like lambda

    Return:
        all_v:  All weigths v saved in all_v with shape (b, H, W)

    Info: The comment next to each calculation gives its shape where
        Q: number of bins
        b: batch size
        H: image height
        W: image width
    """
    # Parameters
    _lambda = params['lambda']
    Q = prior_probs.shape[0]

    # The weights are proportional to
    all_w = ((1 -_lambda)*prior_probs + _lambda/Q)**(-1) # (Q,)

    # The weighted distribution must sum to one: E[w] = sum(p_tilde*w) = 1
    all_w = all_w / tf.reduce_sum(prior_probs * all_w) # (Q,)

    # Find q_star
    q_star = tf.argmax(y_true, axis=3) # (b, H, W)

    # Select weights
    all_v = tf.gather(all_w, q_star)  # (b, H, W)

    # Cast to float32, which is necessary for further calculations
    all_v = tf.cast(all_v, tf.float32) # (b, H, W)

    return all_v


def weighted_categorical_crossentropy_wrapper(prior_probs, params):
    """Wrapper for weighted_categorical_crossentropy (defined below) to pass some inputs.

    Input:
        prior_probs:    Smoothed empirical distribution of colors
        params:         Needed parameters like lambda

    Return:
        weighted_cross_entropy: Weighted categorical cross entropy (value)

    Info: The comment next to each calculation gives its shape where
        Q: number of bins
        b: batch size
        H: image height
        W: image width
    """

    def weighted_categorical_crossentropy(y_true, y_pred):
        """Calculates the weighted categorical cross entropy.

        Input:
            y_true: Converted ground truth color Z = H_gt(Y)^(-1) with
                    shape (b, H, W, Q)
            y_pred: Prediction Z_hat with shape (b, H, W, Q)

        Return:
            weighted_cross_entropy: Weighted categorical cross entropy (value)

        Note that the following code is based on the function
        categorical_crossentropy defined in keras/keras/backend/cntk_backend.py
        (see github: keras-team/keras).
        """
        # Scale preds so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True) # (b, H, W, Q)

        # Avoid numerical instability with epsilon clipping
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon()) # (b, H, W, Q)

        # Calculate categorical cross entropy
        cross_entropy = tf.multiply(y_true, K.log(y_pred)) # (b, H, W, Q)
        cross_entropy = tf.reduce_sum(cross_entropy, axis=3) # (b, H, W)

        # Get weights
        weights = get_weights(y_true, prior_probs, params) # (b, H, W)

        # Calculate weighted categorical cross entropy
        weighted_cross_entropy = tf.multiply(weights, cross_entropy) #(b, H, W)
        #weighted_cross_entropy = tf.reduce_sum(weighted_cross_entropy) # ()
        weighted_cross_entropy = - weighted_cross_entropy # ()

        return weighted_cross_entropy

    return weighted_categorical_crossentropy
