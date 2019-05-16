from keras.activations import softmax
from keras.layers import Activation, BatchNormalization, Conv2D, Conv2DTranspose, Input
from keras.models import Model
from keras.optimizers import Adam

from network.model.loss import weighted_categorical_crossentropy_wrapper


def get_model(params):
    """Define model/network architecture.

    Input:
        params: Needed parameters like image height and width

    Return:
        model:  Instance of Keras model class
    """
    # The input has shape (image height, image width, channel = 1)
    # The channel = 1, since only the L channel is of interest
    H = params['input']['height']
    W = params['input']['width']
    input_shape = (H, W, 1)
    inputs = Input(input_shape)

    # conv1-1
    conv1 = Conv2D(filters=64,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   padding="same",
                   dilation_rate=(1, 1),
                   kernel_initializer="he_normal")(inputs)
    conv1 = Activation("relu")(conv1)

    # conv1-2
    conv1 = Conv2D(filters=64, 
                   kernel_size=(3, 3),
                   strides=(2, 2),
                   padding="same", 
                   dilation_rate=(1, 1),
                   kernel_initializer="he_normal")(conv1)
    conv1 = Activation("relu")(conv1)
    conv1 = BatchNormalization(axis=3)(conv1)

    # conv2-1
    conv2 = Conv2D(filters=128, 
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   padding="same",
                   dilation_rate=(1, 1), 
                   kernel_initializer="he_normal")(conv1)
    conv2 = Activation("relu")(conv2)

    # conv2-2
    conv2 = Conv2D(filters=128, 
                   kernel_size=(3, 3),
                   strides=(2, 2),
                   padding="same", 
                   dilation_rate=(1, 1),
                   kernel_initializer="he_normal")(conv2)
    conv2 = Activation("relu")(conv2)
    conv2 = BatchNormalization(axis=3)(conv2)

    # conv3-1
    conv3 = Conv2D(filters=256, 
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   padding="same",
                   dilation_rate=(1, 1), 
                   kernel_initializer="he_normal")(conv2)
    conv3 = Activation("relu")(conv3)

    # conv3-2
    conv3 = Conv2D(filters=256, 
                   kernel_size=(3, 3),
                   strides=(2, 2),
                   padding="same", 
                   dilation_rate=(1, 1),
                   kernel_initializer="he_normal")(conv3)
    conv3 = Activation("relu")(conv3)
    conv3 = BatchNormalization(axis=3)(conv3)

    # conv4-1
    conv4 = Conv2D(filters=512, 
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   padding="same", 
                   dilation_rate=(1, 1),
                   kernel_initializer="he_normal")(conv3)
    conv4 = Activation("relu")(conv4)

    # conv4-2
    conv4 = Conv2D(filters=512, 
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   padding="same", 
                   dilation_rate=(1, 1),
                   kernel_initializer="he_normal")(conv4)
    conv4 = Activation("relu")(conv4)
    conv4 = BatchNormalization(axis=3)(conv4)

    # conv5-1
    conv5 = Conv2D(filters=512, 
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   padding="same",
                   dilation_rate=(2, 2), 
                   kernel_initializer="he_normal")(conv4)
    conv5 = Activation("relu")(conv5)

    # conv5-2
    conv5 = Conv2D(filters=512, 
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   padding="same",
                   dilation_rate=(2, 2), 
                   kernel_initializer="he_normal")(conv5)
    conv5 = Activation("relu")(conv5)
    conv5 = BatchNormalization(axis=3)(conv5)

    # conv6-1
    conv6 = Conv2D(filters=512, 
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   padding="same",
                   dilation_rate=(2, 2), 
                   kernel_initializer="he_normal")(conv5)
    conv6 = Activation("relu")(conv6)

    # conv6-2
    conv6 = Conv2D(filters=512, 
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   padding="same",
                   dilation_rate=(2, 2), 
                   kernel_initializer="he_normal")(conv6)
    conv6 = Activation("relu")(conv6)
    conv6 = BatchNormalization(axis=3)(conv6)

    # conv7-1
    conv7 = Conv2D(filters=512, 
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   padding="same",
                   dilation_rate=(1, 1), 
                   kernel_initializer="he_normal")(conv6)
    conv7 = Activation("relu")(conv7)

    # conv7-2
    conv7 = Conv2D(filters=512, 
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   padding="same",
                   dilation_rate=(1, 1), 
                   kernel_initializer="he_normal")(conv7)
    conv7 = Activation("relu")(conv7)
    conv7 = BatchNormalization(axis=3)(conv7)

    # conv8-1
    conv8 = Conv2DTranspose(filters=256, 
                            kernel_size=(3, 3), #(4, 4)
                            strides=(2, 2),
                            padding="same",
                            dilation_rate=(1, 1), 
                            kernel_initializer="he_normal")(conv7)
    conv8 = Activation("relu")(conv8)

    # conv8-2
    conv8 = Conv2DTranspose(filters=256, 
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding="same",
                            dilation_rate=(1, 1), 
                            kernel_initializer="he_normal")(conv8)
    conv8 = Activation("relu")(conv8)


    # (a, b) probability distribution
    outputs = Conv2D(313, (1, 1))(conv8)
    outputs = Activation(activation=softmax)(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def compile_model(model, prior_probs, params):
    """Configure the model for training.

    Input:
        model:          Instance of Keras model class
        prior_probs:    Smoothed empirical distribution of colors
        params:         Needed parameters like lambda

    Return:
        model:  Instance of Keras model class
    """
    # Set optimizer
    learning_rate = params['learning_rate']
    optimizer = Adam(lr=learning_rate)
    
    # Set loss function
    loss = weighted_categorical_crossentropy_wrapper(prior_probs, params)

    # Compile model
    model.compile(optimizer=optimizer,
                  loss=loss 
                  )

    return model
