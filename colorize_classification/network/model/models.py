from keras.activations import softmax
from keras.layers import AActivation, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, Input
from keras.models import Model
from keras.optimizers import Adam

from network.model.accuracy import accuracy_probs_wrapper
from network.model.loss import weighted_categorical_crossentropy_wrapper


def get_model(network, params):
    """Get model/network architecture.

    Input:
        network:    The network type (standard, U-Net)
        params:     Needed parameters like image height and width

    Return:
        model:  Instance of Keras model class
    """
    if network == "standard":
        return get_model_standard(params)
    else:
        return get_model_unet(params)


def get_model_standard(params):
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

    # Define number of filters
    n_filters = 64

    # conv1-1
    conv1 = conv2d_block(1, inputs, n_filters*1, batch_norm=False)

    # conv1-2
    conv1 = conv2d_block(1, conv1, n_filters*1, strides=(2, 2))

    # conv2-1
    conv2 = conv2d_block(1, conv1, n_filters*2, batch_norm=False)

    # conv2-2
    conv2 = conv2d_block(1, conv2, n_filters*2, strides=(2, 2))

    # conv3-1
    conv3 = conv2d_block(1, conv2, n_filters*4, batch_norm=False)

    # conv3-2
    conv3 = conv2d_block(1, conv3, n_filters*4, strides=(2, 2))

    # conv4-1
    conv4 = conv2d_block(1, conv3, n_filters*8, batch_norm=False)

    # conv4-2
    conv4 = conv2d_block(1, conv4, n_filters*8)

    # conv5-1
    conv5 = conv2d_block(1, conv4, n_filters*8, dilation_rate=(2, 2), 
                         batch_norm=False)

    # conv5-2
    conv5 = conv2d_block(1, conv5, n_filters*8, dilation_rate=(2, 2))

    # conv6-1
    conv6 = conv2d_block(1, conv5, n_filters*8, dilation_rate=(2, 2), 
                         batch_norm=False)

    # conv6-2
    conv6 = conv2d_block(1, conv6, n_filters*8, dilation_rate=(2, 2))

    # conv7-1
    conv7 = conv2d_block(1, conv6, n_filters*8, batch_norm=False)

    # conv7-2
    conv7 = conv2d_block(1, conv7, n_filters*8)

    # conv8-1
    conv8 = conv2d_block(1, conv7, n_filters*4, strides=(2, 2), 
                         batch_norm=False)

    # conv8-2
    conv8 = Conv2DTranspose(filters=n_filters*4,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding="same",
                            dilation_rate=(1, 1),
                            kernel_initializer="he_normal")(conv8)
    conv8 = Activation("relu")(conv8)


    # (a, b) probability distribution
    outputs = Conv2D(filters=313,
                     kernel_size=(1, 1))(conv8)
    outputs = Activation(activation=softmax)(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def get_model_unet(params):
    """Define U-Net model/network architecture.

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

    # Define number of filters
    n_filters = 64

    # Contracting path
    conv1 = conv2d_block(1, inputs, n_filters*1)
    conv2 = conv2d_downsampling(conv1, n_filters*1)

    conv3 = conv2d_block(1, conv2, n_filters*2)
    conv4 = conv2d_downsampling(conv3, n_filters*2)

    conv5 = conv2d_block(1, conv4, n_filters*4)
    conv6 = conv2d_downsampling(conv5, n_filters*4)

    conv7 = conv2d_block(1, conv6, n_filters*8)
    conv8 = conv2d_block(2, conv7, n_filters*8, dilation_rate=(2, 2))
    conv9 = conv2d_block(1, conv8, n_filters*8)

    # Expansive path
    conv10 = conv2d_upsampling(conv9, n_filters*4)
    conv10 = Concatenate(axis=3)([conv10, conv5])
    conv11 = conv2d_block(1, conv10, n_filters*4)

    conv12 = conv2d_upsampling(conv11, n_filters*2)
    conv12 = Concatenate(axis=3)([conv12, conv3])
    conv13 = conv2d_block(1, conv12, n_filters*2)

    conv14 = conv2d_upsampling(conv13, n_filters*1)
    conv14 = Concatenate(axis=3)([conv14, conv1])
    conv15 = conv2d_block(1, conv14, n_filters*1, batch_norm=False)

    # (a, b) probability distribution
    outputs = Conv2D(filters=313,
                     kernel_size=(1, 1))(conv15)
    outputs = Activation(activation=softmax)(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def conv2d_block(n, conv, filters, strides=(1, 1), dilation_rate=(1, 1), 
                 batch_norm=True):
    """Repeat block of convolutional layer, activation layer and batch normalization layer (otional) n times.
    """
    for _ in range(n):
        conv = Conv2D(filters=filters,
                      kernel_size=(3, 3),
                      strides=strides,
                      padding="same",
                      dilation_rate=dilation_rate,
                      kernel_initializer="he_normal")(conv)
        conv = Activation("relu")(conv)
        if batch_norm:
            conv = BatchNormalization(axis=3)(conv)

    return conv


def conv2d_downsampling(conv, filters, strides=(2, 2)):
    """Define block for downsampling."""
    conv = Conv2D(filters=filters,
                  kernel_size=(3, 3),
                  strides=strides,
                  padding="same",
                  dilation_rate=(1, 1),
                  kernel_initializer="he_normal")(conv)

    return conv


def conv2d_upsampling(conv, filters):
    """Define block for upsampling."""
    conv = Conv2DTranspose(filters=filters,
                            kernel_size=(3, 3),
                            strides=(2, 2),
                            padding="same",
                            dilation_rate=(1, 1),
                            kernel_initializer="he_normal")(conv)

    return conv


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

    # Set accuracy function
    threshold = params['accuracy_threshold']
    accuracy = accuracy_probs_wrapper(threshold)
    
    # Compile model
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[accuracy]
                  )

    return model
