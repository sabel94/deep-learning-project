__author__ = "Johan Sabel, Felix Büttner, Joel Ekelöf"

from config import image_height, image_width, accuracy_epsilon
import tensorflow as tf
import keras.backend as K

def accuracy(y_true, y_predicted):
    num_images = K.shape(y_true)[0]
    num_images = tf.cast(num_images, tf.float32)
    num_pixels = num_images * (image_height * image_width)
    abs_distance = tf.abs(y_true - y_predicted)
    indicator = tf.where(tf.less_equal(abs_distance, tf.abs(y_true) * accuracy_epsilon),
                         tf.ones_like(abs_distance), abs_distance)
    indicator = tf.where(tf.greater(abs_distance, tf.abs(y_true) * accuracy_epsilon),
                         tf.zeros_like(indicator), indicator)
    product = tf.multiply(indicator[:,:,:,0], indicator[:,:,:,1])
    accuracy = tf.reduce_sum(product) / num_pixels
    return accuracy
