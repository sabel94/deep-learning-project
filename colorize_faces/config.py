__author__ = "Johan Sabel, Felix Büttner, Joel Ekelöf"


#Image dimensions.
image_height = 256
image_width = 256
#Input paths.
model_input_path = "drive/My Drive/colorize_faces/input/model/"
history_input_path = "drive/My Drive/colorize_faces/input/history/"
training_path = "drive/My Drive/colorize_faces/input/datasets/training/"
validation_path = "drive/My Drive/colorize_faces/input/datasets/validation/"
testing_path = "drive/My Drive/colorize_faces/input/datasets/testing/"
#Output paths.
model_output_path = "drive/My Drive/colorize_faces/output/model/"
history_output_path = "drive/My Drive/colorize_faces/output/history/"
loss_plot_path = "drive/My Drive/colorize_faces/output/loss_plot/"
ground_truth_test_data_path = "drive/My Drive/colorize_faces/output/ground_truth_test_data/"
grayscale_test_data_path = "drive/My Drive/colorize_faces/output/grayscale_test_data/"
colorized_test_data_path = "drive/My Drive/colorize_faces/output/colorized_test_data/"
#Parameters.
batch_size = 32
num_epochs = 1
learning_rate = 0.0015
l2_regularization_lambda = 0.0005
dropout_rate = 0.1
early_stopping_patience = 20
#Settings.
accuracy_epsilon = 0.02 #Percentage threshold.
using_l2_regularization = False
using_dropout = False
using_early_stopping = False
