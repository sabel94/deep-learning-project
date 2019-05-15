__author__ = "Johan Sabel, Felix Büttner, Joel Ekelöf"


#Image dimensions.
image_height = 256
image_width = 256
#Input paths.
model_input_path = "input/model/"
history_input_path = "input/history/"
training_path = "input/datasets/training/"
validation_path = "input/datasets/validation/"
testing_path = "input/datasets/testing/"
#Output paths.
model_output_path = "output/model/"
history_output_path = "output/history/"
loss_plot_path = "output/loss_plot/"
ground_truth_test_data_path = "output/ground_truth_test_data/"
grayscale_test_data_path = "output/grayscale_test_data/"
colorized_test_data_path = "output/colorized_test_data/"
#Parameters.
batch_size = 32
num_epochs = 100
learning_rate = 0.0015
l2_regularization_lambda = 0.0005
dropout_rate = 0.1
early_stopping_patience = 20
#Settings.
using_l2_regularization = False
using_dropout = False
using_early_stopping = False
