__author__ = "Johan Sabel, Felix Büttner, Joel Ekelöf"


from model import get_model
from data_generator import get_image_generator
from train import train_model
from helpers import plot_loss
from test import test_model


def main():
    #Get a new model, or load an existing one if possible.
    model, using_existing_model = get_model()
    #Train the model.
    training_datagen = get_image_generator("training")
    validation_datagen = get_image_generator("validation")
    model = train_model(model, using_existing_model,
                        training_datagen, validation_datagen)
    #Plot the training and validation loss for each training epoch.
    plot_loss()
    #Test the model.
    testing_datagen = get_image_generator("testing")
    test_model(model, testing_datagen)


if __name__ == '__main__':
    main()
