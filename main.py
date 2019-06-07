import os
import json
import logging
import numpy as np
from utils.config import get_config_from_json
from data_loader.data_loader import DataLoader
from models.dense_model import DenseModel
from trainer.trainer import Trainer
from os import listdir
from os.path import isfile, join
import tensorflow as tf
from tensorflow.data import Dataset

# GLOBAL VARIABLES
mypath = "D:\\Dropbox\\9. Data\\Mercury Data\\CSV"


def main():
    # Processing config file
    config = get_config_from_json('.\\utils\\config.json')

    # Processing data
    train_dataset, test_data_set, num_train_features, num_train_samples, num_test_samples = getData(mypath, config)

    # Creating an empty model
    dense_model = DenseModel(num_train_features, config)
    load_flag = config.experiment.load

    # load model from h5 file
    if load_flag == True:
        print('Loading saved model')
        dense_model.load(".\saved_models\\Mercury 2.h5")
        results = dense_model.model.evaluate(test_data_set,steps=int(num_test_samples/(config.model.batch_size)))
        print('test loss, test acc:', results)

    # build and train and save a new model
    elif load_flag == False:
        dense_model.build_model()

        print('Create the trainer')
        trainer = Trainer(dense_model.model, train_dataset, config, steps_per_epoch = int(num_train_samples/config.model.batch_size))
        print('Start training the model.')
        trainer.train()
        dense_model.save(".\saved_models\\Mercury 2.h5")
    else:
        print("Invalid load flag in config file")

    logging.info('---------Successful execution---------')


def getData(mypath, config):

    # get list of filepaths
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    data_dict = [mypath + "\\" + s for s in onlyfiles]

    # create numpy datasets for each stock
    data = []
    for fname in data_dict:
        data.append(DataLoader(fname,
                               window=config.experiment.window,
                               threshold=config.experiment.threshold,
                               featselect=config.experiment.featselect,
                               drop=config.experiment.drop))

    # initialize numpy arrays for training and test data
    X_train = data[0].X_train_std
    Y_train = data[0].Y_train
    X_test = data[0].X_test_std
    Y_test = data[0].Y_test

    # add other stocks to previously initialized numpy arrays
    for i in range(1, len(data)):
        X_train = np.concatenate((X_train, data[i].X_train_std), axis=0)
        Y_train = np.concatenate((Y_train, data[i].Y_train), axis=0)
        X_test = np.concatenate((X_test, data[i].X_test_std), axis=0)
        Y_test = np.concatenate((Y_test, data[i].Y_test), axis=0)

    # Save number of features and samples
    num_train_samples = X_train.shape[0]
    num_test_samples = X_test.shape[0]
    num_train_features = X_train.shape[1]

    # Generate TF dataset for Keras model
    logging.info('------Final Training and Test Datasets------)
    logging.info('Size of X_Train: %s', X_train.shape)
    logging.info('Size of Y_Train: %s', Y_train.shape)
    logging.info('Size of X_Test: %s', X_test.shape)
    logging.info('Size of Y_Test: %s', Y_test.shape)
    train_dataset = Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = train_dataset.shuffle(config.model.shuffle).batch(config.model.batch_size).repeat()
    test_dataset = Dataset.from_tensor_slices((X_test, Y_test))
    test_dataset = test_dataset.shuffle(config.model.shuffle).batch(config.model.batch_size).repeat()

    return train_dataset, test_dataset, num_train_features, num_train_samples, num_test_samples

if __name__ == '__main__':
    main()
    os.system("tensorboard --logdir=.\\logs\\")
