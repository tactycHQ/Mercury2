import os
import numpy as np
import logging
from utils.config import get_config_from_json
from data_loader.data_loader import DataLoader
from models.dense_model import DenseModel
from trainer.trainer import Trainer
from os import listdir
from os.path import isfile, join
from sklearn.svm import SVC

# GLOBAL VARIABLES
mypath = "D:\\Dropbox\\9. Data\\Mercury Data\\CSV"


def main():
    # Processing config file
    config = get_config_from_json('.\\utils\\config.json')

    # Processing data
    X_train, Y_train, X_val, Y_val, X_test, Y_test, num_train_features, num_train_samples, num_val_samples, num_test_samples = getData(mypath, config)

    svm = SVC(kernel='linear',verbose=1)
    print("SVC started")
    svm.fit(X_train,Y_train)
    print("fit complete")
    answer = svm.score(X_test,Y_test)
    print("answer:/n",answer)
    print("Model completed")



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
    tar = np.array([-1,0,1])

    X_train = data[0].X_train_std
    Y_train = data[0].Y_train
    Y_train=np.sum(Y_train*tar,axis=1)
    X_val = data[0].X_val_std
    Y_val = data[0].Y_val[:X_val.shape[0]]
    Y_val = np.sum(Y_val*tar, axis=1)
    X_test = data[0].X_test_std
    Y_test = data[0].Y_test[:X_test.shape[0]]
    Y_test = np.sum(Y_test*tar, axis=1)

    # add other stocks to previously initialized numpy arrays
    for i in range(1, len(data)):
        X_train = np.concatenate((X_train, data[i].X_train_std), axis=0)
        Y_train = np.concatenate((Y_train, np.sum(data[i].Y_train*tar,axis=1)), axis=0)
        X_val = np.concatenate((X_val, data[i].X_val_std), axis=0)
        Y_val = np.concatenate((Y_val, np.sum(data[i].Y_val*tar,axis=1)), axis=0)
        X_test = np.concatenate((X_test, data[i].X_test_std), axis=0)
        Y_test = np.concatenate((Y_test, np.sum(data[i].Y_test*tar,axis=1)), axis=0)

    # Save number of features and samples
    num_train_samples = X_train.shape[0]
    num_val_samples = X_val.shape[0]
    num_test_samples = X_test.shape[0]
    num_train_features = X_train.shape[1]

    # Generate TF dataset for Keras model
    logging.info('------Final Training and Test Datasets------')
    logging.info('Size of X_Train: %s', X_train.shape)
    logging.info('Size of Y_Train: %s', Y_train.shape)
    logging.info('Size of X_val: %s', X_val.shape)
    logging.info('Size of Y_val: %s', Y_val.shape)
    logging.info('Size of X_Test: %s', X_test.shape)
    logging.info('Size of Y_Test: %s', Y_test.shape)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, num_train_features, num_train_samples, num_val_samples, num_test_samples

if __name__ == '__main__':
    main()

