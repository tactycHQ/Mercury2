import os
import logging
import numpy as np
from data_loader.data_loader import DataLoader
from models.dense_model import DenseModel
from trainer.trainer import Trainer
from os import listdir
from os.path import isfile, join


#GLOBAL VARIABLES
mypath = "D:\\Dropbox\\9. Data\\Mercury Data\\CSV"


def main(data,load=0):

    X_train = data[0].X_train_std
    Y_train = data[0].Y_train
    X_test = data[0].X_test_std
    Y_test = data[0].Y_test

    for i in range(1,len(data)):
        X_train = np.concatenate((X_train,data[i].X_train_std), axis=0)
        Y_train = np.concatenate((Y_train,data[i].Y_train), axis = 0)
        X_test = np.concatenate((X_test,data[i].X_test_std), axis = 0)
        Y_test = np.concatenate((Y_test,data[i].Y_test), axis = 0)

    num_features = X_train.shape[1]

    logging.info('Size of X_Train: %s',X_train.shape)
    logging.info('Size of X_Train: %s', Y_train.shape)
    logging.info('Data loaded succesfully')

    dense_model = DenseModel(num_features)

    # load model from h5 file
    if load == 1:
        dense_model.load(".\saved_models\\Mercury 1.h5")
        results = dense_model.model.evaluate(X_test,Y_test,batch_size=32)
        print('test loss, test acc:',results)

    # build and train and save model
    else:
        print('Create the model.')
        dense_model.build_model()

        print('Create the trainer')
        trainer = Trainer(dense_model.model,X_train,Y_train,epochs=100,batch_size=32)

        print('Start training the model.')
        trainer.train()

        dense_model.save(".\saved_models\\Mercury 1.h5")


def combineData(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    data_dict = {}
    arr = []

    for f in onlyfiles:
        key = f.replace('.csv', '')
        data_dict.update({key: mypath + "\\" + f})

    for k in data_dict:
        fname = data_dict[k]
        arr.append(DataLoader(fname,
                              window=10,
                              threshold=0.03,
                              technicals=0,
                              featselect=0,
                              drop=0))
    return arr


if __name__ == '__main__':

    data = combineData(mypath)
    main(data,load=0)
    logging.info('Successful execution')
    os.system("tensorboard --logdir=.\\logs\\")

