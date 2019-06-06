#MERCURY 1
import logging
import tensorflow as tf
from keras import layers, Sequential, optimizers, models

class DenseModel:

    def __init__(self,features):
        self.model=None
        self.features=features
        tf.logging.set_verbosity(tf.logging.ERROR)

    def build_model(self):
        """
        builds the keras model
        :return: model
        """
        logging.info('Building model...')
        self.model = Sequential()

        self.model.add(layers.Dense(256, activation='relu',input_shape=(self.features,)))
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dense(3, activation='softmax'))

        logging.info('Compiling model...')
        self.model.compile(optimizer=optimizers.Adam(0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print(self.model.summary())
        return self.model

    def save(self,checkpoint_path):
        """
        saves the model into a h5 file
        :param checkpoint_path: file path
        :return:
        """
        logging.info("Saving model...")
        self.model.save(checkpoint_path)
        logging.info("Model saved")

    def load(self,checkpoint_path):
        """
        loads an H5 file
        :param checkpoint_path:file path
        :return:
        """
        logging.info("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model= models.load_model(checkpoint_path)
        logging.info('Model loaded')

