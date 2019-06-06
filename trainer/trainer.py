#MERCURY 1
import logging
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger

class Trainer:
    def __init__(self,model,X_train,Y_train,epochs,batch_size):
        self.model=model
        self.X_train=X_train
        self.Y_train=Y_train
        self.epochs=epochs
        self.batch_size=batch_size
        self.callbacks=[]
        self.loss=[]
        self.acc=[]
        # self.val_loss=[]
        # self.val_acc=[]
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            CSVLogger('.\\logs\\training_log.csv',
                      separator=',',
                      append=False)
        )

        self.callbacks.append(
            TensorBoard(
                log_dir='.\\logs\\',
                write_graph=True,
            )
        )


    def train(self):
        logging.info("Beginning Model Fit")
        history = self.model.fit(
        self.X_train,
        self.Y_train,
        epochs=self.epochs,
        batch_size=self.batch_size,
        callbacks=self.callbacks
        )
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        # self.val_loss.extend(history.history['val_loss'])
        # self.val_acc.extend(history.history['val_acc'])
        logging.info("Model fit success")