#MERCURY 1
import logging
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger

class Trainer:
    def __init__(self,model,train_dataset,val_dataset,config,steps_per_epoch, val_steps):
        self.model=model
        self.train_dataset=train_dataset
        self.val_dataset = val_dataset
        self.epochs=config.model.epochs
        self.steps_per_epoch=steps_per_epoch
        self.val_steps=val_steps
        self.callbacks=[]
        self.loss=[]
        self.acc=[]
        self.val_loss=[]
        self.val_acc=[]
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
        history = self.model.fit(x=self.train_dataset,
                                 validation_data=self.val_dataset,
                                 epochs=self.epochs,
                                 steps_per_epoch = self.steps_per_epoch,
                                 validation_steps = self.val_steps,
                                 callbacks=self.callbacks
                                 )
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])
        logging.info("Model fit success")