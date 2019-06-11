#MERCURY 1 DEV
import logging
import pandas as pd
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
logging.basicConfig(level=logging.DEBUG ,
                    format='%(asctime)s-%(process)d-%(levelname)s-%(message)s' ,
                    datefmt='%d-%b-%y %H:%M:%S',
                    stream=sys.stdout)

class DataLoader:

    def __init__(self,fname, window=1,threshold=0.05,split=0.2):

        logging.info("----------Loading Data for %s-----------",fname)

        self.window = window
        self.threshold = threshold
        self.split=split

        self.dates = None
        self.features=None
        self.inputs = None
        self.inputs_trunc = None
        self.targets=None
        self.targets_ohe=None
        self.targets_ohe_trunc = None
        self.relReturns = None

        self.df = self.createDf(fname)
        self.features = self.df.columns  # Feature Names
        self.inputs = self.df.values  # Feature Values
        self.truncateData()
        self.splitData()
        self.NormalizeData()

    def createDf(self,fname):
        """
        :param fname: The file to load
        :return: dataframes for training
        """

        df = pd.read_csv(fname,low_memory=False)
        logging.info("Creating Dataframe from CSV")

        #excluding date column as not important for model. dates saved under self.dates
        df['DATE'] = pd.to_datetime(df['DATE'])
        self.dates = df.loc[:,'DATE'].values.reshape(-1, 1)
        logging.info("Date Reformatted")
        df = df.drop(['DATE'],axis=1)

        #extracting prices and benchmark for target label creation
        self.prices = df['IQ_LASTSALEPRICE'].values.reshape(-1, 1)
        self.bmark = df['BENCHMARK'].values.reshape(-1, 1)
        self.createTargets()

        #save summary of all features to csv
        df.describe(include='all').to_csv(".\\utils\\csv\\all_features.csv")
        logging.info("All Features List Saved Under all_features.csv")

        return df


    def truncateData(self):
        """
        Truncates data to accomodate for window length at the end of the data
        :param dframe:
        :return:
        """
        # Truncates feature values to accomodate window
        self.inputs_trunc = self.inputs[:-self.window]
        logging.info("Inputs created of shape %s",self.inputs_trunc.shape)

        # Truncates target labels to match size of feature
        self.targets_ohe_trunc = self.targets_ohe[:-self.window]
        logging.info("Targets are one hot encoded and transformed to shape %s", self.targets_ohe_trunc.shape)

    def createTargets(self):
        """
        creates target labels
        relReturns: 1d vector of all relReturns
        targets = 1d vector of all (-1,0,1) labels
        targets_ohe = OHE matrix of targets vector and also truncated for window
        :return:
        """

        #compute relative returns to benchmark
        pctReturns = self.createPctReturns(self.prices)
        bMarkReturns = self.createPctReturns(self.bmark)
        self.relReturns = pctReturns - bMarkReturns

        #create target vector of class labels. 1: up, 2: down, 3: flat
        targets = []
        for ret in self.relReturns:
            if ret>self.threshold:
                targets.append(1)
            elif ret < -self.threshold:
                targets.append(-1)
            else:
                targets.append(0)
        self.targets = np.array(targets).reshape(-1,1)

        #create output showing distribution of class labels
        unique, counts = np.unique(self.targets, return_counts=True)
        logging.info("Target counts are %s %s", unique, counts)

        #one hot encode targets
        ohe = OneHotEncoder(categories='auto')
        self.targets_ohe = ohe.fit_transform(self.targets).toarray()


    def createPctReturns(self,close):
        """
        computes % returns
        :param close: closing prices
        :return:
        """
        len = close.shape[0]
        pctReturns = np.empty((len, 1))
        for i in range (0,len-self.window):
            pctReturns[i] = close[i+self.window,0]/close[i,0]-1
        return pctReturns

    def splitData(self):
        """
        splits inputs and targets into training and test sets
        :return:
        """
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.inputs_trunc,self.targets_ohe_trunc,test_size=self.split,random_state=1,stratify=None)
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train, self.Y_train,test_size=self.split, random_state=1, stratify=None)
        self.Y_train=np.reshape(self.Y_train,(-1, self.Y_train.shape[1]))
        self.Y_test=np.reshape(self.Y_test,(-1, self.Y_test.shape[1]))
        logging.info("Train, validation, test sets have been split")

    def NormalizeData(self):
        """
        normalizes the training and test data
        :return:
        """
        sc = StandardScaler()
        sc.fit(self.X_train)
        self.X_train_std = sc.transform(self.X_train)
        self.X_val_std = sc.transform(self.X_val)
        self.X_test_std = sc.transform(self.X_test)


        logging.info("Train, validation, and test sets have been normalized")
        logging.info("X_train_std shape is %s", self.X_train_std.shape)
        logging.info("X_val_std shape is %s", self.X_val_std.shape)
        logging.info("X_test_std is %s", self.X_test_std.shape)
        logging.info("Y_train shape is %s", self.Y_train.shape)
        logging.info("Y_test shape is %s", self.Y_val.shape)
        logging.info("Y_test shape is %s", self.Y_test.shape)











