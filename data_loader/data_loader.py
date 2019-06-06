#MERCURY 1
import logging
import pandas as pd
import sys
import numpy as np
from ta import *
from feature_selector import  FeatureSelector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s-%(process)d-%(levelname)s-%(message)s',datefmt='%d-%b-%y %H:%M:%S',stream=sys.stdout)

class DataLoader:

    def __init__(self,fname, window=1,threshold=0.05,technicals=0,featselect=0,drop=0):

        logging.info("Loading Data")

        self.window = window
        self.threshold = threshold
        self.featselect=featselect
        self.drop=drop
        self.technicals=technicals

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

        #add technical features
        if self.technicals==1:
            df = add_all_ta_features(df, "IQ_OPENPRICE", "IQ_HIGHPRICE", "IQ_LOWPRICE", "IQ_CLOSEPRICE", "IQ_VOLUME",fillna=False)
            df.to_csv(".\\utils\\csv\\DF with TA.csv")
        else: pass

        #save summary of all features to csv
        df.describe(include='all').to_csv(".\\utils\\csv\\all_features.csv")
        logging.info("All Features List Saved Under all_features.csv")

        # run and remove unimportant features
        if self.featselect==1:
            df = self.runFeatureSelector(df)
        else: pass

        # save summary of final features to csv
        df.describe(include='all').to_csv(".\\utils\\csv\\final_features.csv")
        logging.info("All Features List Saved Under final_features.csv")

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


    def runFeatureSelector(self,df):
        logging.info(("Running Feature Selection"))
        fs = FeatureSelector(data = df, labels=self.targets)

        # Identify Missing Values
        fs.identify_missing(missing_threshold=0.6)

        # Identify Collinearity
        fs.identify_collinear(correlation_threshold=0.98)
        fs.record_collinear.to_csv(".\\utils\\csv\\record_collinear.csv")

        # Identify Single Unique
        fs.identify_single_unique()
        fs.record_single_unique.to_csv(".\\utils\\csv\\record_single_unique.csv")

        # Zero importance
        fs.identify_zero_importance(task='classification',
                                    eval_metric='multi_logloss',
                                    n_iterations=10,
                                    early_stopping=True)
        fs.record_zero_importance.to_csv(".\\utils\\csv\\record_zero_importance.csv")

        # Low Importance
        fs.identify_low_importance(cumulative_importance=0.99)
        fs.feature_importances.to_csv(".\\utils\\csv\\feature_importance.csv")

        #generate summary of all operations
        summary = pd.DataFrame.from_dict(fs.ops, orient='index')
        summary.to_csv(".\\utils\\csv\\summary.csv")

        #if drop flag is 1, go ahead and remove the suggested features
        if self.drop==1:
            df = fs.remove(methods='all')
        else: pass

        return df

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
        self.X_train, self.X_test, Y_train,Y_test = train_test_split(self.inputs_trunc,self.targets_ohe_trunc,test_size=0.2,random_state=1,stratify=None)
        self.Y_train=np.reshape(Y_train,(-1, Y_train.shape[1]))
        self.Y_test=np.reshape(Y_test,(-1, Y_test.shape[1]))
        logging.info("Train and test sets have been split")

    def NormalizeData(self):
        """
        normalizes the training and test data
        :return:
        """
        sc = StandardScaler()
        sc.fit(self.X_train)
        self.X_train_std = sc.transform(self.X_train)
        self.X_test_std = sc.transform(self.X_test)

        logging.info("Train and test sets have been normalized")
        logging.info("X_train_std shape is %s", self.X_train_std.shape)
        logging.info("X_test_std is %s", self.X_test_std.shape)
        logging.info("Y_train shape is %s", self.Y_train.shape)
        logging.info("Y_test shape is %s", self.Y_test.shape)











