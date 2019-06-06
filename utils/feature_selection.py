import logging
import pandas as pd
import numpy as np
from feature_selector import  FeatureSelector
from data_loader.data_loader import DataLoader
import matplotlib.pyplot as plt
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

def main():

    #Identify Missing Values
    fs.identify_missing(missing_threshold=0.6)

    #Identify Collinearity
    fs.identify_collinear(correlation_threshold = 0.98)
    fs.record_collinear.to_csv("csv//record_collinear.csv")

    #Identify Single Unique
    fs.identify_single_unique()
    fs.record_single_unique.to_csv("csv//record_single_unique.csv")

    #Zero importance
    fs.identify_zero_importance(task='classification',
                                eval_metric='multi_logloss',
                                n_iterations=10,
                                early_stopping=True)
    fs.record_zero_importance.to_csv("csv//record_zero_importance.csv")

    #Low Importance
    fs.identify_low_importance(cumulative_importance=0.99)
    fs.feature_importances.to_csv("csv//feature_importance.csv")

    #Identified features for removal
    summary = pd.DataFrame.from_dict(fs.ops,orient='index')
    summary.to_csv("csv//summary.csv")

if __name__ == '__main__':

    __AAPL__ = "D:\\Dropbox\\9. Data\\Mercury Data\\CSV\\CIQ_AAPL.csv"
    data = DataLoader(__AAPL__, window=10, threshold=0.03,drop=1)
    fs = FeatureSelector(data = data.df, labels=data.targets)

    main()










