import numpy as np
import logging
import matplotlib.pyplot as plt
from data_loader.data_loader import DataLoader
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier


def main():
    # StockPriceChart()
    # TargetHistogram()
    FeatureImportance()

def StockPriceChart():
    fig, ax = plt.subplots()
    ax.plot(dates, prices, dates, bmark)
    ax.set_title("Asset Price vs. Benchmark")
    ax.legend(loc='upper left')
    ax.set_ylabel("Price")
    ax.set_xlabel("Date")

    fig.tight_layout()
    plt.show()

def TargetHistogram():
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.scatter(x=train_dates,y=relReturns,marker='.',c='r',edgecolor='b')
    ax1.set_title("Scatter Plot of Target Labels")
    ax1.set_ylabel("Relative Return to Benchmark")
    ax1.set_xlabel("Date")

    # ax2.hist()

    fig.tight_layout()
    plt.show()

def FeatureImportance():

    impt_factor = model.feature_importances_
    np.savetxt("impt.csv",impt_factor)

    fig, ax = plt.subplots(figsize=(30,10))
    ax.bar(features,impt_factor)
    ax.set_title("Feature Importance")
    ax.set_ylabel("Importance Value")
    ax.set_xlabel("Feature")
    plt.xticks(rotation=90)

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    __AAPL__ = "D:\\Dropbox\\9. Data\\Mercury Data\\CSV\\CIQ_AAPL.csv"
    data = DataLoader(__AAPL__, window=10, threshold=0.03)
    prices = data.prices
    bmark = data.bmark
    dates = data.dates
    targets = data.targets
    train_dates =  dates[:-data.window]
    relReturns = data.relReturns[:-data.window]
    features = data.features
    X_train = data.X_train_std
    Y_train = data.Y_train

    model=ExtraTreesClassifier()
    model.fit(X_train,Y_train)


    main()






































if __name__ == '__main__':
    main()
