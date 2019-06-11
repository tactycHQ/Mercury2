import os
import numpy as np
import pandas as pd
from data_loader.data_loader import DataLoader
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder


predict_path = "D:\\Dropbox\\9. Data\\Mercury Data\\XLS\\predict\\CIQ_AAPL_predict.csv"
window=45
threshold=0.035

df = pd.read_csv(predict_path,low_memory=False)
df = df.drop(['DATE'],axis=1)
prices = df['IQ_LASTSALEPRICE'].values.reshape(-1, 1)
bmark = df['BENCHMARK'].values.reshape(-1, 1)

len = prices.shape[0]
priceReturns = np.empty((len, 1))
bmarkReturns = np.empty((len, 1))
for i in range (0,len-window):
    priceReturns[i] = prices[i+window,0]/prices[i,0]-1
    bmarkReturns[i] = bmark[i+window, 0]/bmark[i, 0] - 1
priceReturns=priceReturns[:-window]
bmarkReturns=bmarkReturns[:-window]

relReturns = priceReturns - bmarkReturns
targets = []
for ret in relReturns:
    if ret > threshold:targets.append(1)
    elif ret < -threshold:targets.append(-1)
    else: targets.append(0)
targets = np.array(targets).reshape(-1, 1)
unique, counts = np.unique(targets, return_counts=True)
print("Target counts are %s %s", unique, counts)

ohe = OneHotEncoder(categories='auto')
targets_ohe = ohe.fit_transform(targets).toarray()

sc = StandardScaler()
sc.fit(df.values)
x_pred = sc.transform(df.values)

dense_model =load_model("C:\\Users\\anubhav\\Desktop\\Projects\\Mercury2\\saved_models\\run17.h5")
pred = dense_model.predict(x_pred)
np.savetxt("pred.csv",pred,delimiter=",")




print("prices:\n",prices)
print("priceReturns:\n",priceReturns)
print("bmarkReturns:\n",bmarkReturns)
print("relReturns:\n",relReturns)
print("targets:\n",targets)
print("targets_ohe:\n",targets_ohe)
print("pred:\n",pred)



# print(test_data)
