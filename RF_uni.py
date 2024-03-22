# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 14:27:43 2022

@author: suvapna.maheswaran
"""

#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from numpy import sqrt 
from keras.models import Sequential, load_model, save_model
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=15,10
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit



#Read the csv file
df = pd.read_csv('./new_df.csv')
print(df.head()) #7 columns, including the Date. 

# set the index as date
df.set_index('LOAD_DATE',inplace=True)
df.head()


#just take the Usage (8760,)
dataset = df.USAGE.values #numpy.ndarray
#convert into Array of float 
dataset = dataset.astype('float32')
#new shape to an array without changing its data. (8760,1)
dataset = np.reshape(dataset, (-1, 1))
#scale data
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

#test and train set
train = dataset[:7000, :]
test =  dataset[7000:, :]
print(len(train), len(test))


#creating time frames with lookbacks
def series_to_supervised(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

def mean_absolute_percentage_error(inv_testY, inv_yhat): 
    inv_testY, inv_yhat = np.array(inv_testY), np.array(inv_yhat)
    mape = np.mean(np.abs((inv_testY - inv_yhat) / inv_testY)) * 100
    print('Test Mape: %.3f' % mape)
        
look_back = 8
trainX, trainY = series_to_supervised(train, look_back)
testX, testY = series_to_supervised(test, look_back)


model = RandomForestRegressor(n_estimators=100,max_depth=5)
model.fit(trainX, trainY)

train_sizes, train_scores, test_scores = learning_curve(
    estimator=model,
    X=trainX,
    y=trainY,
    cv=5,
    scoring="neg_root_mean_squared_error",
)

train_mean = -train_scores.mean(axis=1)
test_mean = -test_scores.mean(axis=1)

plt.subplots(figsize=(15,10))
plt.plot(train_sizes, train_mean, label="train")
plt.plot(train_sizes, test_mean, label="test")

plt.title("model loss")
plt.xlabel("Training Set Size")
plt.ylabel("RMSE")
plt.legend(loc="best")

plt.show()


#predict values for textX
yhat = model.predict(testX)


#show last 100 entries
pyplot.figure(figsize=(20,8))
pyplot.plot(yhat[-100:], label='predict')
pyplot.plot(testY[-100:], label='true')
pyplot.legend()

pyplot.ylabel('usage', size=15)
pyplot.xlabel('time step', size=15)
pyplot.legend(fontsize=15)

pyplot.show()


# plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()

pyplot.title('model loss',size=15)
pyplot.ylabel('loss',size=15)
pyplot.xlabel('epochs',size=15)
pyplot.legend(loc='upper right',fontsize=15)

pyplot.show()

#scatter plot
plt.figure(figsize=(10,10))
plt.scatter(testY, yhat, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(yhat), max(testY))
p2 = min(min(yhat), min(testY))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()

#calcualte RMSE
rmse = sqrt(mean_squared_error(testY, yhat))
print('Test RMSE: %.3f' % rmse)



##################################inverse Data################################

#inverse scaled Data
rs_testY= testY.reshape(-1, 1)
inv_testY = scaler.inverse_transform(rs_testY)

#inverse scaled Data 
rs_yhat= yhat.reshape(-1, 1) 
inv_yhat = scaler.inverse_transform(rs_yhat)


#calcualte RMSE
rmse = sqrt(mean_squared_error(inv_testY, inv_yhat))
print('Test RMSE: %.3f' % rmse)

# calculate MAE
mae_= (mean_absolute_error(inv_testY, inv_yhat))
print('Test MAE: %.3f' % mae_)

mean_absolute_percentage_error(inv_testY, inv_yhat)

#scatter plot
plt.figure(figsize=(10,10))
plt.scatter(inv_testY, inv_yhat, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(inv_yhat), max(inv_testY))
p2 = min(min(inv_yhat), min(inv_testY))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()


#plotting the last 100 entries
pyplot.figure(figsize=(20,8))
pyplot.plot(inv_yhat[-100:], label='predict')
pyplot.plot(inv_testY[-100:], label='true')
pyplot.legend()

pyplot.ylabel('Usage', size=15)
pyplot.xlabel('Time step', size=15)
pyplot.legend(fontsize=15)

pyplot.show()
