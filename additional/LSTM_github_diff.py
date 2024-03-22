# -*- coding: utf-8 -*-
"""
Created on Thu May 12 12:13:55 2022

@author: suvapna.maheswaran
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 10:12:01 2022

@author: suvapna.maheswaran
"""

#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.pylab import rcParams
import seaborn as sns
rcParams['figure.figsize']=15,10
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense
from keras.layers import LSTM
import glob
from datetime import datetime
from keras.callbacks import EarlyStopping
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping



df = pd.read_csv('./new_df.csv')
print(df.head()) #7 columns, including the Date. 
usage_diff = pd.read_csv('./usage_diff_2.csv')


#merge datasets by index
finalDf = pd.merge(df, usage_diff, left_index=True, right_index=True)


#look for null values
print('Left out missing value:',finalDf.shape[0]-finalDf.count() )

#drop nullv value in the first two rows row
finalDf = finalDf.iloc[1: , :]
finalDf = finalDf.iloc[1: , :]

#Date as Index
finalDf.set_index('LOAD_DATE',inplace=True) 

#Dividing data in test and train sets
dataset = finalDf.USAGE_y.values #numpy.ndarray
dataset = dataset.astype('float32')
dataset = np.reshape(dataset, (-1, 1))
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset) * 0.70)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

''' Helper to create time frames with look backs '''
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

''' Creating time frames with look backs '''
look_back = 8
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

''' Re-shaping data for model requirement '''
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print('Training data size:',trainX.shape)
print('Test data size:',testX.shape)

''' Fitting the data in LSTM Deep Learning model '''
model = Sequential()
#trainX = (614,1,8) => input shape (1,8)
# 1 = number of window, 8= number of variables
model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
#history = model.fit(trainX, trainY, epochs=20, batch_size=100, verbose=2)
history = model.fit(trainX, trainY, epochs=50, batch_size=100, validation_data=(testX, testY), verbose=1, shuffle=False)
#end = time.time()
# Training Phase
model.summary()
# store model to json file

# To save model
model.save('LSTM_model_diff.hdf5')


''' Predicting 1 years data based on 5 years of previous data '''
yhat = model.predict(testX)

''' Plotting the first 500 entries to see prediction '''
pyplot.figure(figsize=(20,8))
pyplot.plot(yhat[:500], label='predict')
pyplot.plot(testY[:500], label='true')
pyplot.legend()

pyplot.ylabel('Usage', size=15)
pyplot.xlabel('Time step', size=15)
pyplot.legend(fontsize=15)

pyplot.show()


# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()

pyplot.title('model loss',size=15)
pyplot.ylabel('loss',size=15)
pyplot.xlabel('epochs',size=15)
pyplot.legend(loc='upper right',fontsize=15)

pyplot.show()

#####scatter plot
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

from sklearn.metrics import mean_squared_error
from numpy import sqrt 

rmse = sqrt(mean_squared_error(testY, yhat))
print('Test RMSE: %.3f' % rmse)


#check adfuller
from statsmodels.tsa.stattools import adfuller

result = adfuller(df.USAGE.dropna())
print(result)
print('ADF Test Statistic: %.2f' % result[0])
print('5%% Critical Value: %.2f' % result[4]['5%'])
print('p-value: %.2f' % result[1])
