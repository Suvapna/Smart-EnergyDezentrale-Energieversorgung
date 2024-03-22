# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 15:24:03 2022

@author: suvapna.maheswaran
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 11:28:55 2022

@author: suvapna.maheswaran
"""

from pandas import read_csv
import pandas as pd
import numpy as np
from datetime import datetime
from keras.layers import concatenate
from numpy import sqrt 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=15,10
 


#Load Dataset
dataset = pd.read_csv('./final_user_6000024897.csv')

#change index name
dataset.set_index('LOAD_DATE',inplace=True)

################################

# prepare data for lstm

# transform a time series dataset into a supervised learning dataset with names of cols and values 
# data : Sequence of observations as a list or 2D NumPy array.
# n_in : Number of lag observations as input ( X ).
# n_out : Number of observations as output ( y ).
#dropnan : Boolean to decide to drop rows with NaN values or not.
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
    #create two emmpty lists
	cols, names = list(), list()
    # range() returns a sequence of numbers, starting from 0 by default, increments by 1 by default, and stops before a specified number.
    # range(start, stop, step)
	# input sequence (t-n, ... t-1) => Value X
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
        #create column names
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# output sequence (t, t+1, ... t+n) => Value Y
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	#print(cols)
	agg = concat(cols, axis=1)
	#print(names)
	agg.columns = names
	print(agg)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

#var1(t-1)  var2(t-1)  var3(t-1)  ...   var4(t)  var5(t)  var6(t)
#0           NaN        NaN        NaN  ...  0.166667      0.0      1.0

#swap columns
columns_titles = ['USAGE', 'INTERVAL', 'TEMP', 'DAY_OF_WEEK', 'IS_WEEKEND', 'IS_HOLIDAY','RADIATION_SURFACE','RADIATION_TOA']
dataset=dataset.reindex(columns=columns_titles)

#Cols for training
cols = list(dataset)[0:8]
#Date and volume columns are not used in training. 
print(cols) #['INTERVAL', 'USAGE', 'TEMP', 'DAY_OF_WEEK', 'IS_WEEKEND', 'IS_HOLIDAY']
#convert all columns to float 
dataset = dataset[cols].astype(float)

# load dataset
values = dataset.values
# normalize features between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
#using 7 varuables and the the "hour before pollution" to predict the "next hour pollution"
print(reframed.head())

'''
Splits the dataset into train and test sets
then splits the train and test sets into input and output variables.
Finally, the inputs (X) are reshaped into the 3D format expected by LSTMs, 
namely [samples, timesteps, features].
'''
# split into train and test sets
values = reframed.values

train = values[:7000, :]
test =  values[7000:, :]

# split into input and outputs
#first 6 Columns in train_X as input and tain_y the ouput ( the usage of the next hour)
#trainX = train[all rows, all except the last column]
train_X, train_y = train[:, :-1], train[:, -1]
#same as above for test set
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features] for LSTM
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape) 
#(4392, 1, 8) (4392,) (4367, 1, 8) (4367,)

'''
Defining the LSTM with 50 neurons in the first hidden layer and 
1 neuron in the output layer for predicting usage.
The input shape will be 1 time step with 6 features.

The model will be fit for 50 training epochs with a batch size of 72. 
Remember! The internal state of the LSTM in Keras is reset at the end of each batch.

Finally, keeping track of both the training and test loss during training by setting the validation_data argument in the fit() function.
At the end of the run both the training and test loss are plotted.

'''


# design network
model = Sequential()
#train_X_shape = (4392, 1, 8) -> [1]= 1 output  and [2] = 8 features
model.add(LSTM(50,input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=1000, batch_size=64, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
 


# make a prediction
yhat = model.predict(test_X)
#(4367,1)

test_X = test_X.reshape((test_X.shape[0], test_X.shape[2])) 
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]


rmse_noInv = sqrt(mean_squared_error(test_y, yhat))
print('Test RMSE no Invers: %.3f' % rmse_noInv)


''' Plotting the last 100 entries to see prediction '''
plt.plot(inv_yhat[-100:], label='predict')
plt.plot(inv_y[-100:], label='true')
plt.legend()

plt.ylabel('Usage', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)

plt.show()


#scatter plot
plt.figure(figsize=(10,10))
plt.scatter(inv_y, inv_yhat, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(inv_yhat), max(inv_y))
p2 = min(min(inv_yhat), min(inv_y))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()


# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

# calculate MAE
mae_= (mean_absolute_error(inv_y, inv_yhat))
print('Test MAE: %.3f' % mae_)

def mean_absolute_percentage_error(inv_y, inv_yhat): 
    inv_y, inv_yhat = np.array(inv_y), np.array(inv_yhat)
    mape = np.mean(np.abs((inv_y - inv_yhat) / inv_y)) * 100
    print('Test Mape: %.3f' % mape)


mean_absolute_percentage_error(inv_y, inv_yhat)