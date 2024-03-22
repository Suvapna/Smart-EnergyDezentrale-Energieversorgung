# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 15:51:13 2022

@author: suvapna.maheswaran
"""

#Import packages
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt


#mean and variance
data = pd.read_csv('./new_df.csv')
X = data.USAGE

data = data.set_index('LOAD_DATE')

split = round(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))

#AdFuller
# Import adfuller
from statsmodels.tsa.stattools import adfuller

result = adfuller(data.USAGE.dropna())
print(result)
print('ADF Test Statistic: %.2f' % result[0])
print('5%% Critical Value: %.2f' % result[4]['5%'])
print('p-value: %.2f' % result[1])


#AdFuller2
result = adfuller(X, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}') 
    
    
#Kwiatkowski-Phillips-Schmidt-Shin test
from statsmodels.tsa.stattools import kpss

result = kpss(data.USAGE)
print(result)
print('KPSS Test Statistic: %.2f' % result[0])
print('5%% Critical Value: %.2f' % result[3]['5%'])
print('p-value: %.2f' % result[1])


#NON STATIONARY !


#Differencing
data['Difference'] = data['USAGE'].diff().diff(periods=1)

# Plot the Change
plt.figure(figsize=(10, 7))
plt.plot(data['Difference'])
plt.title('First Order Differenced Series', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Difference', fontsize=12)
plt.show()

#check stationary again
result = adfuller(data.Difference.dropna())
print(result)
print('ADF Test Statistic: %.2f' % result[0])
print('5%% Critical Value: %.2f' % result[4]['5%'])
print('p-value: %.2f' % result[1])


#differencing
from statsmodels.graphics.tsaplots import plot_acf

usage_diff = data['Difference'].diff(periods=1)
plot_acf(usage_diff[1:])
plt.show