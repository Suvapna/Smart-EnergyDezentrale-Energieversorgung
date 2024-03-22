# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 16:25:28 2022

@author: suvapna.maheswaran
"""

#Import packages
import pandas as pd
from sklearn.metrics import mean_squared_error
from numpy import sqrt 
import matplotlib.pyplot as plt

y = []
yhat = []



data = pd.read_csv('./new_df.csv')

usage_thursday = data[["USAGE", "DAY_OF_WEEK", "INTERVAL"]]

def predictBymean(day,interval):
    
    thursday = usage_thursday[(usage_thursday["DAY_OF_WEEK"] == day) & (usage_thursday["INTERVAL"] == interval)]
    
    predict = (thursday.tail(1)).USAGE.item() #save 
    
    thursday.drop(thursday.tail(1).index,inplace=True) # drop last row
    
    predicted = thursday["USAGE"].mean()
    
    
    print("expected: " + str(predict) + " predicted: " + str(predicted) + " Day: " +  str(day) + " Interval:" + str(interval))
    y.append(predict)
    yhat.append(predicted)
    

    
for i in range(2, 7):
    for j in range(1, 25):  
        predictBymean(i, j)
        
for i in range(0, 1):
    for j in range(1, 25):  
        predictBymean(i, j)
        
for i in range(1, 2):
    for j in range(1, 25):  
        predictBymean(i, j)

# calculate RMSE
rmse = sqrt(mean_squared_error(y, yhat))
print('Test RMSE: %.3f' % rmse)



''' Plotting entries to see prediction '''
plt.plot(yhat, label='predict')
plt.plot(y, label='true')
plt.legend()

plt.ylabel('Usage', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)

plt.show()
