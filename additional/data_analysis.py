# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 13:19:08 2022

@author: suvapna.maheswaran
"""

#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.pylab import rcParams
import statsmodels as sm
import seaborn as sns
import datetime




#---------------------------------import delivered data----------------------------
df = pd.read_csv('./groupByDate.csv')
df2 = pd.read_csv('./weather_.csv')
print(df.head()) #7 columns, including the Date.

#final = df2 = pd.read_csv('./finalDf.csv')
#search value
#print(df['INTERVAL'].where(df['INTERVAL'] == 23))



#drop not necessary clomun
df2= df2.drop(columns=['INTERVAL'])

groupBy= df2.groupby(['LOAD_DATE'],as_index=False)['temperature'].mean()


#change to date
df['LOAD_DATE'] = pd.to_datetime(df['LOAD_DATE'], infer_datetime_format= True)
#change to date
df2['LOAD_DATE'] = pd.to_datetime(df2['LOAD_DATE'], infer_datetime_format= True)

#merge  of datsets !!!!! df2 and groupBY should be merged
finalDf = pd.merge_asof(df, df2, on='LOAD_DATE')





#Use heatmap to see corelation between variables icluding weather
sns.heatmap(finalDf.corr(),annot=True,cmap='coolwarm', fmt =".2")
plt.title('Heatmap of co-relation between variables',fontsize=16)
plt.show()

finalDf.set_index('LOAD_DATE',inplace=True)

#plot weeekly 
df_sum_weekly = finalDf['USAGE'].resample('W').mean()
df_feature1 = finalDf['temperature'].resample('W').mean()
df_feature2 = finalDf['IS_WEEKEND'].resample('W').mean()

fig,ax = plt.subplots(figsize=(24,8))
ax.plot(df_sum_weekly.index,df_sum_weekly,color="red", marker=".")
ax.set_ylabel("Kwh")
ax.set_xlabel('Date')
ax2 = ax.twinx()
ax3 = ax.twinx()
ax2.plot(df_sum_weekly.index,df_feature1,color="blue",marker=".")
ax2.set_ylabel("Celcius")
fig.legend(["Weekly Energy Consumption","Weekly Temp"])
fig.show()

#plot dayly
df_sum_weekly = finalDf['USAGE'].resample('D').mean()
df_feature1 = finalDf['temperature'].resample('D').mean()
df_feature2 = finalDf['IS_WEEKEND'].resample('D').mean()

fig,ax = plt.subplots(figsize=(24,8))
ax.plot(df_sum_weekly.index,df_sum_weekly,color="red", marker=".")
ax.set_ylabel("Kwh")
ax.set_xlabel('Date')
ax2 = ax.twinx()
ax3 = ax.twinx()
ax2.plot(df_sum_weekly.index,df_feature1,color="blue",marker=".")
ax2.set_ylabel("Celcius")
fig.legend(["Dayly Energy Consumption","Dayly Temp"])
fig.show()


#save as csv
finalDf.to_csv('finalDayly.csv', index=True)


#--------------------------Durbin Watson Test---------------------------------
from statsmodels.formula.api import ols
from statsmodels.stats.stattools import durbin_watson
df = pd.read_csv('./new_df.csv')
df['LOAD_DATE'] = pd.to_datetime(df['LOAD_DATE'], infer_datetime_format= True)


model = ols('USAGE ~ LOAD_DATE + INTERVAL + TEMP + DAY_OF_WEEK ', data=df).fit()
print(model.summary())
durbin_watson(model.resid) 
# goupBy.csv = 0.14858286431404208 --> positive autocorrelation
# new_df.csv = 0.1729027090310036 --> positive autocorrelation

#--------------------differnecing Usage + Durbin Watson Test------------------


