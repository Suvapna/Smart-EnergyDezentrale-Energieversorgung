# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 09:00:37 2022

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
rcParams['figure.figsize']=15,10

#read the data from csv
df = pd.read_csv('./INTERVAL_DATA_2019.csv')
print(df.head()) #7 columns, including the Date.

df2 = pd.read_csv('./weather.csv')
print(df.head()) #7 columns, including the Date.

#search value
#winter = (df['INTERVAL'].where(df['INTERVAL'] == 23))
#save as csv
#winter.to_csv('winter.csv', index=False)



#change into datetime in each dataset (timestamp)
df['LOAD_DATE'] = pd.to_datetime(df['LOAD_DATE'], infer_datetime_format= True)
df2['time'] = pd.to_datetime(df2['time'], infer_datetime_format= True)

#------------- prepare wetaher data to use 


#extract time and date
df2['LOAD_DATE'] = df2['time'].dt.date
df2['INTERVAL'] = df2['time'].dt.time

#change object to datetime
df2['LOAD_DATE'] = pd.to_datetime(df2['LOAD_DATE'], infer_datetime_format= True)

#datatypes
df.dtypes
df2.dtypes


#drop not necessary clomun
df2= df2.drop(columns=['time','local_time','TIME'])
df= df.drop(columns=('CSPT_ID'))


#change Interval to number -->run stepby step!
#df2.replace({'INTERVAL' : { '00:00:00' : '1', '01:00:00' : '2', '02:00:00' : '3','03:00:00' : '4','04:00:00' : '5','05:00:00' : '6','06:00:00' : '7','07:00:00' : '8','08:00:00' : '9','09:00:00' : '10','10:00:00' : '11','11:00:00' : '12','12:00:00' : '13','13:00:00' : '14','14:00:00' : '15','15:00:00' : '16','16:00:00' : '17','17:00:00' : '18','18:00:00' : '19','19:00:00' : '20','20:00:00' : '21','21:00:00' : '22','22:00:00' : '23','23:00:00' : '24'}})
df2['INTERVAL'] = df2['INTERVAL'].astype(str).str.replace('00:00:00','1')
df2['INTERVAL'] = df2['INTERVAL'].astype(str).str.replace('01:00:00','2')
df2['INTERVAL'] = df2['INTERVAL'].astype(str).str.replace('02:00:00','3')
df2['INTERVAL'] = df2['INTERVAL'].astype(str).str.replace('03:00:00','4')
df2['INTERVAL'] = df2['INTERVAL'].astype(str).str.replace('04:00:00','5')
df2['INTERVAL'] = df2['INTERVAL'].astype(str).str.replace('05:00:00','6')
df2['INTERVAL'] = df2['INTERVAL'].astype(str).str.replace('06:00:00','7')
df2['INTERVAL'] = df2['INTERVAL'].astype(str).str.replace('07:00:00','8')
df2['INTERVAL'] = df2['INTERVAL'].astype(str).str.replace('08:00:00','9')
df2['INTERVAL'] = df2['INTERVAL'].astype(str).str.replace('09:00:00','10')
df2['INTERVAL'] = df2['INTERVAL'].astype(str).str.replace('10:00:00','11')
df2['INTERVAL'] = df2['INTERVAL'].astype(str).str.replace('11:00:00','12')
df2['INTERVAL'] = df2['INTERVAL'].astype(str).str.replace('12:00:00','13')
df2['INTERVAL'] = df2['INTERVAL'].astype(str).str.replace('13:00:00','14')
df2['INTERVAL'] = df2['INTERVAL'].astype(str).str.replace('14:00:00','15')
df2['INTERVAL'] = df2['INTERVAL'].astype(str).str.replace('15:00:00','16')
df2['INTERVAL'] = df2['INTERVAL'].astype(str).str.replace('16:00:00','17')
df2['INTERVAL'] = df2['INTERVAL'].astype(str).str.replace('17:00:00','18')
df2['INTERVAL'] = df2['INTERVAL'].astype(str).str.replace('18:00:00','19')
df2['INTERVAL'] = df2['INTERVAL'].astype(str).str.replace('19:00:00','20')
df2['INTERVAL'] = df2['INTERVAL'].astype(str).str.replace('20:00:00','21')
df2['INTERVAL'] = df2['INTERVAL'].astype(str).str.replace('21:00:00','22')
df2['INTERVAL'] = df2['INTERVAL'].astype(str).str.replace('22:00:00','23')
df2['INTERVAL'] = df2['INTERVAL'].astype(str).str.replace('23:00:00','24')


#group byDate and Interval 
groupBy = df.groupby(['LOAD_DATE', 'INTERVAL'])['USAGE'].sum()

#change the dtypes
df2['LOAD_DATE'] = pd.to_datetime(df2['LOAD_DATE'], infer_datetime_format= True)
df2['INTERVAL'] = df2['INTERVAL'].astype(np.int64)


#save as CSV
df2.to_csv('weather_.csv', index=False)

#plot weather data
#df2.set_index('time')[['temperature']].plot(subplots=True)


#save as CSV
groupBy.to_csv('groupBy.csv', index=True)

#import
df3 = pd.read_csv('./groupBy.csv')
print(df3.head()) #7 columns, including the Date.
df3['LOAD_DATE'] = pd.to_datetime(df3['LOAD_DATE'], infer_datetime_format= True)

#import
df4 = pd.read_csv('./weather_.csv')
print(df4.head()) #7 columns, including the Date.
df4['LOAD_DATE'] = pd.to_datetime(df4['LOAD_DATE'], infer_datetime_format= True)


#shape with weekends 
#df.shape #(1445736, 6)

#look for Null-Values in Dataset
print('Count of missing values:\n',df4.shape[0]-df4.count())

#merge both datsets
#finalDf = pd.merge_asof(df3, df4, on=["LOAD_DATE", "INTERVAL"])
new_df = pd.merge(df3, df4,  how='left', left_on=['LOAD_DATE','INTERVAL'], right_on = ['LOAD_DATE','INTERVAL'])


#getweekends
#finalDf["LOAD_DATE"] = pd.to_datetime(groupBy["LOAD_DATE"])
new_df["DAY_OF_WEEK"] = new_df["LOAD_DATE"].dt.weekday
# display the dataframe
# check if the date is weekend or not
new_df["IS_WEEKEND"] = new_df["DAY_OF_WEEK"] >= 5
# display the dataframe
print(new_df)


#is an holiday?
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
dr = pd.date_range(start='2019-01-01', end='2019-12-31')
cal = calendar()
holidays = cal.holidays(start=dr.min(), end=dr.max())
new_df['IS_HOLIDAY'] = new_df['LOAD_DATE'].isin(holidays)

#save as csv
new_df.to_csv('new_df.csv', index=False)

#Use heatmap to see corelation between variables
sns.heatmap(new_df.corr(),annot=True,cmap='coolwarm', fmt =".2")
plt.title('Heatmap of co-relation between variables',fontsize=16)
plt.show()

#------------------plot temp and usage/ need to sum up the load date-------------------------

# drop Interval Column
plot_df3= df3.drop(columns=['INTERVAL'])
plot_df4= df4.drop(columns=['INTERVAL'])


#group byDate 
plot_df3 = plot_df3.groupby(['LOAD_DATE'])['USAGE'].sum()
plot_df4 = plot_df4.groupby(['LOAD_DATE'])['temperature'].mean()

#save as csv
plot_df3.to_csv('plot_df3.csv', index=True)
plot_df4.to_csv('plot_df4.csv', index=True)

#import
plot_df3 = pd.read_csv('./plot_df3.csv')
plot_df3['LOAD_DATE'] = pd.to_datetime(plot_df3['LOAD_DATE'], infer_datetime_format= True)

#import
plot_df4 = pd.read_csv('./plot_df4.csv')
plot_df4['LOAD_DATE'] = pd.to_datetime(plot_df4['LOAD_DATE'], infer_datetime_format= True)

plot_dayly = pd.merge_asof(plot_df3, plot_df4, on=["LOAD_DATE"])

#save as csv
plot_dayly.to_csv('plot_dayly.csv', index=False)


#------------------------------create dataset for dayly usage--------------------------
plot = pd.read_csv('./plot_dayly.csv')
plot['LOAD_DATE'] = pd.to_datetime(plot['LOAD_DATE'], infer_datetime_format= True)


#change index to datetime
plot.set_index('LOAD_DATE',inplace=True)      

df_sum_weekly = plot['USAGE'].resample('D').mean()
df_feature1 = plot['temperature'].resample('D').mean()

fig,ax = plt.subplots(figsize=(24,8))
ax.plot(df_sum_weekly.index,df_sum_weekly,color="red", marker=".")
ax.set_ylabel("Kwh")
ax.set_xlabel('Date')
ax2 = ax.twinx()
ax3 = ax.twinx()
ax2.plot(df_sum_weekly.index,df_feature1,color="blue",marker=".")
ax2.set_ylabel("Celcius")
fig.legend(["Dayly Energy Consumption","Dayly Interval"])
fig.show()


#------------------autocorrelation/diff-------------------------
acf_diff = pd.read_csv('./new_df.csv')
acf_diff['LOAD_DATE'] = pd.to_datetime(acf_diff['LOAD_DATE'], infer_datetime_format= True)

#two variables not more (date+usage or date+temp) and put the datetime as index!
acf_diff= acf_diff.drop(columns=['INTERVAL','TEMP', 'IS_WEEKEND', 'DAY_OF_WEEK', 'IS_HOLIDAY'])
#acf_diff= acf_diff.drop(columns=['temperature'])


#check data types
acf_diff.dtypes

#set date as index!
acf_diff = acf_diff.set_index('LOAD_DATE')

#ACF --> Auto Correrlation Function
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(acf_diff)
plt.show()

#differencing
usage_diff = acf_diff.diff(periods=1)
plot_acf(usage_diff[1:])
plt.show

#2differencing
usage_diff_2 = usage_diff.diff(periods=1).dropna()
plot_acf(usage_diff_2[1:])
plt.show

#get result 
from statsmodels.tsa.stattools import adfuller
result = adfuller(usage_diff_2.USAGE.dropna())
print(result)
print('ADF Test Statistic: %.2f' % result[0])
print('5%% Critical Value: %.2f' % result[4]['5%'])
print('p-value: %.2f' % result[1])



#save as csv
usage_diff.to_csv('usage_diff.csv', index=True)

usage_diff_2.to_csv('usage_diff_2.csv', index=False)

#------------------autocorrelation/diff test with new_df-------------------------
acf_diff = pd.read_csv('./new_df.csv')
acf_diff['LOAD_DATE'] = pd.to_datetime(acf_diff['LOAD_DATE'], infer_datetime_format= True)

#two variables not more (date+usage or date+temp) and put the datetime as index!
acf_diff= acf_diff.drop(columns=['TEMP', 'IS_WEEKEND', 'DAY_OF_WEEK','IS_HOLIDAY','INTERVAL'])


#check data types
acf_diff.dtypes

#set date as index!
acf_diff = acf_diff.set_index('LOAD_DATE')

#ACF --> Auto Correrlation Function
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(acf_diff)
plt.show()

#differencing
usage_diff = acf_diff.diff(periods=1)
plot_acf(usage_diff[1:])
plt.show


#------------------plot Dayly with weeknds, holiday --> heatmap -------------------------

#import
df3 = pd.read_csv('./plot_dayly.csv')
print(df3.head()) #7 columns, including the Date.
df3['LOAD_DATE'] = pd.to_datetime(df3['LOAD_DATE'], infer_datetime_format= True)


#getweekends
#finalDf["LOAD_DATE"] = pd.to_datetime(groupBy["LOAD_DATE"])
df3["DAY_OF_WEEK"] = df3["LOAD_DATE"].dt.weekday
# display the dataframe
# check if the date is weekend or not
df3["IS_WEEKEND"] = df3["DAY_OF_WEEK"] >= 5



#is an holiday?
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
dr = pd.date_range(start='2019-01-01', end='2019-12-31')
cal = calendar()
holidays = cal.holidays(start=dr.min(), end=dr.max())
df3['IS_HOLIDAY'] = df3['LOAD_DATE'].isin(holidays)


#Use heatmap to see corelation between variables
sns.heatmap(df3.corr(),annot=True,cmap='coolwarm', fmt =".2")
plt.title('Heatmap of co-relation between variables',fontsize=16)
plt.show()

