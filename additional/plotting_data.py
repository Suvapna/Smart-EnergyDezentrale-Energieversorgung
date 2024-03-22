# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:04:09 2022

@author: suvapna.maheswaran
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns




#---------------------------------import delivered data----------------------------
df = pd.read_csv('./DELIVERED_LOAD_2019.csv')
print(df.head()) #7 columns, including the Date.

#change to date
df['LOAD_DATE'] = pd.to_datetime(df['LOAD_DATE'], infer_datetime_format= True)

#by date
df.set_index('LOAD_DATE')[['INTERVAL_USAGE']].plot(subplots=True)

#by hour
df.set_index('INTERVAL')[['INTERVAL_USAGE']].plot(subplots=True)

#y=usage x=date
print(df.pivot(index='LOAD_DATE',columns='INTERVAL',values='INTERVAL_USAGE'))

ax= (df.pivot(index='LOAD_DATE',columns='INTERVAL',values='INTERVAL_USAGE')).plot()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


#y=usage x=hour
print(df.pivot(index='INTERVAL',columns='LOAD_DATE',values='INTERVAL_USAGE'))

ax = (df.pivot(index='INTERVAL',columns='LOAD_DATE',values='INTERVAL_USAGE')).plot()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


#23 hour 
filt = (df['INTERVAL'] == 23)
user = df[filt]
user.set_index('LOAD_DATE')[['INTERVAL_USAGE']].plot(subplots=True)



###########################getweekends
df["LOAD_DATE"] = pd.to_datetime(df["LOAD_DATE"])
df["DayOfWeek"] = df["LOAD_DATE"].dt.weekday
# display the dataframe
# check if the date is weekend or not
df["IsWeekend"] = df["DayOfWeek"] >= 5
# display the dataframe
print(df)


#filter USAGE by weekdays
filt = (df['IsWeekend'] == True)
user = df[filt]
user.set_index('LOAD_DATE')[['INTERVAL_USAGE']].plot(subplots=True)



#---------------------------------import intervall data-----------------------------
df = pd.read_csv('./INTERVAL_DATA_2019.csv')
print(df.head()) #7 columns, including the Date.

#change to date
df['LOAD_DATE'] = pd.to_datetime(df['LOAD_DATE'], infer_datetime_format= True)


#bydate
df.set_index('LOAD_DATE')[['USAGE']].plot(subplots=True)


#hour
df.set_index('INTERVAL')[['USAGE']].plot(subplots=True)

#byID
df.set_index('CSPT_ID')[['USAGE']].plot(subplots=True)

#delete other colums and plot/ ist das selbe wie #bydate
x = df[['INTERVAL']]
y = df[['USAGE']]
plt.plot(x,y)



###############filtered by ID 6000000249
filt = (df['CSPT_ID'] == 6000024897)
user = df[filt]
#just use the Columns 2,3,4
cols = list(user)[1:4]
user_for_plotting = user[cols]
#group by date and sum() the usage
bd = user_for_plotting.groupby(['LOAD_DATE'])['USAGE'].sum()
bd.plot(x='LOAD_DATE',y='USAGE')

#save as csv
#user.to_csv('user_6000000249.csv', index=False)

#group by hour and sum() the usage
bd = df.groupby(['INTERVAL'])['USAGE'].sum()
bd.plot(x='INTERVAL',y='USAGE')


#### FRAGE? Wieso sieht das komplett anders aus ?



# ID 6000000249 by date
user.set_index('LOAD_DATE')[['USAGE']].plot(subplots=True)

# ID 6000000249 by hour
user.set_index('INTERVAL')[['USAGE']].plot(subplots=True)




################filtered Thursdays; November 14 --> Normal Day

filt = (df['LOAD_DATE'] == '2019-11-16')
user = df[filt]

# ID 2019-11-14 by hour
user.set_index('INTERVAL')[['USAGE']].plot(subplots=True)




################filtered  Thanksgiving & 6000000249
filt = (df['CSPT_ID'] == 6000094179) & (df['LOAD_DATE'] == '2019-11-14')
user = df[filt]


# ID 6000000249 by hour
user.set_index('INTERVAL')[['USAGE']].plot(subplots=True)


#23 hour 
filt = (df['CSPT_ID'] == 6000024897) & (df['INTERVAL'] == 23)
user = df[filt]
user.set_index('LOAD_DATE')[['USAGE']].plot(subplots=True)




###########################getweekends
df["LOAD_DATE"] = pd.to_datetime(df["LOAD_DATE"])
df["DayOfWeek"] = df["LOAD_DATE"].dt.weekday
# display the dataframe
# check if the date is weekend or not
df["IsWeekend"] = df["DayOfWeek"] >= 5
# display the dataframe
print(df)


#filter USAGE by weekdays
filt = (df['IsWeekend'] == True)
user = df[filt]
user.set_index('LOAD_DATE')[['USAGE']].plot(subplots=True)


#scatterplot group by Date
sc = df.groupby(['LOAD_DATE'])['USAGE'].sum()
sc.plot(x='LOAD_DATE',y='USAGE',style='.')




#-------------------------------import monthly data---------------------------------
df = pd.read_csv('./USA_2019.csv')
print(df.head()) #7 columns, including the Date.

#change to date
df['START_DATE'] = pd.to_datetime(df['START_DATE'], infer_datetime_format= True)
df['END_DATE'] = pd.to_datetime(df['END_DATE'], infer_datetime_format= True)


#byenddate
df.set_index('END_DATE')[['USAGE']].plot(subplots=True)

#bystartdate
df.set_index('START_DATE')[['USAGE']].plot(subplots=True)



################filtered 6000067280
filt = (df['CSPT_ID'] == 6000083152)
user = df[filt]


# ID 6000067280 monthly USAGE
user.set_index('START_DATE')[['USAGE']].plot(subplots=True)

#selbes Ergebnis
#user.set_index('START_DATE')[['USAGE']].plot(subplots=True)

###############group by external ID count usage
bd = df.groupby(['EXTERNAL_ID'])['USAGE'].sum()
bd.plot(x='INTERVAL',y='USAGE')

################filtered EXTERNAL_ID
filt = (df['EXTERNAL_ID'] == 'OUTLIGHT')
user = df[filt]

# ID Domestic monthly USAGE
user.set_index('START_DATE')[['USAGE']].plot(subplots=True)

##############filtered EXTERNAL ID each line each ID##### 
filt = (df['EXTERNAL_ID'] == 'DOMESTIC')
domestic = df[filt]
dom = (domestic[['START_DATE']] ,domestic[['USAGE']])

filt = (df['EXTERNAL_ID'] == 'OUTLIGHT')
outlight = df[filt]
out = outlight.set_index('START_DATE')[['USAGE']]

filt = (df['EXTERNAL_ID'] == 'GEN1')
genOne = df[filt]
#genOne.set_index('START_DATE')[['USAGE']].plot(subplots=True)

filt = (df['EXTERNAL_ID'] == 'GEN2')
genTwo = df[filt]
#genTwo.set_index('START_DATE')[['USAGE']].plot(subplots=True)

#funktioniert noch nicht
#plt.plot(dom, out)







#------------------------------realloc--------------------------------------#

df = pd.read_csv('./REALLOC_SPLR_LOADS_BY_CLASS_2019.csv')
print(df.head()) #7 columns, including the Date.

#change to date
df['LOAD_DATE'] = pd.to_datetime(df['LOAD_DATE'], infer_datetime_format= True)

df.set_index('LOAD_DATE')[['SUM(RECONCILED_USAGE)']].plot(subplots=True)


##################filter nach ID 
filt = (df['RPRL_RGPT_ID_ESP'] == 5180)
user = df[filt]


# ID 5034 monthly USAGE
user.set_index('LOAD_DATE')[['SUM(RECONCILED_USAGE)']].plot(subplots=True)

#########################filter by month


filt = (df['RPRL_RGPT_ID_ESP'] == 5034) & (df['LOAD_DATE'].dt.to_period('M')== '2019-03')
user = df[filt]


# ID 6000000249 by hour
user.set_index('LOAD_DATE')[['SUM(RECONCILED_USAGE)']].plot(subplots=True)



#------------------------------backcasting------------------------------------------#

df = pd.read_csv('./BACKCAST_SPLR_LOADS_BY_CLASS_2019.csv')
print(df.head()) #7 columns, including the Date.

#change to date
df['LOAD_DATE'] = pd.to_datetime(df['LOAD_DATE'], infer_datetime_format= True)

df.set_index('LOAD_DATE')[['SUM(RECONCILED_USAGE)']].plot(subplots=True)


##################filter nach ID 
filt = (df['RPRL_RGPT_ID_ESP'] == 5180)
user = df[filt]


# ID 5034 monthly USAGE
user.set_index('LOAD_DATE')[['SUM(RECONCILED_USAGE)']].plot(subplots=True)



#----------------------getTheDate/holiyday, normladay-------------------------------#


