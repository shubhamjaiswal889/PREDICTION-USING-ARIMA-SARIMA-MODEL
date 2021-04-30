#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv(r'C:\Users\shubham.kj\Downloads\Perrin Freres monthly champagne sales millions.csv')


# In[5]:


df.head()


# In[6]:


## Change the Column Names 
df.columns=["Month","Sales"]
df.head()


# In[7]:


df.tail()


# # Removal of Null values
# 

# In[8]:


## Drop last 2 rows
df.drop(106,axis=0,inplace=True)


# In[9]:


df.drop(105,axis=0,inplace=True)


# In[10]:


# Convert Month into Datetime
df['Month']=pd.to_datetime(df['Month'])


# In[11]:


df.head()


# In[12]:


df.set_index('Month',inplace=True)


# In[13]:


df.head()


# In[14]:


df.describe()


# In[ ]:





# # Sales Visualization

# In[15]:


df.plot()


# In[16]:


from statsmodels.tsa.stattools import adfuller


# adfuller" is a function / module used to check the STATIONARITY in dataset.

# In[17]:


test_result=adfuller(df['Sales'])


# In[18]:


#HYPOTHESIS TEST:
#Ho: It is non stationary
#H1: It is stationary

def adfuller_test(sales):
    
    result=adfuller(sales)
    
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")


# In[19]:


adfuller_test(df['Sales'])


# Differencing is a popular and widely used data transform for making time series data stationary.
# 
# Differencing can help stabilise the mean of a time series by removing changes in the level of a time series, and therefore eliminating (or reducing) trend and seasonality.
# 
# Differencing shifts ONE/MORE row towards downwards.

# In[20]:


df['Seasonal First Difference']=df['Sales']-df['Sales'].shift(12)


# In[21]:


df.head(14)


# In[22]:


## Again test dickey fuller test
adfuller_test(df['Seasonal First Difference'].dropna())


# In[23]:


df['Seasonal First Difference'].plot()


# # AUTO-CORRELATION | PARTIAL AUTO-CORRELATION:

# In[24]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[25]:


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df['Sales'])
plt.show()


# In[26]:


import statsmodels.api as sm
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax2)


# Here these two graphs will help us to find the p and q values.
# Partial AutoCorrelation Graph is for the p-value.
# AutoCorrelation Graph for the q-value

# # ARIMA MODEL

# AR: Autoregression. A model that uses the dependent relationship between an observation and some number of lagged observations.
# 
# I: Integrated. The use of differencing of raw observations in order to make the time series stationary.
# 
# MA: Moving Average. A model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.
# 
# The parameters of the ARIMA model are defined as follows:
# 
# p: The number of lag observations included in the model, also called the lag order.
# d: The number of times that the raw observations are differenced, also called the degree of differencing.
# q: The size of the moving average window, also called the order of moving average.

# In[27]:


# For non-seasonal data
#p=1, d=1, q=0 or 1
from statsmodels.tsa.arima_model import ARIMA


# In[28]:


model=ARIMA(df['Sales'],order=(1,1,1))
model_fit=model.fit()


# In[29]:


model_fit.summary()


# In[30]:


df['forecast']=model_fit.predict(start=90,end=103,dynamic=True)
df[['Sales','forecast']].plot(figsize=(12,8))


# # SARIMA MODEL

# In[31]:


import statsmodels.api as sm


# In[32]:


model=sm.tsa.statespace.SARIMAX(df['Sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
results=model.fit()


# In[33]:


df['forecast']=results.predict(start=90,end=103,dynamic=True)
df[['Sales','forecast']].plot(figsize=(12,8))


# HERE THE BLUE LINE IS ACTUAL DATA & ORANGE LINE IS PREDICTED DATA. HOW GOOD IT GAVE US THE RESULTS

# # PREDICTION

# In[34]:


from pandas.tseries.offsets import DateOffset

#Here USING FOR LOOP we are adding some additional data for prediction purpose:

future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,24)]


# In[35]:


#Converting list into DATAFRAME:

future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)


# In[36]:


future_datest_df.tail()


# In[37]:


#CONCATING THE ORIGINAL AND THE NEWLY CREATED DATASET FOR VISUALIZATION PURPOSE:
future_df=pd.concat([df,future_datest_df])


# In[38]:


#PREDICT
future_df['forecast'] = results.predict(start = 104, end = 120, dynamic= True)  
future_df[['Sales', 'forecast']].plot(figsize=(12, 8))


# In[ ]:




