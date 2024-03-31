import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import preprocessing   
from statsmodels.tsa.stattools import kpss, adfuller

# TODO : Implement moving average smoothing to remove spikes
# TODO : Evaluate the stationarity of the data KPSS and ADF tests after additional pre-processing(AGAIN)
# TODO : Select ARIMA Parameters using ACF and PACF

def moving_average_smooth(df, window_size):
    moving_avg = df.rolling(window=window_size).mean()
    return moving_avg

##### MAIN #####

## Load the data from global pre-processing.py ##

# Data is adjusted for inflation and decomposed into trend, seasonality and residuals
wine_df_decomp, watch_df_decomp, art_dfdecomp = preprocessing.main(univariate=True)
# print(wine_df.head(10))
# print(wine_df.tail(10))

## Data is non-stationary, so we apply first order differencing
wine_df_diff = wine_df_decomp.observed.diff().dropna()
watch_df_diff = watch_df_decomp.observed.diff().dropna()
art_df_diff = art_dfdecomp.observed.diff().dropna()

# Plotting Differenced Data 
plt.plot(wine_df_diff)
plt.title('Wine Index First Order Differenced')
plt.xlabel('Time')
plt.ylabel('Absolute Change in Index')
plt.xticks([0, len(wine_df_diff)/2, len(wine_df_diff)-1])
plt.show()

plt.plot(watch_df_diff)
plt.title('Watch Index First Order Differenced')
plt.xlabel('Time')
plt.ylabel('Absolute Change in Index')
plt.xticks([0, len(watch_df_diff)/2, len(watch_df_diff)-1])
plt.show()

plt.plot(art_df_diff)
plt.title('Art Index First Order Differenced')
plt.xlabel('Time')
plt.ylabel('Absolute Change in Index')
plt.xticks([0, len(art_df_diff)/2, len(art_df_diff)-1])
plt.show()

# NB The data exhibits WAY better stationary after first order differencing
# Might need moving average to smoothen big spikes











