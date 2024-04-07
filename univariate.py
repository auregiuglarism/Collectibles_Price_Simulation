import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import preprocessing   
from statsmodels.tsa.stattools import kpss, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# TODO : Select ARIMA Parameters using ACF and PACF

def moving_average_smooth(df, window_size):
    moving_avg = df.rolling(window=window_size).mean()
    return moving_avg

def is_stationary_with_ADF(data, significance_level=0.05):
    # We want to reject the null hypothesis for the data to be stationary
    adf_test = adfuller(data, regression='c', autolag='BIC')
    print(f"ADF Test Statistic: {adf_test[0]}")
    print(f"P-value: {adf_test[1]}")
    print("Critical Values: \n", adf_test[4])
    return adf_test[1] < significance_level

def is_stationary_with_KPSS(data, significance_level=0.05):
    # We want to FAIL to reject the null hypothesis for the data to be stationary
    kpss_stat, p_value, lags, crit_values  = kpss(data, regression='c')
    print(f"KPSS Test Statistic: {kpss_stat}")
    print(f"P-value: {p_value}")
    print("Critical Values: \n", crit_values)
    return p_value > significance_level

##### MAIN #####

## Load the data from global pre-processing.py ##

# Data is adjusted for inflation and decomposed into trend, seasonality and residuals
wine_df_decomp, watch_df_decomp, art_dfdecomp = preprocessing.main(univariate=True)
# print(wine_df.head(10))
# print(wine_df.tail(10))

## Data is non-stationary, so we apply first order differencing ##
wine_df_diff = wine_df_decomp.observed.diff().dropna()
watch_df_diff = watch_df_decomp.observed.diff().dropna()
art_df_diff = art_dfdecomp.observed.diff().dropna()

# NB The data exhibits WAY better stationary after first order differencing
# Might need moving average to smoothen big spikes
wine_df_smooth = moving_average_smooth(wine_df_diff, 30).dropna() # First 30 days are NaN
watch_df_smooth = moving_average_smooth(watch_df_diff, 30).dropna() # First 30 days are NaN
art_df_smooth = moving_average_smooth(art_df_diff, 30).dropna() # First 30 days are NaN

# Smoothing the data with a 30 day moving average messes (for some reason) the stationarity of the data.
# Increasing the window size makes it worse.

## Evaluating stationarity of the transformed (not smoothed) data using KPSS and ADF tests ##
# Wine
# stationary = is_stationary_with_KPSS(wine_df_diff, significance_level=0.05)
# print(f"Is the data stationary according to the KPSS Test? {stationary}") # True
# stationary = is_stationary_with_ADF(wine_df_diff, significance_level=0.05)
# print(f"Is the data stationary according to the ADF Test? {stationary}") # True

# Watch
# stationary = is_stationary_with_KPSS(watch_df_diff, significance_level=0.05)
# print(f"Is the data stationary according to the KPSS Test? {stationary}") # True
# stationary = is_stationary_with_ADF(watch_df_diff, significance_level=0.05)
# print(f"Is the data stationary according to the ADF Test? {stationary}") # True

# Art
# stationary = is_stationary_with_KPSS(art_df_diff, significance_level=0.05)
# print(f"Is the data stationary according to the KPSS Test? {stationary}") # True
# stationary = is_stationary_with_ADF(art_df_diff, significance_level=0.05)
# print(f"Is the data stationary according to the ADF Test? {stationary}") # True

# Data is stationary after first order differencing

## ACF and PACF plots to determine ARIMA parameters ##
fig = plot_acf(wine_df_diff, color = "blue", lags=50)
plt.title('Wine Index ACF 50 lags')
plt.show()

fig = plot_acf(watch_df_diff, color = "blue", lags=50)
plt.title('Watch Index ACF 50 lags')
plt.show()  

fig = plot_acf(art_df_diff, color = "blue", lags=50)
plt.title('Art Index ACF 50 lags')
plt.show()

fig = plot_pacf(wine_df_diff, color = "green", lags=50)
plt.title('Wine Index PACF 50 lags')
plt.show()

fig = plot_pacf(watch_df_diff, color = "green", lags=50)
plt.title('Watch Index PACF 50 lags')
plt.show()

fig = plot_pacf(art_df_diff, color = "green", lags=50)
plt.title('Art Index PACF 50 lags')
plt.show()




##### VISUALIZATION PLOTS #####

## Plotting Differenced Data ##
# plt.plot(wine_df_diff)
# plt.title('Wine Index First Order Differenced')
# plt.xlabel('Time')
# plt.ylabel('Absolute Change in Index')
# plt.xticks([0, len(wine_df_diff)/2, len(wine_df_diff)-1])
# plt.show()

# plt.plot(watch_df_diff)
# plt.title('Watch Index First Order Differenced')
# plt.xlabel('Time')
# plt.ylabel('Absolute Change in Index')
# plt.xticks([0, len(watch_df_diff)/2, len(watch_df_diff)-1])
# plt.show()

# plt.plot(art_df_diff)
# plt.title('Art Index First Order Differenced')
# plt.xlabel('Time')
# plt.ylabel('Absolute Change in Index')
# plt.xticks([0, len(art_df_diff)/2, len(art_df_diff)-1])
# plt.show()

## Plotting Smoothed Data ##
# plt.plot(wine_df_smooth)
# plt.title('Wine Index Differenced Smoothed')
# plt.xlabel('Time')
# plt.ylabel('Absolute Change in Index (30 Day Moving Average)')
# plt.xticks([0, len(wine_df_smooth)/2, len(wine_df_smooth)-1])
# plt.show()

# plt.plot(watch_df_smooth)
# plt.title('Watch Index Differenced Smoothed')
# plt.xlabel('Time')
# plt.ylabel('Absolute Change in Index (30 Day Moving Average)')
# plt.xticks([0, len(watch_df_smooth)/2, len(watch_df_smooth)-1])
# plt.show()

# plt.plot(art_df_smooth)
# plt.title('Art Index Differenced Smoothed')
# plt.xlabel('Time')
# plt.ylabel('Absolute Change in Index (30 Day Moving Average)')
# plt.xticks([0, len(art_df_smooth)/2, len(art_df_smooth)-1])
# plt.show()













