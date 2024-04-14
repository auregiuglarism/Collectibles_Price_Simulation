import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import preprocessing   
from statsmodels.tsa.stattools import kpss, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMAResults
from sklearn.metrics import mean_squared_error, mean_absolute_error

# TODO : Justify the choice of SARIMA parameters for each asset in the report

##### PREPROCESSING #####

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

##### MODELS #####

def create_SARIMA_wine(wine_train):
    model = ARIMA(wine_train, trend='n', order=(1,1,0),  # MA here does not change anything as expected
            enforce_stationarity=True,
            enforce_invertibility=False, # this param inverts the fit which isn't good for our data
            seasonal_order=(0,1,1,53)) # A large seasonal order to account to capture subtle seasonality and complex pattern of the data.

    fit_results = model.fit()
    print(fit_results.summary())
    fit_results.save('models\wine_sarima.pkl')

def test_SARIMA_wine(wine_test): # Testing data
    wine_model = ARIMAResults.load('models\wine_sarima.pkl')

    # Testing Forecast
    forecast_steps = wine_test.shape[0]
    forecast = wine_model.get_forecast(steps=forecast_steps)
    forecast_ci = forecast.conf_int()
    yhat_test = forecast.predicted_mean.values # Apply the exp transformation if you used log transform before to invert scales back

    y_test = wine_test
    baseline = np.full(len(y_test), y_test[0])
    baseline_mean = np.full(len(y_test), y_test.mean())

    # Evaluate the model
    mae = mean_absolute_error(y_test, yhat_test)
    mse = mean_squared_error(y_test, yhat_test)
    mae_baseline = mean_absolute_error(y_test, baseline)
    mse_baseline = mean_squared_error(y_test, baseline)
    mae_baseline_mean = mean_absolute_error(y_test, baseline_mean)
    mse_baseline_mean = mean_squared_error(y_test, baseline_mean)
    print("WINE MAE SARIMA (test): {:0.1f}".format(mae))
    print("WINE MSE SARIMA (test): {:0.1f}".format(mse))
    print("WINE MAE Baseline (test): {:0.1f}".format(mae_baseline))
    print("WINE MSE Baseline (test): {:0.1f}".format(mse_baseline))
    print("WINE MAE Baseline Mean (test): {:0.1f}".format(mae_baseline_mean))
    print("WINE MSE Baseline Mean (test): {:0.1f}".format(mse_baseline_mean))


    # Plot the results
    plt.plot(yhat_test, color="green", label="predicted")
    plt.plot(y_test, color="blue", label="observed")
    plt.plot(baseline, color="red", label="baseline")
    plt.plot(baseline_mean, color="purple", label="mean")
    plt.legend(loc='best')
    plt.title('Compare Forecasted and Observed Wine Index Values for Test Set')
    plt.xticks([0, len(y_test)/2, len(y_test)-1])
    plt.xlabel('Time')
    plt.ylabel('Index Value')
    plt.show()

def forecast_SARIMA_wine(wine_data, wine_train, wine_test, forecast_steps, length, end_date):
    wine_model = ARIMAResults.load('models\wine_sarima.pkl')
    forecast = wine_model.get_forecast(steps=forecast_steps)
    forecast_ci = forecast.conf_int()
    yhat = forecast.predicted_mean.values # Apply the exp transformation if you used log transform during fit before to invert scales back

    x_axis = pd.date_range(start=wine_data.index[0], end=wine_data.index[-1], freq = 'M')
    x_axis_forecast = pd.date_range(start=wine_test.index[0], end = end_date, freq = 'M')

    plt.plot(x_axis, wine_data.values, color="blue", label="observed data")
    plt.plot(x_axis_forecast, yhat, color="red", label="forecast", linestyle="--")
    plt.legend(loc='best')
    plt.title(f'{length} term forecast of wine index values')
    plt.xlabel('Time')
    plt.ylabel('Index Value')
    plt.show()
    

def create_SARIMA_watch(watch_train):
    model = ARIMA(watch_train, trend='n', order=(1,1,0), # MA here does not change anything as expected
            enforce_stationarity=True,
            enforce_invertibility=False, # This param inverts the fit and makes us hover just above baseline
            seasonal_order=(0,1,1,35)) # A large seasonal order to capture subtle seasonality and complex pattern of the data.

    fit_results = model.fit()
    print(fit_results.summary())
    fit_results.save('models\watch_sarima.pkl')

def test_SARIMA_watch(watch_test): # Testing data
    watch_model = ARIMAResults.load('models\watch_sarima.pkl')

    # Testing Forecast
    forecast_steps = watch_test.shape[0]
    forecast = watch_model.get_forecast(steps=forecast_steps)
    forecast_ci = forecast.conf_int()
    yhat_test = forecast.predicted_mean.values # Apply the exp transformation if you used log transform before to invert scales back

    y_test = watch_test
    baseline = np.full(len(y_test), y_test[0])
    baseline_mean = np.full(len(y_test), y_test.mean())

    # Evaluate the model
    mae = mean_absolute_error(y_test, yhat_test)
    mse = mean_squared_error(y_test, yhat_test)
    mae_baseline = mean_absolute_error(y_test, baseline)
    mse_baseline = mean_squared_error(y_test, baseline)
    mae_baseline_mean = mean_absolute_error(y_test, baseline_mean)
    mse_baseline_mean = mean_squared_error(y_test, baseline_mean)
    print("WATCH MAE SARIMA (test): {:0.1f}".format(mae))
    print("WATCH MSE SARIMA (test): {:0.1f}".format(mse))
    print("WATCH MAE Baseline (test): {:0.1f}".format(mae_baseline))
    print("WATCH MSE Baseline (test): {:0.1f}".format(mse_baseline))
    print("WATCH MAE Baseline Mean (test): {:0.1f}".format(mae_baseline_mean))
    print("WATCH MSE Baseline Mean (test): {:0.1f}".format(mse_baseline_mean))

    # Plot the results
    plt.plot(yhat_test, color="green", label="predicted")
    plt.plot(y_test, color="blue", label="observed")
    plt.plot(baseline, color="red", label="baseline")
    plt.plot(baseline_mean, color="purple", label="mean")
    plt.legend(loc='best')
    plt.title('Compare Forecasted and Observed Watch Index Values for Test Set')
    plt.xticks([0, len(y_test)/2, len(y_test)-1])
    plt.xlabel('Time')
    plt.ylabel('Index Value')
    plt.show()

def forecast_SARIMA_watch(watch_data, watch_train, watch_test, forecast_steps, length, end_date):
    watch_model = ARIMAResults.load('models\watch_sarima.pkl')
    forecast = watch_model.get_forecast(steps=forecast_steps)
    forecast_ci = forecast.conf_int()
    yhat = forecast.predicted_mean.values # Apply the exp transformation if you used log transform during fit before to invert scales back

    x_axis = pd.date_range(start=watch_data.index[0], end=watch_data.index[-1], freq = 'MS')
    x_axis_forecast = pd.date_range(start=watch_test.index[0], end = end_date, freq = 'MS')

    plt.plot(x_axis, watch_data.values, color="blue", label="observed data")
    plt.plot(x_axis_forecast, yhat, color="red", label="forecast", linestyle="--")
    plt.legend(loc='best')
    plt.title(f'{length} term forecast of watch index values')
    plt.xlabel('Time')
    plt.ylabel('Index Value')
    plt.show()

def create_SARIMA_art(art_train):
    model = ARIMA(art_train, trend='n', order=(1,1,1), # Correlogram indicates the need for MA and AR.
            enforce_stationarity=True, 
            enforce_invertibility=True, # Invertibility is necessary since the MA component is active
            seasonal_order=(0,1,1,42)) # A large seasonal order to capture subtle seasonality and complex pattern of the data.
    
    fit_results = model.fit()
    print(fit_results.summary())
    fit_results.save(f"models/art_sarima.pkl")

def test_SARIMA_art(art_test): # Testing data
    art_model = ARIMAResults.load(f'models/art_sarima.pkl')

    forecast_steps = art_test.shape[0]
    forecast = art_model.get_forecast(steps=forecast_steps)
    forecast_ci = forecast.conf_int()
    yhat_test = forecast.predicted_mean.values # Apply the exp transformation if you used log transform before to invert scales back

    y_test = art_test
    baseline = np.full(len(y_test), y_test[0])
    baseline_mean = np.full(len(y_test), y_test.mean())

    # Evaluate the model
    mae = mean_absolute_error(y_test, yhat_test)
    mse = mean_squared_error(y_test, yhat_test)
    mae_baseline = mean_absolute_error(y_test, baseline)
    mse_baseline = mean_squared_error(y_test, baseline)
    mae_baseline_mean = mean_absolute_error(y_test, baseline_mean)
    mse_baseline_mean = mean_squared_error(y_test, baseline_mean)
    print("ART MAE SARIMA (test): {:0.1f}".format(mae))
    print("ART MSE SARIMA (test): {:0.1f}".format(mse))
    print("ART MAE Baseline (test): {:0.1f}".format(mae_baseline))
    print("ART MSE Baseline (test): {:0.1f}".format(mse_baseline))
    print("ART MAE Baseline Mean (test): {:0.1f}".format(mae_baseline_mean))
    print("ART MSE Baseline Mean (test): {:0.1f}".format(mse_baseline_mean))

    # Plot the results
    plt.plot(yhat_test, color="green", label="predicted")
    plt.plot(y_test, color="blue", label="observed")
    plt.plot(baseline, color="red", label="baseline")
    plt.plot(baseline_mean, color="purple", label="mean")
    plt.legend(loc='best')
    plt.title('Compare Forecasted and Observed Art Index Values for Test Set')
    plt.xticks([0, len(y_test)/2, len(y_test)-1])
    plt.xlabel('Time')
    plt.ylabel('Index Value')
    plt.show()

def forecast_SARIMA_art(art_data, art_train, art_test, forecast_steps, length, end_date):
    art_model = ARIMAResults.load(f'models/art_sarima.pkl')
    forecast = art_model.get_forecast(steps=forecast_steps)
    forecast_ci = forecast.conf_int()
    yhat = forecast.predicted_mean.values # Apply the exp transformation if you used log transform during fit before to invert scales back

    x_axis = pd.date_range(start=art_data.index[0], end=art_data.index[-1], freq = 'MS')
    x_axis_forecast = pd.date_range(start=art_test.index[0], end = end_date, freq = 'MS')

    plt.plot(x_axis, art_data.values, color="blue", label="observed data")
    plt.plot(x_axis_forecast, yhat, color="red", label="forecast", linestyle="--")
    plt.legend(loc='best')
    plt.title(f'{length} term forecast of art index values')
    plt.xlabel('Time')
    plt.ylabel('Index Value')
    plt.show()

##### MAIN #####

## Load the data from global pre-processing.py ##

# Data is adjusted for inflation and decomposed into trend, seasonality and residuals
wine_df_decomp, watch_df_decomp, art_dfdecomp = preprocessing.main(univariate=True)

## Evaluating stationarity and the (S)ARIMA Parameters ##

# # Data is non-stationary, so we apply first order differencing
# wine_df_diff = wine_df_decomp.observed.diff().dropna()
# watch_df_diff = watch_df_decomp.observed.diff().dropna()
# art_df_diff = art_dfdecomp.observed.diff().dropna()

# # NB The data exhibits WAY better stationary after first order differencing
# # Might need moving average to smoothen big spikes
# wine_df_smooth = moving_average_smooth(wine_df_diff, 30).dropna() # First 30 days are NaN
# watch_df_smooth = moving_average_smooth(watch_df_diff, 30).dropna() # First 30 days are NaN
# art_df_smooth = moving_average_smooth(art_df_diff, 30).dropna() # First 30 days are NaN

# Smoothing the data with a 30 day moving average messes (for some reason) the stationarity of the data.
# Increasing the window size makes it worse.

# Evaluating stationarity of the transformed (not smoothed) data using KPSS and ADF tests 
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

## SARIMA (p,q,d)*(P,D,Q) Model Forecasting ##

# The significant lags in the ACF and PACF at lag 1 indicate the need for AR in all three assets.
# In the ART index, there are some significant lags at multiple intervals indicating the need for MA as well
# The ARIMA model will be (1,1,0) for the Wine and Watch indices and (1,1,1) for the Art index
# First order differencing makes the data stationary so I will set my d = 1

# WINE INDEX DATA FORECASTING
# Split data into train and test
wine_train = wine_df_decomp.observed[:int(0.8*len(wine_df_decomp.observed))]
wine_test = wine_df_decomp.observed[int(0.8*len(wine_df_decomp.observed)):]

# Create (S)ARIMA model
# create_SARIMA_wine(wine_train) # Only run once

# Test (S)ARIMA model
# test_SARIMA_wine(wine_test)

# Now that model is trained + evaluated, use it to forecast
short_term = wine_test.shape[0] + 12 # 1 year
medium_term = wine_test.shape[0] + 12*5 # 5 years
long_term = wine_train.shape[0] # Full training set can go beyond that but it would be extrapolation, so less reliable

# Short, medium and long term forecasts
ref_start = wine_df_decomp.observed.index[-1] # "2023-12-31"
end_short = "2024-12-31"
end_medium = "2028-12-31"
end_long = "2037-06-30"
# forecast_SARIMA_wine(wine_df_decomp.observed, wine_train, wine_test, long_term, "Long", end_date=end_long)

# WATCH INDEX DATA FORECASTING
# Split data into train and test
watch_train = watch_df_decomp.observed[:int(0.8*len(watch_df_decomp.observed))]
watch_test = watch_df_decomp.observed[int(0.8*len(watch_df_decomp.observed)):]

# Create (S)ARIMA model
# create_SARIMA_watch(watch_train) # Only run once

# Test (S)ARIMA model
# test_SARIMA_watch(watch_test)

# Now that model is trained + evaluated, use it to forecast
short_term = watch_test.shape[0] + 12 # 1 year
medium_term = watch_test.shape[0] + 12*5 # 5 years
long_term = watch_train.shape[0] # Full training set can go beyond that but it would be extrapolation, so less reliable

# Short, medium and long term forecasts
ref_start = watch_df_decomp.observed.index[-1] # "2023-12-01"
end_short = "2024-12-01"
end_medium = "2028-12-01"
end_long = "2034-02-01"
# forecast_SARIMA_watch(watch_df_decomp.observed, watch_train, watch_test, long_term, "Long", end_date=end_long)

# ART INDEX DATA FORECASTING
# Split data into train and test
art_train = art_dfdecomp.observed[:int(0.8*len(art_dfdecomp.observed))]
art_test = art_dfdecomp.observed[int(0.8*len(art_dfdecomp.observed)):]

# Create (S)ARIMA model
# create_SARIMA_art(art_train) # Only run once

# Test (S)ARIMA model
# test_SARIMA_art(art_test)

# Now that model is trained + evaluated, use it to forecast
short_term = art_test.shape[0] + 12 # 1 year
medium_term = art_test.shape[0] + 12*5 # 5 years
long_term = art_train.shape[0] # Full training set can go beyond that but it would be extrapolation, so less reliable

# Short, medium and long term forecasts
ref_start = art_dfdecomp.observed.index[-1] # "2023-09-01"
end_short = "2024-09-01"
end_medium = "2028-09-01"
end_long = "2051-02-01"
# forecast_SARIMA_art(art_dfdecomp.observed, art_train, art_test, long_term, "Long", end_date=end_long)
      
##### VISUALIZATION PLOTS #####

# plt.plot(wine_df_decomp.observed)
# plt.title('Wine Index')
# plt.xlabel('Time')
# plt.ylabel('Index Value')
# plt.xticks([0, len(wine_df_decomp.observed)/2, len(wine_df_decomp.observed)-1])
# plt.show()

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

# Data is stationary after first order differencing

# ## ACF and PACF plots to determine ARIMA parameters ##
# fig = plot_acf(wine_df_diff, color = "blue", lags=50)
# plt.title('Wine Index ACF 50 lags')
# plt.show()

# fig = plot_acf(watch_df_diff, color = "blue", lags=50)
# plt.title('Watch Index ACF 50 lags')
# plt.show()  

# fig = plot_acf(art_df_diff, color = "blue", lags=50)
# plt.title('Art Index ACF 50 lags')
# plt.show()

# fig = plot_pacf(wine_df_diff, color = "green", lags=50)
# plt.title('Wine Index PACF 50 lags')
# plt.show()

# fig = plot_pacf(watch_df_diff, color = "green", lags=50)
# plt.title('Watch Index PACF 50 lags')
# plt.show()

# fig = plot_pacf(art_df_diff, color = "green", lags=50)
# plt.title('Art Index PACF 50 lags')
# plt.show()
















