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
import statsmodels.api as sm

# TODO : Justify the choice of SARIMA parameters for each asset in the report
# TODO : Continue Fine-tuning Parameters

# Links to understand more about SARIMA Parameters : 

# https://en.wikipedia.org/w<iki/Box%E2%80%93Jenkins_method
# https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
# Paper to cite : https://otexts.com/fpp2/non-seasonal-arima.html#acf-and-pacf-plots

# https://medium.com/latinxinai/time-series-forecasting-arima-and-sarima-450fb18a9941
# https://neptune.ai/blog/arima-sarima-real-world-time-series-forecasting-guide
# https://towardsdev.com/time-series-forecasting-part-5-cb2967f18164
# https://www.geeksforgeeks.org/box-jenkins-methodology-for-arima-models/

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

def is_white_noise_with_LjungBox(data, significance_level=0.05):
    # We want to FAIL to reject the null hypothesis for the data to be white noise
    # Null Hypothesis : The residuals are independently distributed
    # Alternative Hypothesis : The residuals are not independently distributed
    # If p-value < 0.05, reject the null hypothesis thus we want to see a p-value > 0.05
    df_ljungbox = sm.stats.acorr_ljungbox(data, lags=[50], return_df=True)
    print(df_ljungbox)
    return df_ljungbox.loc[50,"lb_pvalue"] > significance_level
    
##### MODELS #####

def create_ARIMA_wine(wine_train, order):
    model = ARIMA(wine_train, trend='n', order=order,  
            enforce_stationarity=True,
            enforce_invertibility=False) 
            
    fit_results = model.fit()
    # print(fit_results.summary()) # Comment this when evaluating multiple models
    
    # fit_results.save('models\wine_arima.pkl') # Comment this when evaluating multiple models
    return fit_results

def test_ARIMA_wine(wine_test, wine_model=None): # Testing data
    if wine_model == None:
        wine_model = ARIMAResults.load('models\wine_arima.pkl')

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
    # print("WINE MAE ARIMA (test): {:0.1f}".format(mae)) # Comment this when evaluating multiple models
    # print("WINE MSE ARIMA (test): {:0.1f}".format(mse)) # Comment this when evaluating multiple models
    # print("WINE MAE Baseline (test): {:0.1f}".format(mae_baseline)) # Comment this when evaluating multiple models
    # print("WINE MSE Baseline (test): {:0.1f}".format(mse_baseline)) # Comment this when evaluating multiple models
    # print("WINE MAE Baseline Mean (test): {:0.1f}".format(mae_baseline_mean)) # Comment this when evaluating multiple models
    # print("WINE MSE Baseline Mean (test): {:0.1f}".format(mse_baseline_mean)) # Comment this when evaluating multiple models

    # Plot the results
    # plt.plot(yhat_test, color="green", label="predicted") # Comment this when evaluating multiple models
    # plt.plot(y_test, color="blue", label="observed") # Comment this when evaluating multiple models
    # plt.plot(baseline, color="red", label="baseline") # Comment this when evaluating multiple models 
    # plt.plot(baseline_mean, color="purple", label="mean") # Comment this when evaluating multiple models 
    # plt.legend(loc='best') # Comment this when evaluating multiple models
    # plt.title('Compare Forecasted and Observed Wine Index Values for Test Set') # Comment this when evaluating multiple models
    # plt.xticks([0, len(y_test)/2, len(y_test)-1]) # Comment this when evaluating multiple models 
    # plt.xlabel('Time') # Comment this when evaluating multiple models 
    # plt.ylabel('Index Value') # Comment this when evaluating multiple models
    # plt.show() # Comment this when evaluating multiple models

    return yhat_test, mae, mse, mae_baseline, mse_baseline, mae_baseline_mean, mse_baseline_mean

def evaluate_ARIMA_wine_with_Plots(wine_train, wine_test, candidates, eval_df,): 
    # Take the model with the lowest eval metrics and errors
    for candidate in candidates:
        # Fit candidate model
        cd_fit_results = create_ARIMA_wine(wine_train, candidate) 
            
        # Test candidate model on test set
        _, cd_mae, cd_mse, mae_bas, mse_bas, mae_mean, mse_mean = test_ARIMA_wine(wine_test, cd_fit_results)

        # Store evaluation information
        eval_df.loc[len(eval_df)] = [candidate, cd_fit_results.aic, cd_fit_results.bic, cd_mae, cd_mse]
        print("MAE Baseline:", mae_bas)
        print("MSE Baseline:", mse_bas)
        print("MAE Mean:", mae_mean)
        print("MSE Mean:", mse_mean)
        
    return eval_df

def evaluate_ARIMA_wine_with_BoxJenkins(wine_train, wine_test, start_cd, eval_df):
    # Create ARIMA Model
    start = start_cd[0]
    fit_results = create_ARIMA_wine(wine_train, start)

    # Test ARIMA Model
    yhat_test, _, _, _, _, _, _ = test_ARIMA_wine(wine_test, fit_results)

    # Get Evaluation Metrics for this model:
    eval_df = evaluate_ARIMA_wine_with_Plots(wine_train, wine_test, start_cd, eval_df)
    print("Model Evaluation Metrics: \n", eval_df)

    # Compute Residuals
    y_test = wine_test
    residuals = y_test - yhat_test

    # Plot Residuals - Does it follow a white noise pattern ?
    plt.plot(residuals, color="blue", label="residuals", linestyle=":")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.legend(loc='best')
    plt.title('Model Residuals on Wine Index Test Set')
    plt.xticks([0, len(residuals)/2, len(residuals)-1])
    plt.xlabel('Time')
    plt.ylabel('Residual Value')
    plt.show()

    # Check ACF and PACF of Residuals
    fig = plot_acf(residuals, color = "blue", lags=50)
    plt.title('Wine Index Model Residuals ACF 50 lags')
    plt.show()

    fig = plot_pacf(residuals, color = "green", lags=26) # PACF cannot be longer than 50% of the data
    plt.title('Wine Index Model Residuals PACF 26 lags')
    plt.show()

    # Perform Ljung-Box Test on Residuals to test if they are white noise/independently distributed
    # Null Hypothesis : The residuals are independently distributed
    # Alternative Hypothesis : The residuals are not independently distributed
    # If p-value < 0.05, reject the null hypothesis thus we want to see a p-value > 0.05

    is_white_noise = is_white_noise_with_LjungBox(residuals, significance_level=0.05)
    print(f"Are the residuals white noise? {is_white_noise}")

def forecast_ARIMA_wine(wine_data, wine_train, wine_test, forecast_steps, length, end_date):
    wine_model = ARIMAResults.load('models\wine_arima.pkl')
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
    

def create_ARIMA_watch(watch_train):
    model = ARIMA(watch_train, trend='n', order=(1,1,0), # MA here does not change anything as expected
            enforce_stationarity=True,
            enforce_invertibility=False, # This param inverts the fit and makes us hover just above baseline
            seasonal_order=(0,1,1,35)) # A large seasonal order to capture subtle seasonality and complex pattern of the data.

    fit_results = model.fit()
    print(fit_results.summary())
    fit_results.save('models\watch_arima.pkl')

def test_ARIMA_watch(watch_test): # Testing data
    watch_model = ARIMAResults.load('models\watch_arima.pkl')

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
    print("WATCH MAE ARIMA (test): {:0.1f}".format(mae))
    print("WATCH MSE ARIMA (test): {:0.1f}".format(mse))
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

def forecast_ARIMA_watch(watch_data, watch_train, watch_test, forecast_steps, length, end_date):
    watch_model = ARIMAResults.load('models\watch_arima.pkl')
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

def create_ARIMA_art(art_train):
    model = ARIMA(art_train, trend='n', order=(1,1,1), # Correlogram indicates the need for MA and AR.
            enforce_stationarity=True, 
            enforce_invertibility=True, # Invertibility is necessary since the MA component is active
            seasonal_order=(0,1,1,42)) # A large seasonal order to capture subtle seasonality and complex pattern of the data.
    
    fit_results = model.fit()
    print(fit_results.summary())
    fit_results.save(f"models/art_arima.pkl")

def test_ARIMA_art(art_test): # Testing data
    art_model = ARIMAResults.load(f'models/art_arima.pkl')

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
    print("ART MAE ARIMA (test): {:0.1f}".format(mae))
    print("ART MSE ARIMA (test): {:0.1f}".format(mse))
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

def forecast_ARIMA_art(art_data, art_train, art_test, forecast_steps, length, end_date):
    art_model = ARIMAResults.load(f'models/art_arima.pkl')
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

## Evaluating stationarity and the ARIMA Parameters ##

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

## SARIMA (p,d,q) Model Forecasting ##

# First order differencing makes the data stationary so I will set my d = 1 as confirmed by ADF + KPSS tests

# WINE INDEX DATA FORECASTING
# Split data into train and test
wine_train = wine_df_decomp.observed[:int(0.8*len(wine_df_decomp.observed))]
wine_test = wine_df_decomp.observed[int(0.8*len(wine_df_decomp.observed)):]
eval_df = pd.DataFrame(columns=["ARIMA", "AIC", "BIC", "MAE", "MSE"]) # To store the most important evaluation metrics

# Evaluate Wine ARIMA model with ACF + PACF plots
# Candidates are chosen based on the ACF and PACF plots
# candidates = [(17,1,0), (3,1,0), (0,1,20), (0,1,12), (0,1,3), (17,1,20)]
# eval_df = evaluate_ARIMA_wine_with_Plots(wine_train, wine_test, candidates, eval_df)
# print(eval_df)

# Best model seems to be (17,1,0) if we want a simpler model 
# (17,1,20) is the best in terms of MAE and MSE, but it is more complex and thus penalized by AIC and BIC
# We still do manage to be better than the baseline and the mean so this at least one success

# However this suggests that the optimal cannot be precisely determined by the ACF and PACF plots 
# Because in this case, both the MA and AR components are active, i.e p,q > 0
# Thus we need to look at the residuals and the Box-Jenkins model diagnostic to determine the optimal model 

# Evaluate Wine ARIMA model with Box-Jenkins model diagnostic
# Starting point : previous best model (17,1,20)
start_cd = [(25,1,1)] 
evaluate_ARIMA_wine_with_BoxJenkins(wine_train, wine_test, start_cd, eval_df)
# The residual of this model (17,1,20) indicate a significant autocorrelation at lag 25 in the PACF
# Thus I will try a model with (25,1,20) to see if we improve the performance
# Best model yet : (25,1,1) or (50,1,1) but penalized by AIC and BIC for slightly better MAE and MSE


# Create optimal ARIMA model
# create_ARIMA_wine(wine_train, eval_df, optimal) # Only run once to save the optimal model

# Now that the optimal has been found, use it to forecast
short_term = wine_test.shape[0] + 12 # 1 year
medium_term = wine_test.shape[0] + 12*5 # 5 years
long_term = wine_train.shape[0] # Full training set can go beyond that but it would be extrapolation, so less reliable

# Short, medium and long term forecasts
ref_start = wine_df_decomp.observed.index[-1] # "2023-12-31"
end_short = "2024-12-31"
end_medium = "2028-12-31"
end_long = "2037-06-30"
# forecast_ARIMA_wine(wine_df_decomp.observed, wine_train, wine_test, long_term, "Long", end_date=end_long)

# WATCH INDEX DATA FORECASTING
# Split data into train and test
watch_train = watch_df_decomp.observed[:int(0.8*len(watch_df_decomp.observed))]
watch_test = watch_df_decomp.observed[int(0.8*len(watch_df_decomp.observed)):]

# Create ARIMA model
# create_ARIMA_watch(watch_train) # Only run once

# Test ARIMA model
# test_ARIMA_watch(watch_test)

# Now that model is trained + evaluated, use it to forecast
short_term = watch_test.shape[0] + 12 # 1 year
medium_term = watch_test.shape[0] + 12*5 # 5 years
long_term = watch_train.shape[0] # Full training set can go beyond that but it would be extrapolation, so less reliable

# Short, medium and long term forecasts
ref_start = watch_df_decomp.observed.index[-1] # "2023-12-01"
end_short = "2024-12-01"
end_medium = "2028-12-01"
end_long = "2034-02-01"
# forecast_ARIMA_watch(watch_df_decomp.observed, watch_train, watch_test, long_term, "Long", end_date=end_long)

# ART INDEX DATA FORECASTING
# Split data into train and test
art_train = art_dfdecomp.observed[:int(0.8*len(art_dfdecomp.observed))]
art_test = art_dfdecomp.observed[int(0.8*len(art_dfdecomp.observed)):]

# Create ARIMA model
# create_ARIMA_art(art_train) # Only run once

# Test ARIMA model
# test_ARIMA_art(art_test)

# Now that model is trained + evaluated, use it to forecast
short_term = art_test.shape[0] + 12 # 1 year
medium_term = art_test.shape[0] + 12*5 # 5 years
long_term = art_train.shape[0] # Full training set can go beyond that but it would be extrapolation, so less reliable

# Short, medium and long term forecasts
ref_start = art_dfdecomp.observed.index[-1] # "2023-09-01"
end_short = "2024-09-01"
end_medium = "2028-09-01"
end_long = "2051-02-01"
# forecast_ARIMA_art(art_dfdecomp.observed, art_train, art_test, long_term, "Long", end_date=end_long)
      
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
















