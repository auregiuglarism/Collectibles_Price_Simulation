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
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import statsmodels.api as sm

# TODO: Implement the rolling window validation method for the optimal (S)ARIMA model for each index

def create_model(train, order, seasonal_order=None):
    if seasonal_order == None: # ARIMA Model
        model = ARIMA(train, trend='n', order=order,  
            enforce_stationarity=True,
            enforce_invertibility=True) 
        
        fit_results = model.fit()
        
    else: # SARIMA Model
        model = ARIMA(train, trend='n', order=order,  
                enforce_stationarity=True,
                enforce_invertibility=True,
                seasonal_order=seasonal_order) 
        
        model.initialize_approximate_diffuse() # Avoid LU Decomposition error when searching for optimal parameters
        
        fit_results = model.fit()

    training_residuals = fit_results.resid

    return fit_results, training_residuals

def test_model(test, model): # Testing data

    # Testing Forecast
    forecast_steps = test.shape[0]
    forecast = model.get_forecast(steps=forecast_steps)
    forecast_ci = forecast.conf_int()
    yhat_test = forecast.predicted_mean # Apply the exp transformation if you used log transform before to invert scales back

    y_test = test
    baseline = np.full(len(y_test), y_test[0])
    baseline_mean = np.full(len(y_test), y_test.mean())

    # Evaluate the model
    mae = mean_absolute_error(y_test, yhat_test)
    mse = mean_squared_error(y_test, yhat_test)
    mae_baseline = mean_absolute_error(y_test, baseline)
    mse_baseline = mean_squared_error(y_test, baseline)
    mae_baseline_mean = mean_absolute_error(y_test, baseline_mean)
    mse_baseline_mean = mean_squared_error(y_test, baseline_mean)
    rmse = np.sqrt(mse)
    rmse_baseline = np.sqrt(mse_baseline)
    rmse_baseline_mean = np.sqrt(mse_baseline_mean)
    mape = np.mean(np.abs((y_test - yhat_test) / y_test)) * 100
    mape_baseline = np.mean(np.abs((y_test - baseline) / y_test)) * 100
    mape_baseline_mean = np.mean(np.abs((y_test - baseline_mean) / y_test)) * 100

    # Plot the results
    # plt.plot(yhat_test, color="green", label="predicted") # Comment this when evaluating multiple models
    # plt.plot(y_test, color="blue", label="observed") # Comment this when evaluating multiple models
    # plt.plot(baseline, color="red", label="baseline") # Comment this when evaluating multiple models 
    # plt.plot(baseline_mean, color="purple", label="mean") # Comment this when evaluating multiple models 
    # plt.legend(loc='best') # Comment this when evaluating multiple models
    # plt.title(f'Compare forecasted and observed {index} index values for test set') # Comment this when evaluating multiple models
    # plt.xticks([0, len(y_test)/2, len(y_test)-1]) # Comment this when evaluating multiple models 
    # plt.xlabel('Time') # Comment this when evaluating multiple models 
    # plt.ylabel('Index value') # Comment this when evaluating multiple models
    # plt.show() # Comment this when evaluating multiple models

    return yhat_test, mae, mse, mae_baseline, mse_baseline, mae_baseline_mean, mse_baseline_mean, rmse, rmse_baseline, rmse_baseline_mean, mape, mape_baseline, mape_baseline_mean

def rolling_window_forecast(data, train, test, model_order, window_size, forecast_length, end_date, index='wine', seasonal_order=None):
    history = [x for x in train]
    forecast = []

    for i in range(int(forecast_length/window_size)):

        model,_ = create_model(history, model_order, seasonal_order) # Leave seasonal_order as None for ARIMA in rolling window function call

        output = model.get_forecast(steps=window_size)
        output_forecast = output.predicted_mean

        # Update the training data
        forecast = np.append(forecast, output_forecast)
        history = np.append(history, output_forecast)

    if index=='wine' or index=='wine_residuals':
        x_axis = pd.date_range(start=data.index[0], end=data.index[-1], freq = 'M')
        x_axis_forecast = pd.date_range(start=test.index[0], end = end_date, freq = 'M')

    elif index=='watch' or index=='watch_residuals':
        x_axis = pd.date_range(start=data.index[0], end=data.index[-1], freq = 'MS')
        x_axis_forecast = pd.date_range(start=test.index[0], end = end_date, freq = 'MS')

    else: # index=='art' or index=='art_residuals'
        x_axis = pd.date_range(start=data.index[0], end=data.index[-1], freq = 'MS')
        x_axis_forecast = pd.date_range(start=test.index[0], end = end_date, freq = 'MS')

    if len(x_axis_forecast) > len(forecast): # Window size sometimes does not exactly finish at the end of the data
        x_axis_forecast = x_axis_forecast[:len(forecast)]

    plt.plot(x_axis, data.values, color="blue", label="observed data")
    plt.plot(x_axis_forecast, forecast, color="red", label="rolling-window forecast", linestyle="--")
    plt.legend(loc='best')
    plt.title(f'Long term forecast of {index} index values')
    plt.xlabel('Time')
    plt.ylabel('Index value')
    plt.show()

def rolling_window_validation(data, model_order, window_size, eval_df, seasonal_order=None):
    # NB: Make sure that window_size is an integer
    mae_l = []
    mse_l = []
    rmse_l = []
    mape_l = []
    aic_l = []
    bic_l = []

    mae_l_bas = []
    mse_l_bas = []
    rmse_l_bas = []
    mape_l_bas = []

    mae_l_mean = []
    mse_l_mean = []
    rmse_l_mean = []
    mape_l_mean = []

    start_train = data[:window_size]
    history = [x for x in start_train]
    test = data[len(history):(len(history)+window_size)]

    end = False

    for i in range(int(len(data)/window_size)):
        model,_ = create_model(history, model_order, seasonal_order) # Leave seasonal_order as None for ARIMA in rolling window function call

        if end == True:
            break

        _, mae, mse, mae_baseline, mse_baseline, mae_baseline_mean, mse_baseline_mean, rmse, rmse_baseline, rmse_baseline_mean, mape, mape_baseline, mape_baseline_mean = test_model(test, model)

        # Model Evaluation Metrics
        mae_l.append(mae)
        mse_l.append(mse)
        rmse_l.append(rmse)
        mape_l.append(mape)
        aic_l.append(model.aic)
        bic_l.append(model.bic)

        # Baseline Evaluation Metrics
        mae_l_bas.append(mae_baseline)
        mse_l_bas.append(mse_baseline)
        rmse_l_bas.append(rmse_baseline)
        mape_l_bas.append(mape_baseline)

        # Mean Evaluation Metrics
        mae_l_mean.append(mae_baseline_mean)
        mse_l_mean.append(mse_baseline_mean)
        rmse_l_mean.append(rmse_baseline_mean)
        mape_l_mean.append(mape_baseline_mean)

        history = np.append(history, test)

        if (len(history) + window_size) > len(data) and len(history) < len(data):
            test = data[len(history):len(data)]
            end = True

        elif len(history) >= len(data):
            end = True

        else:
            test = data[len(history):(len(history)+window_size)]

    eval_df.loc[len(eval_df)] = [model_order, None, np.mean(aic_l), np.mean(bic_l), np.mean(mae_l), np.mean(mse_l), np.mean(rmse_l), np.mean(mape_l)]

    print("MAE Baseline:", np.mean(mae_l_bas))
    print("MSE Baseline:", np.mean(mse_l_bas))
    print("RMSE Baseline:", np.mean(rmse_l_bas))
    print("MAPE % Baseline:", np.mean(mape_l_bas))
    print("MAE Mean:", np.mean(mae_l_mean))
    print("MSE Mean:", np.mean(mse_l_mean))
    print("RMSE Mean:", np.mean(rmse_l_mean))
    print("MAPE % Mean:", np.mean(mape_l_mean))

    return eval_df
        
##### MAIN #####

## Load the data from global pre-processing.py ##

# Data is adjusted for inflation and decomposed into trend, seasonality and residuals
wine_df_decomp, watch_df_decomp, art_df_decomp = preprocessing.main(univariate=True)

### (S)ARIMA (p,d,q)*(P,D,Q)m Model Forecasting (Third Method) Rolling-Window ####

# WINE
wine_train = wine_df_decomp.observed[:int(0.8*len(wine_df_decomp.observed))]
wine_test = wine_df_decomp.observed[int(0.8*len(wine_df_decomp.observed)):]

arima_wine = (3,1,3) # We already know from previous code that this is the optimal ARIMA

# eval_df = pd.DataFrame(columns=['ARIMA', 'Seasonal Order', 'AIC', 'BIC', 'MAE', 'MSE', 'RMSE', 'MAPE %'])
# eval_df = rolling_window_validation(wine_df_decomp.observed, arima_wine, 10, eval_df)
# print(eval_df)

long_term = wine_train.shape[0] # Full training set can go beyond that but it would be extrapolation, so less reliable
ref_start = wine_df_decomp.observed.index[-1] # "2023-12-31"
end_long = "2037-06-30"

window = 10
# rolling_window_forecast(wine_df_decomp.observed, wine_train, wine_test, arima_wine, window, long_term, end_long, index='wine')

# WATCH
watch_train = watch_df_decomp.observed[:int(0.8*len(watch_df_decomp.observed))]
watch_test = watch_df_decomp.observed[int(0.8*len(watch_df_decomp.observed)):]

arima_watch = (2,1,3) # We already know from previous code that this is the optimal ARIMA

# eval_df = pd.DataFrame(columns=['ARIMA', 'Seasonal Order', 'AIC', 'BIC', 'MAE', 'MSE', 'RMSE', 'MAPE %'])
# eval_df = rolling_window_validation(watch_df_decomp.observed, arima_watch, 10, eval_df)
# print(eval_df)

long_term = watch_train.shape[0] # Full training set can go beyond that but it would be extrapolation, so less reliable
ref_start = watch_df_decomp.observed.index[-1] # "2023-12-01"
end_long = "2034-02-01"

window = 10
# rolling_window_forecast(watch_df_decomp.observed, watch_train, watch_test, arima_watch, window, long_term, end_long, index='watch')

# ART
art_train = art_df_decomp.observed[:int(0.8*len(art_df_decomp.observed))]
art_test = art_df_decomp.observed[int(0.8*len(art_df_decomp.observed)):]

arima_art = (13,1,6) # We already know from previous code that this is the optimal ARIMA
sarima_art = [(4,1,2),(5,0,6,6)]

# eval_df = pd.DataFrame(columns=['ARIMA', 'Seasonal Order', 'AIC', 'BIC', 'MAE', 'MSE', 'RMSE', 'MAPE %'])
# eval_df = rolling_window_validation(art_df_decomp.observed, sarima_art[0], 10, eval_df, seasonal_order=sarima_art[1])
# print(eval_df)

long_term = art_train.shape[0] # Full training set can go beyond that but it would be extrapolation, so less reliable
ref_start = art_df_decomp.observed.index[-1] # "2023-09-01"
end_long = "2051-02-01"

window = 20
# rolling_window_forecast(art_df_decomp.observed, art_train, art_test, sarima_art[0], window, long_term, end_long, index='art', seasonal_order=sarima_art[1])



    

