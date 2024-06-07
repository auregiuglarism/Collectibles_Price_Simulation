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

# TODO: Report : Include a more detailed explanation (since you are the first one) of the indices used in this thesis (art, watch and wine) + sources
# TODO: Report : Include a future work section of what could be done from here on out after my work for future researchers

# Links to understand more about (S)ARIMA Parameters : 

# https://en.wikipedia.org/wiki/Box%E2%80%93Jenkins_method
# https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
# Paper to cite : https://otexts.com/fpp2/non-seasonal-arima.html#acf-and-pacf-plots

# https://medium.com/latinxinai/time-series-forecasting-arima-and-sarima-450fb18a9941
# https://neptune.ai/blog/arima-sarima-real-world-time-series-forecasting-guide
# https://towardsdev.com/time-series-forecasting-part-5-cb2967f18164
# https://www.geeksforgeeks.org/box-jenkins-methodology-for-arima-models/

# https://towardsdatascience.com/understanding-the-seasonal-order-of-the-sarima-model-ebef613e40fa

# https://dsri.maastrichtuniversity.nl/
# https://medium.com/rapids-ai/arima-forecast-large-time-series-datasets-with-rapids-cuml-18428a00d02e
# https://docs.rapids.ai/install#pip

# https://www.sciencedirect.com/science/article/pii/S0925231219309178
# https://stats.stackexchange.com/questions/124955/is-it-unusual-for-the-mean-to-outperform-arima/125016#125016

# https://pdf.sciencedirectassets.com/271689/1-s2.0-S0304407616X00097/1-s2.0-S0304407616301713/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEGEaCXVzLWVhc3QtMSJGMEQCIBGKZ%2BIrCxk2a71fYwONLJuBuxZpZtfv24kUuTvEb29PAiBb3iegDxBGuh8X4ehfJt9anx%2FUEtcdl%2FNaH4uwcouNzSq8BQjZ%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAUaDDA1OTAwMzU0Njg2NSIMzU09aK%2B0FVQpoXWbKpAFCeXs0jdJrJqxc6b7jeNX425Pn%2B64F2xaebXs9Rps%2FuDsumh61e%2BcbfIMrlELbWV%2FFBU3SkZVwT%2BRDSeRE4lQtHXHOZangJWjGwg1rOftvFMjgpKfOYhXZv7VFlw%2BcoWK0m%2BMlbePiO7UIifiPFaamT0d%2Fk2BNSa6bjrRq0Nvx6mFOL%2ByXiJpc5b0AFPCHuvGBJlrfIiT2f4qLgisIXAdW7n0J%2BFLyV7I38A%2BYbW2Ngq5H5nXNoSkyWLe7u8BLfMIOxmknPgL1yS%2BzQX%2FZMVoUuuSCQwOOJLV9NT6CbMaROnyg4rotDrQoWVf0W2cc%2FAMZvbEi0VEi0HiY2YjT3rRb%2F1nrurgAybRlXpw4mDgnhQn7X2udFeD35gcYXx8xGbBPCudYMGrSproLyH%2BjzSoJwZiGk%2B3VCx7GIc8rD0oZ6FLnS9mvm9WBSIDujfJaeekV4y7rJJDimz3xBB4g6CMTw%2FhkeZWDPl9%2FDJ1C9BvudrnWuuKV%2BI9TdHBVTl0jQ24b9zYdPtw77bF0EbKufqxZb89DVxAUlm7rrhg8tu%2BYRGmz0quVjZ1vWgJIgN50RWG6G77F68fOJHRfY5O5upn6VYhLizeyU%2FnbOF1DIPUu5PMxPhDPFD6L6Gj20y9sjFagrJAdj2hSXjHUdHuDPZJciXJ0s91dtd0mWeYVaaB%2FdHvRi1VM5Nw26vCrpEfTxZV%2BiBhcXGpS2A6vwWwDnvWNBkwixlRqu%2FhlEjN3AJoZ%2BgNKtEdUijYoIBUZMyoW5HooBWDctKF7yMq9BjndT%2Br9Q8jqS0Rltx7Dcr%2BD4%2BJvzpkhsgNvEht9m40N7Ui0jmfSTsB6qMq%2FzqoTyP8kHVVmUfJqxVstx%2F7me2eK35GWWUw%2FuPSsgY6sgFMH0RS5c5DfWtaDLYCPRFJ2HnYjUCGohdpMGKjqHGavXjWbMWJ9jTOwsIwWMuqYNFxSENfu9IvcayXs50iFZkip36UbyuOCuiecojmeVto%2FFV%2FD9eINpC5x3VL0hZdaNftEoixpAckcI3f9ClqQpVlKp7lUnNeojR%2FhFrcZNNPd4UDB8eYxr6ybgomQZRbFOIqIkxaC9EWQtmm5JPHncMvDbXJN2x91jTKUO6oKvPHjQPW&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240527T170829Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYXNRMRAGR%2F20240527%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=5d1945dc2162a6a63dfd7fe1b86be1e1caca26b5217700889024b309bf81676a&hash=2afe5ca0247f8853be7fd4f9a7bfc146f4e634e2ea96fa779afa38a5770a7611&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0304407616301713&tid=spdf-034c920f-5125-43b5-84c0-83f620cc22eb&sid=87600c4d5d9e354ccd9b4a35cecdd1abe9a8gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=140a5c5f02585b00525558&rr=88a79b747c909fe2&cc=nl
# https://www.bing.com/search?pglt=41&q=olling-window+forecasting+approach+when+using+ARIMA&cvid=c360b87664694dfeb3f892b2d89e6f31&gs_lcrp=EgZjaHJvbWUyBggAEEUYOdIBCDE5NTlqMGoxqAIAsAIA&FORM=ANNTA1&adppc=EDGEXST&PC=ASTS
# https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/

##### STATISTICAL TESTS #####

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

def is_white_noise_with_LjungBox(data, significance_level=0.05, lags=50):
    # We want to FAIL to reject the null hypothesis for the data to be white noise
    # Null Hypothesis : The residuals are independently distributed
    # Alternative Hypothesis : The residuals are not independently distributed
    # If p-value < 0.05, reject the null hypothesis thus we want to see a p-value > 0.05
    df_ljungbox = sm.stats.acorr_ljungbox(data, lags=[lags], return_df=True)
    print(df_ljungbox)
    return df_ljungbox.loc[lags,"lb_pvalue"] > significance_level
    
##### MODELS #####

def create_model(train, order, seasonal_order=None, index='wine'):
    if seasonal_order == None: # ARIMA Model
        model = ARIMA(train, trend='n', order=order,  
            enforce_stationarity=True,
            enforce_invertibility=True) 
        
        fit_results = model.fit()
        fit_results.save(f'models\{index}_arima.pkl') # Comment this when evaluating multiple models
        
    else: # SARIMA Model
        model = ARIMA(train, trend='n', order=order,  
                enforce_stationarity=True,
                enforce_invertibility=True,
                seasonal_order=seasonal_order) 
        
        model.initialize_approximate_diffuse() # Avoid LU Decomposition error when searching for optimal parameters
        
        fit_results = model.fit()
        fit_results.save(f'models\{index}_sarima.pkl') # Comment this when evaluating multiple models

    # print(fit_results.summary()) # Comment this when evaluating multiple models
    training_residuals = fit_results.resid

    return fit_results, training_residuals

def test_model(test, model=None, seasonal=False, index='wine'): # Testing data
    if model == None and seasonal == False: # ARIMA Model
        model = ARIMAResults.load(f'models\{index}_arima.pkl')
    
    elif model == None and seasonal == True: # SARIMA Model
        model = ARIMAResults.load(f'models\{index}_sarima.pkl')

    # Testing Forecast
    forecast_steps = test.shape[0]
    forecast = model.get_forecast(steps=forecast_steps)
    forecast_ci = forecast.conf_int()
    yhat_test = forecast.predicted_mean.values # Apply the exp transformation if you used log transform before to invert scales back

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

def evaluate_model_with_Plots(data, candidates, eval_df, seasonal=False, index='wine', arima_order=None): 
    # Take the model with the lowest eval metrics and errors
    for candidate in candidates:
        if seasonal == False:
            # Split cross validation
            aic, bic, mae, mse, rmse, mape, mae_bas, mse_bas, rmse_bas, mape_bas, mae_mean, mse_mean, rmse_mean, mape_mean = split_cross_validation(data, candidate, index, None, seasonal)
            
            # Store evaluation information (those are already avg calculated in the split cross validation function)
            eval_df.loc[len(eval_df)] = [candidate, None, aic, bic, mae, mse, rmse, mape]
        
        else:
            # Split cross validation
            aic, bic, mae, mse, rmse, mape, mae_bas, mse_bas, rmse_bas, mape_bas, mae_mean, mse_mean, rmse_mean, mape_mean = split_cross_validation(data, order=arima_order, index=index, seasonal_order=candidate, seasonal=seasonal)

            # Store evaluation information (those are already avg calculated in the split cross validation function)
            eval_df.loc[len(eval_df)] = [arima_order, candidate, aic, bic, mae, mse, rmse, mape]
        
    print("MAE Baseline:", mae_bas)
    print("MSE Baseline:", mse_bas)
    print("RMSE Baseline:", rmse_bas)
    print("MAPE % Baseline:", mape_bas)
    print("MAE Mean:", mae_mean)
    print("MSE Mean:", mse_mean)
    print("RMSE Mean:", rmse_mean)
    print("MAPE % Mean:", mape_mean)
        
    return eval_df

def check_model_with_BoxJenkins(train, start_cd, seasonal_start_cd=None, index='wine'):
    # Test model
    _, train_residuals = create_model(train, start_cd, seasonal_start_cd, index)

    # Plot Train Residuals - Does it follow a white noise pattern ?
    plt.plot(train_residuals, color="black", label="train residuals", linestyle=":")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.legend(loc='best')
    plt.title(f'Model train residuals on {index} index test set')
    plt.xticks([0, len(train_residuals)/2, len(train_residuals)-1])
    plt.xlabel('Time')
    plt.ylabel('Residual value')
    plt.show()

    # Check ACF and PACF of Train Residuals
    if index=='wine':
        fig = plot_acf(train_residuals, color = "blue", lags=len(train_residuals)-1)
        plt.title(f'Index {index} model train residuals ACF')
        plt.show()

        fig = plot_pacf(train_residuals, color = "green", lags=int(len(train_residuals)/2)-1) # PACF cannot be longer than 50% of the data
        plt.title(f'Index {index} model train residuals PACF')
        plt.show()

    elif index=='watch':
        fig = plot_acf(train_residuals, color = "blue", lags=len(train_residuals)-1) # ACF cannot be longer than testing data.
        plt.title(f'Index {index} model train residuals ACF lags')
        plt.show()

        fig = plot_pacf(train_residuals, color = "green", lags=int(len(train_residuals)/2)-1) # PACF cannot be longer than 50% of the data
        plt.title(f'Index {index} model train residuals PACF lags')
        plt.show()

    else: # index=='art'
        fig = plot_acf(train_residuals, color = "blue", lags=len(train_residuals)-1) # ACF cannot be longer than testing data.
        plt.title(f'Index {index} model train residuals ACF lags')
        plt.show()

        fig = plot_acf(train_residuals, color = "blue", lags=50) # Interesting part
        plt.title(f'Index {index} model train residuals ACF lags Zoomed')
        plt.show()

        fig = plot_pacf(train_residuals, color = "green", lags=int(len(train_residuals)/2)-1) # PACF cannot be longer than 50% of the data
        plt.title(f'Index {index} model train residuals PACF lags')
        plt.show()

        fig = plot_pacf(train_residuals, color = "green", lags=50) # Interesting part
        plt.title(f'Index {index} model train residuals PACF lags Zoomed')
        plt.show()

    # Perform Ljung-Box Test on Residuals to test if they are white noise/independently distributed
    # Null Hypothesis : The residuals are independently distributed
    # Alternative Hypothesis : The residuals are not independently distributed
    # If p-value < 0.05, reject the null hypothesis thus we want to see a p-value > 0.05
    if index=='wine' or index=='wine_residuals':
        is_white_noise = is_white_noise_with_LjungBox(train_residuals, significance_level=0.05)
        print(f"Are the train residuals white noise? {is_white_noise}")

    elif index=='watch' or index=='watch_residuals':
        is_white_noise = is_white_noise_with_LjungBox(train_residuals, significance_level=0.05, lags=41)
        print(f"Are the train residuals white noise? {is_white_noise}")

    else: # index=='art' or index=='art_residuals'
        is_white_noise = is_white_noise_with_LjungBox(train_residuals, significance_level=0.05, lags=41)
        print(f"Are the train residuals white noise? {is_white_noise}")

def forecast_model(data, test, forecast_steps, length, end_date, model=None, seasonal=False, index='wine'):
    if model == None and seasonal == False: # ARIMA Model
        model = ARIMAResults.load(f'models\{index}_arima.pkl')
    elif model == None and seasonal == True: # SARIMA Model
        model = ARIMAResults.load(f'models\{index}_sarima.pkl')

    forecast = model.get_forecast(steps=forecast_steps)
    forecast_ci = forecast.conf_int()
    yhat = forecast.predicted_mean.values # Apply the exp transformation if you used log transform during fit before to invert scales back

    if index=='wine' or index=='wine_residuals':
        x_axis = pd.date_range(start=data.index[0], end=data.index[-1], freq = 'M')
        x_axis_forecast = pd.date_range(start=test.index[0], end = end_date, freq = 'M')

    elif index=='watch' or index=='watch_residuals':
        x_axis = pd.date_range(start=data.index[0], end=data.index[-1], freq = 'MS')
        x_axis_forecast = pd.date_range(start=test.index[0], end = end_date, freq = 'MS')

    else: # index=='art' or index=='art_residuals'
        x_axis = pd.date_range(start=data.index[0], end=data.index[-1], freq = 'MS')
        x_axis_forecast = pd.date_range(start=test.index[0], end = end_date, freq = 'MS')

    plt.plot(x_axis, data.values, color="blue", label="observed data")
    plt.plot(x_axis_forecast, yhat, color="red", label="forecast", linestyle="--")
    plt.legend(loc='best')
    plt.title(f'{length} term forecast of {index} index values')
    plt.xlabel('Time')
    plt.ylabel('Index value')
    plt.show()

    return yhat

def split_cross_validation(data, order, index='wine', seasonal_order=None, seasonal=False):
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
    
    splits = [0.5, 0.65, 0.85, 1.0] # split cross validation with an 80/20 ratio at each split
    for split in splits:
        split_data = data[:int(split*len(data))]
        train = split_data[:int(0.8*len(split_data))]
        test = split_data[int(0.8*len(split_data)):]
        
        fit_results, _ = create_model(train, order, seasonal_order, index)
        _, mae, mse, mae_baseline, mse_baseline, mae_baseline_mean, mse_baseline_mean, rmse, rmse_baseline, rmse_baseline_mean, mape, mape_baseline, mape_baseline_mean = test_model(test, fit_results, seasonal, index)

        # Model Evaluation Metrics
        mae_l.append(mae)
        mse_l.append(mse)
        rmse_l.append(rmse)
        mape_l.append(mape)
        aic_l.append(fit_results.aic)
        bic_l.append(fit_results.bic)

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

    # Return all eval metrics
    return np.mean(aic_l), np.mean(bic_l), np.mean(mae_l), np.mean(mse_l), np.mean(rmse_l), np.mean(mape_l), np.mean(mae_l_bas), np.mean(mse_l_bas), np.mean(rmse_l_bas), np.mean(mape_l_bas), np.mean(mae_l_mean), np.mean(mse_l_mean), np.mean(rmse_l_mean), np.mean(mape_l_mean)

def generate_arima_candidates(p, d, q, seasonal=False, m=0):
  candidates = []
  for p_val in p:
    for d_val in d:
      for q_val in q:
        if seasonal == True:
            candidates.append((p_val, d_val, q_val, m))
        else:
            candidates.append((p_val, d_val, q_val))
  return candidates

##### MAIN #####

## Load the data from global pre-processing.py ##

# Data is adjusted for inflation and decomposed into trend, seasonality and residuals
wine_df_decomp, watch_df_decomp, art_df_decomp = preprocessing.main(univariate=True)

## Evaluating stationarity of the data for the differencing parameter d ##

# # Data is non-stationary, so we apply first order differencing
wine_df_diff = wine_df_decomp.observed.diff().dropna()
watch_df_diff = watch_df_decomp.observed.diff().dropna()
art_df_diff = art_df_decomp.observed.diff().dropna()

# NB The data exhibits WAY better stationary after first order differencing
# Smoothing the data with a 30 day moving average messes (for some reason) the stationarity of the data.
# Increasing the window size makes it worse.

### (S)ARIMA (p,d,q)*(P,D,Q)m Model Forecasting (First Method) Traditional Forecasting ###

# First order differencing makes the data stationary so I will set my d = 1 as confirmed by ADF + KPSS tests

# Methodology : 
# First determine good ARIMA Model candidates using the ACF and PACF Plots
# Use split-cross validation to evaluate the candidate models on the data and pick the best one
# Then use the box-jenkins methodology to see if you can further improve the ARIMA model by checking the training residuals
# If lag orders are high, and/or performance is not that good while still having white noise residuals, and the seasonal decomposition shows seasonality
# Then do the same iterative process for a SARIMA model

# WINE
# Initial Split into train and test (for after split cross validation)
wine_train = wine_df_decomp.observed[:int(0.8*len(wine_df_decomp.observed))]
wine_test = wine_df_decomp.observed[int(0.8*len(wine_df_decomp.observed)):]
wine_seasonal = wine_df_decomp.seasonal
eval_df = pd.DataFrame(columns=["ARIMA", "SEASONAL", "AIC", "BIC", "MAE", "MSE", "RMSE", "MAPE %"]) # To store the most important evaluation metrics

# Evaluate Wine ARIMA model with ACF + PACF plots
# Candidates are chosen based on the ACF and PACF plots
# p, d, q = [0, 3, 17], [1], [0, 3, 12, 20]
# candidates = generate_arima_candidates(p, d, q)
# eval_df = evaluate_model_with_Plots(wine_df_decomp.observed, candidates, eval_df, index='wine')
# print(eval_df)

# Best model seems to be (3,1,3) within the candidates
# We still do manage to be better than the baseline but worse than the mean so this is at least one success
# We need to apply the Box-Jenkins Methodology to see if there is still room for improvement

# Evaluate Wine ARIMA model with Box-Jenkins model diagnostic
arima_wine = (3,1,3) 
# check_model_with_BoxJenkins(wine_train, arima_wine, seasonal_start_cd=None, index='wine')
# Residuals are white noise.

# Seasonality pattern repeating every 12 lags, thus set m=12. (ACF of the seasonal component)

# Candidates are chosen based on the ACF and PACF plots
# P, D, Q = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13], [0], [1, 2, 4, 5, 6, 7, 12]
# seasonal_candidates = generate_arima_candidates(P, D, Q, seasonal=True, m=12)
# eval_df = evaluate_model_with_Plots(wine_df_decomp.observed, seasonal_candidates, eval_df, seasonal=True, index='wine', arima_order=arima_wine)
# print("Head")
# print(eval_df.head(42))
# print("Tail")
# print(eval_df.tail(43))

sarima_wine = [(3,1,3), (3,0,6,12)] # m needs to be > to AR and MA order of ARIMA 
# check_model_with_BoxJenkins(wine_train, sarima_wine[0], sarima_wine[1], index='wine')
# (3, 0, 6, 12) gives a slightly lower performance than the optimal but the residuals are white noise so choose this one.

# Save optimal (S)ARIMA model
# wine_model = create_model(wine_train, arima_wine, seasonal_order=None, index='wine') # Only run once to save the optimal model
# wine_model_seasonal = create_model(wine_train, sarima_wine[0], sarima_wine[1], index='wine') # Only run once to save the optimal model

# Now that the optimal has been found, use it to forecast
short_term = wine_test.shape[0] + 12 # 1 year
medium_term = wine_test.shape[0] + 12*5 # 5 years
long_term = wine_train.shape[0] # Full training set can go beyond that but it would be extrapolation, so less reliable

# Short, medium and long term forecasts
ref_start = wine_df_decomp.observed.index[-1] # "2023-12-31"
end_short = "2024-12-31"
end_medium = "2028-12-31"
end_long = "2037-06-30"
# forecast_model(wine_df_decomp.observed, wine_test, long_term, "Long", end_date=end_long, model=None, seasonal=True, index='wine')

# WATCH 
# Initial Split into train and test (for after split cross validation)
watch_train = watch_df_decomp.observed[:int(0.8*len(watch_df_decomp.observed))]
watch_test = watch_df_decomp.observed[int(0.8*len(watch_df_decomp.observed)):]
watch_seasonal = watch_df_decomp.seasonal
eval_df = pd.DataFrame(columns=["ARIMA", "SEASONAL", "AIC", "BIC", "MAE", "MSE", "RMSE", "MAPE %"]) # To store the most important evaluation metrics

# Evaluate Watch ARIMA model with ACF + PACF plots
# Candidates are chosen based on the ACF and PACF plots
# p, d, q = [0, 1, 2, 36, 37], [1], [0, 1, 3, 5, 6]
# candidates = generate_arima_candidates(p, d, q)
# eval_df = evaluate_model_with_Plots(watch_df_decomp.observed, candidates, eval_df, index='watch')
# print(eval_df)

# Evaluate Watch ARIMA model with Box-Jenkins model diagnostic
arima_watch = (2,1,3) 
# check_model_with_BoxJenkins(watch_train, arima_watch, seasonal_start_cd=None, index='watch')
# Residuals are white noise.

# eval_df = evaluate_model_with_Plots(watch_df_decomp.observed, [arima_watch], eval_df, index='watch')
# print(eval_df)

# Seasonality pattern repeating every 12 lags, thus set m=12. (ACF of the seasonal component)

# Candidates are chosen based on the ACF and PACF plots
# P, D, Q = [1,2,6,7,8,9,10,12,13,14], [0], [1,3,4,8,9,11,12]
# seasonal_candidates = generate_arima_candidates(P, D, Q, seasonal=True, m=12) 
# eval_df = evaluate_model_with_Plots(watch_df_decomp.observed, seasonal_candidates, eval_df, seasonal=True, index='watch', arima_order=arima_watch)
# print("Head")
# print(eval_df.head(35))
# print("Tail")
# print(eval_df.tail(37))

sarima_watch = [(2,1,3), (1,0,3,12)] # Seasonal order needs to be > to AR and MA order
# check_model_with_BoxJenkins(watch_train, sarima_watch[0], sarima_watch[1], index='watch')
# Residuals are white noise.

# Save optimal (S)ARIMA model
# watch_model = create_model(watch_train, arima_watch, seasonal_order=None, index='watch') # Only run once to save the optimal model
# watch_model_seasonal = create_model(watch_train, sarima_watch[0], sarima_watch[1], index='watch') # Only run once to save the optimal model

# Now that model is trained + evaluated, use it to forecast
short_term = watch_test.shape[0] + 12 # 1 year
medium_term = watch_test.shape[0] + 12*5 # 5 years
long_term = watch_train.shape[0] # Full training set can go beyond that but it would be extrapolation, so less reliable

# Short, medium and long term forecasts
ref_start = watch_df_decomp.observed.index[-1] # "2023-12-01"
end_short = "2024-12-01"
end_medium = "2028-12-01"
end_long = "2034-02-01"
# forecast_model(watch_df_decomp.observed, watch_test, long_term, "Long", end_date=end_long, model=None, seasonal=True, index='watch')

# ART 
# Initial Split into train and test (for after split cross validation)
art_train = art_df_decomp.observed[:int(0.8*len(art_df_decomp.observed))]
art_test = art_df_decomp.observed[int(0.8*len(art_df_decomp.observed)):]
art_seasonal = art_df_decomp.seasonal
eval_df = pd.DataFrame(columns=["ARIMA", "SEASONAL", "AIC", "BIC", "MAE", "MSE", "RMSE", "MAPE %"]) # To store the most important evaluation metrics

# Evaluate Art ARIMA model with ACF + PACF plots
# Candidates are chosen based on the ACF and PACF plots
# p, d, q = [0,1,2,4,5,6,11,12,13], [1], [0,1,2,4,6,8,10,11,12]
# candidates = generate_arima_candidates(p, d, q)
# eval_df = evaluate_model_with_Plots(art_df_decomp.observed, candidates, eval_df, index='art')
# print("Head")
# print(eval_df.head(40))
# print("Tail")
# print(eval_df.tail(41))

# Evaluate Art ARIMA model with Box-Jenkins model diagnostic
arima_art = (13,1,6) 
# check_model_with_BoxJenkins(art_train, arima_art, seasonal_start_cd=None, index='art')
# (6,1,8) gives the best performance but the residuals aren't white noise, they fail the test.
# Same reasoning for (4,1,2) --> pick this one for sarima as m = 6
# (13,1,6) gives a lower performance but the residuals are white noise.

# Seasonal decomposition suggests underlying complex seasonal pattern so we will now optimize the SARIMA model
# ACF and PACF show a seasonal pattern repeating every 6 lags (ACF + PACF of the original data)

# Seasonality pattern repeating every 6 lags, thus set m=6. (ACF of the seasonal component)

# Candidates are chosen based on the ACF and PACF plots
# P, D, Q = [0,2,3,4,5,6,7,12], [0], [0,2,3,4,6,12]
# seasonal_candidates = generate_arima_candidates(P, D, Q, seasonal=True, m=6)
# eval_df = evaluate_model_with_Plots(art_df_decomp.observed, seasonal_candidates, eval_df, seasonal=True, index='art', arima_order=(4,1,2))
# print("Head")
# print(eval_df.head(24))
# print("Tail")
# print(eval_df.tail(28))

sarima_art = [(4,1,2),(5,0,6,6)]
# check_model_with_BoxJenkins(art_train, sarima_art[0], sarima_art[1], index='art')
# Residuals are white noise.
# eval_df = evaluate_model_with_Plots(art_df_decomp.observed, [sarima_art[1]], eval_df, seasonal=True, index='art', arima_order=sarima_art[0])
# print(eval_df)

# Save optimal (S)ARIMA model
# art_model = create_model(art_train, arima_art, seasonal_order=None, index='art') # Only run once to save the optimal model
# art_model_seasonal = create_model(art_train, sarima_art[0], sarima_art[1], index='art') # Only run once to save the optimal model

# Now that model is trained + evaluated, use it to forecast
short_term = art_test.shape[0] + 12 # 1 year
medium_term = art_test.shape[0] + 12*5 # 5 years
long_term = art_train.shape[0] # Full training set can go beyond that but it would be extrapolation, so less reliable

# Short, medium and long term forecasts
ref_start = art_df_decomp.observed.index[-1] # "2023-09-01"
end_short = "2024-09-01"
end_medium = "2028-09-01"
end_long = "2051-02-01"
# forecast_model(art_df_decomp.observed, art_test, long_term, "Long", end_date=end_long, model=None, seasonal=False, index='art')

### STATIONARITY TESTS ###

# Wine
# stationary = is_stationary_with_KPSS(wine_df_diff, significance_level=0.05)
# print(f"Is the data stationary according to the KPSS Test? {stationary}") # True
# stationary = is_stationary_with_ADF(wine_df_diff, significance_level=0.05)
# print(f"Is the data stationary according to the ADF Test? {stationary}") # True

# stationary = is_stationary_with_KPSS(wine_residuals, significance_level=0.05)
# print(f"Are the residuals stationary according to the KPSS Test? {stationary}") # True
# stationary = is_stationary_with_ADF(wine_residuals, significance_level=0.05)
# print(f"Are the residuals stationary according to the ADF Test? {stationary}") # True

# Watch
# stationary = is_stationary_with_KPSS(watch_df_diff, significance_level=0.05)
# print(f"Is the data stationary according to the KPSS Test? {stationary}") # True
# stationary = is_stationary_with_ADF(watch_df_diff, significance_level=0.05)
# print(f"Is the data stationary according to the ADF Test? {stationary}") # True

# stationary = is_stationary_with_KPSS(watch_residuals, significance_level=0.05)
# print(f"Are the residuals stationary according to the KPSS Test? {stationary}") # True
# stationary = is_stationary_with_ADF(watch_residuals, significance_level=0.05)
# print(f"Are the residuals stationary according to the ADF Test? {stationary}") # True

# Art
# stationary = is_stationary_with_KPSS(art_df_diff, significance_level=0.05)
# print(f"Is the data stationary according to the KPSS Test? {stationary}") # True
# stationary = is_stationary_with_ADF(art_df_diff, significance_level=0.05)
# print(f"Is the data stationary according to the ADF Test? {stationary}") # True

# stationary = is_stationary_with_KPSS(art_residuals, significance_level=0.05)
# print(f"Are the residuals stationary according to the KPSS Test? {stationary}") # True
# stationary = is_stationary_with_ADF(art_residuals, significance_level=0.05)
# print(f"Are the residuals stationary according to the ADF Test? {stationary}") # True

# Evaluate Stationarity of the seasonal component wine
# stationary = is_stationary_with_KPSS(wine_seasonal, significance_level=0.05)
# print(f"Is the data stationary according to the KPSS Test? {stationary}") # True
# stationary = is_stationary_with_ADF(wine_seasonal, significance_level=0.05)
# print(f"Is the data stationary according to the ADF Test? {stationary}") # True
# We can set our order D to 0 since the seasonal component is stationary

# Evaluate Stationarity of the seasonal component watch
# stationary = is_stationary_with_KPSS(watch_seasonal, significance_level=0.05)
# print(f"Is the data stationary according to the KPSS Test? {stationary}") # True
# stationary = is_stationary_with_ADF(watch_seasonal, significance_level=0.05)
# print(f"Is the data stationary according to the ADF Test? {stationary}") # True
# We can set our order D to 0 since the seasonal component is stationary

# Evaluate Stationarity of the seasonal component art
# stationary = is_stationary_with_KPSS(art_seasonal, significance_level=0.05)
# print(f"Is the data stationary according to the KPSS Test? {stationary}") # True
# stationary = is_stationary_with_ADF(art_seasonal, significance_level=0.05)
# print(f"Is the data stationary according to the ADF Test? {stationary}") # True
# We can set our order D to 0 since the seasonal component is stationary

### VISUALIZATION / HELPER PLOTS ###

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

# Data is stationary after first order differencing

# ## ACF and PACF plots to determine (S)ARIMA parameters ##
# fig = plot_acf(wine_df_diff, color = "blue", lags=len(wine_df_diff)-1) # ACF cannot be longer than the data.
# plt.title('Wine Index ACF 250+ lags')
# plt.show()

# fig = plot_acf(wine_df_diff, color = "blue", lags=50) # Plotting most interesting subset of the ACF
# plt.title('Wine Index ACF 50 lags')
# plt.show()

# fig = plot_acf(watch_df_diff, color = "blue", lags=len(watch_df_diff)-1) # ACF cannot be longer than the data.
# plt.title('Watch Index ACF 200+ lags')
# plt.show()

# fig = plot_acf(watch_df_diff, color = "blue", lags=50) # Plotting most interesting subset of the ACF
# plt.title('Watch Index ACF 50 lags')
# plt.show()

# fig = plot_acf(art_df_diff, color = "blue", lags=len(art_df_diff)-1) # ACF cannot be longer than the data.
# plt.title('Art Index ACF 500+ lags')
# plt.show()

# fig = plot_acf(art_df_diff, color = "blue", lags=120) # Plotting most interesting subset of the ACF
# plt.title('Art Index ACF 120 lags')
# plt.show()

# fig = plot_pacf(wine_df_diff, color = "green", lags=int((len(wine_df_diff)/2)-1)) # PACF cannot be longer than 50% of the data
# plt.title('Wine Index PACF 120+ lags')
# plt.show()

# fig = plot_pacf(wine_df_diff, color = "green", lags=50) # Plotting most interesting subset of the PACF
# plt.title('Wine Index PACF 50 lags')
# plt.show()

# fig = plot_pacf(watch_df_diff, color = "green", lags=int((len(watch_df_diff)/2)-1)) # PACF cannot be longer than 50% of the data
# plt.title('Watch Index PACF 100+ lags')
# plt.show()

# fig = plot_pacf(watch_df_diff, color = "green", lags=50) # Plotting most interesting subset of the PACF
# plt.title('Watch Index PACF 50 lags')
# plt.show()

# fig = plot_pacf(art_df_diff, color = "green", lags=int((len(art_df_diff)/2)-1)) # PACF cannot be longer than 50% of the data
# plt.title('Art Index PACF 250+ lags')
# plt.show()

# fig = plot_pacf(art_df_diff, color = "green", lags=50) # Plotting most interesting subset of the PACF
# plt.title('Art Index PACF 50 lags')
# plt.show()

# SEASONAL ACF + PACF #

# fig = plot_acf(wine_seasonal, color = "blue", lags=269) 
# plt.title('Wine Seasonality ACF 269 lags')
# plt.show() 

# fig = plot_acf(wine_seasonal, color = "blue", lags=80) # Plotting most interesting subset of the ACF
# plt.title('Wine Seasonality ACF 80 lags')
# plt.show() 

# fig = plot_pacf(wine_seasonal, color = "green", lags=134) # PACF cannot be longer than 50% of the data
# plt.title('Wine Seasonality PACF 134 lags')
# plt.show()

# fig = plot_pacf(wine_seasonal, color = "green", lags=50) # Plotting most interesting subset of the PACF
# plt.title('Wine Seasonality PACF 50 lags')
# plt.show()

# fig = plot_acf(watch_seasonal, color = "blue", lags=len(watch_seasonal)-1) 
# plt.title('Watch Seasonality ACF 200+ lags')
# plt.show() 

# fig = plot_acf(watch_seasonal, color = "blue", lags=90) # Plotting most interesting subset of the ACF
# plt.title('Watch Seasonality ACF 90 lags')
# plt.show() 

# fig = plot_pacf(watch_seasonal, color = "green", lags=int((len(watch_seasonal)/2)-1)) # PACF cannot be longer than 50% of the data
# plt.title('Watch Seasonality PACF 100+ lags')
# plt.show()

# fig = plot_pacf(watch_seasonal, color = "green", lags=50) # Plotting most interesting subset of the PACF
# plt.title('Watch Seasonality PACF 50 lags')
# plt.show()

# fig = plot_acf(art_seasonal, color = "blue", lags=len(art_seasonal)-1) 
# plt.title('Watch Seasonality ACF 500+ lags')
# plt.show() 

# fig = plot_acf(art_seasonal, color = "blue", lags=200) # Plotting most interesting subset of the ACF
# plt.title('Watch Seasonality ACF 200 lags')
# plt.show() 

# fig = plot_pacf(art_seasonal, color = "green", lags=int((len(art_seasonal)/2)-1)) # PACF cannot be longer than 50% of the data
# plt.title('Watch Seasonality PACF 250+ lags')
# plt.show()

# fig = plot_pacf(art_seasonal, color = "green", lags=50) # Plotting most interesting subset of the PACF
# plt.title('Watch Seasonality PACF 50 lags')
# plt.show()

# RESIDUAL ACF + PACF #

# fig = plot_acf(wine_residuals, color = "blue", lags=len(wine_residuals)-1) # ACF cannot be longer than the data.
# plt.title('Wine Index Residuals ACF')
# plt.show()

# fig = plot_acf(wine_residuals, color = "blue", lags=50) # Plotting most interesting subset of the ACF
# plt.title('Wine Index Residuals ACF Zoomed')
# plt.show()

# fig = plot_pacf(wine_residuals, color = "green", lags=int((len(wine_residuals)/2)-1)) # PACF cannot be longer than 50% of the data
# plt.title('Wine Index Residuals PACF')
# plt.show()

# fig = plot_pacf(wine_residuals, color = "green", lags=50) # Plotting most interesting subset of the PACF
# plt.title('Wine Index Residuals PACF Zoomed')
# plt.show()

# fig = plot_acf(watch_residuals, color = "blue", lags=len(watch_residuals)-1) # ACF cannot be longer than the data.
# plt.title('Watch Index Residuals ACF')
# plt.show()

# fig = plot_acf(watch_residuals, color = "blue", lags=50) # Plotting most interesting subset of the ACF
# plt.title('Watch Index Residuals ACF Zoomed')
# plt.show()

# fig = plot_pacf(watch_residuals, color = "green", lags=int((len(watch_residuals)/2)-1)) # PACF cannot be longer than 50% of the data
# plt.title('Watch Index Residuals PACF')
# plt.show()

# fig = plot_pacf(watch_residuals, color = "green", lags=50) # Plotting most interesting subset of the PACF
# plt.title('Watch Index Residuals PACF Zoomed')
# plt.show()

# fig = plot_acf(art_residuals, color = "blue", lags=len(art_residuals)-1) # ACF cannot be longer than the data.
# plt.title('Art Index Residuals ACF')
# plt.show()

# fig = plot_acf(art_residuals, color = "blue", lags=50) # Plotting most interesting subset of the ACF
# plt.title('Art Index Residuals ACF Zoomed')
# plt.show()

# fig = plot_pacf(art_residuals, color = "green", lags=int((len(art_residuals)/2)-1)) # PACF cannot be longer than 50% of the data
# plt.title('Art Index Residuals PACF')
# plt.show()

# fig = plot_pacf(art_residuals, color = "green", lags=50) # Plotting most interesting subset of the PACF
# plt.title('Art Index Residuals PACF Zoomed')
# plt.show()
















