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

##### STATISTICAL TESTS #####

def is_white_noise_with_LjungBox(data, significance_level=0.05, lags=50):
    # We want to FAIL to reject the null hypothesis for the data to be white noise
    # Null Hypothesis : The residuals are independently distributed
    # Alternative Hypothesis : The residuals are not independently distributed
    # If p-value < 0.05, reject the null hypothesis thus we want to see a p-value > 0.05
    df_ljungbox = sm.stats.acorr_ljungbox(data, lags=[lags], return_df=True)
    print(df_ljungbox)
    return df_ljungbox.loc[lags,"lb_pvalue"] > significance_level

##### MODEL #####

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
    # Not using blocked cross-validation because there is not enough data for sufficient blocks
    # Using split cross validation instead with an 80/20 ratio at each split
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
    
    splits = [0.5, 0.65, 0.85, 1.0]
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

def forecast_decomp_recomb_strategy(data, resid_test_data, resid_prediction, seasonal_data, trend_data, end_date, method='mean', index='wine', freq='M'):
    # Forecast seasonality with period
    seasonal = seasonal_data[6:-6] # Period of 12
    seasonal = seasonal[:12]
    seasonal_prediction = []
    counter = 0
    for i in range(0, len(resid_prediction)):
        seasonal_prediction.append(seasonal[counter])
        if counter == (len(seasonal)-1):
            counter = 0
        else:
            counter += 1

    # Forecast trend depending on chosen method:
    trend = trend_data[6:-6] 
    if method == 'mean':
        print(len(trend), len(resid_prediction))
        trend_mean = np.mean(trend[int(0.8*len(trend)):]) # Mean of the last 20% of the data
        trend_prediction = np.full(len(resid_prediction), trend_mean)
    
    elif method == 'walk_forward':
        start = int(0.8*len(trend)) # Start at the beginning of the test set for the walk forward strategy
        trend_prediction = [] 
        for i in range(0, len(resid_prediction)):
            tmp_values = trend[start:].values.tolist()
            if trend_prediction != []:
                tmp_values.extend(trend_prediction)

            trend_prediction.append(np.mean(tmp_values))
            start+=1


    # Forecast the index by building up the original scale again for each data point
    forecast = resid_prediction + seasonal_prediction + trend_prediction

    x_axis = pd.date_range(start=data.index[0], end=data.index[-1], freq = freq)
    x_axis_forecast = pd.date_range(start=resid_test_data.index[0], end = end_date, freq = freq)
    plt.plot(x_axis, data.values, color="blue", label="observed data")
    plt.plot(x_axis_forecast, forecast, color="red", label="forecast", linestyle="--")
    plt.legend(loc='best')
    plt.title(f'Long term forecast of {index} index values using ARIMA decomposition-forecasting-recombination strategy')
    plt.xlabel('Time')
    plt.ylabel('Index value')
    plt.show()


##### MAIN #####

## Load the data from global pre-processing.py ##

# Data is adjusted for inflation and decomposed into trend, seasonality and residuals
wine_df_decomp, watch_df_decomp, art_df_decomp = preprocessing.main(univariate=True)

### (S)ARIMA (p,d,q)*(P,D,Q)m Model Forecasting (Second Method) Decomposition-forecasting-recombination strategy ####

# WINE
# Initial Split into train and test (for after split cross validation)
wine_residuals = wine_df_decomp.resid.dropna() # Remove 6 NaN values at the start + end
wine_residuals_train = wine_residuals[:int(0.8*len(wine_residuals))]
wine_residuals_test = wine_residuals[int(0.8*len(wine_residuals)):]

# Are the wine residuals stationary ? Yes so set d=0 in ARIMA model

# Determine good ARIMA Model candidates using the ACF and PACF Plots and choose the best one
# p, d, q = [0,1,2,3,4,23,24], [0], [0,1,2,5,6,7,12,17,18]
# candidates = generate_arima_candidates(p, d, q)
# eval_df = evaluate_model_with_Plots(wine_residuals, candidates, eval_df, index='wine')
# print("Head")
# print(eval_df.head(35))
# print("Tail")
# print(eval_df.tail(35))

# Evaluate Wine Residual ARIMA model with Box-Jenkins model diagnostic
arima_resid_wine = (4,0,1) # (3,0,12) or (4,0,1) from the candidates
# check_model_with_BoxJenkins(wine_residuals, arima_resid_wine, seasonal_start_cd=None, index='wine')
# (4,0,1) has white noise residuals
# (3,0,12) has white noise residuals

# Save optimal model
# wine_model_resid = create_model(wine_residuals_train, arima_resid_wine, seasonal_order=None, index='wine_residuals') # Only run once to save the optimal model

# Now that model is trained + evaluated, use it to forecast
# Forecast residual
# long_term = wine_residuals_train.shape[0]
# ref_start = wine_residuals.index[-1] # 2023-06-30
# end_long = "2036-04-30"
# wine_resid_prediction = forecast_model(wine_residuals, wine_residuals_test, long_term, "Long", end_date=end_long, model=None, seasonal=False, index='wine_residuals')
# forecast_decomp_recomb_strategy(wine_df_decomp.observed, wine_residuals_test, wine_resid_prediction, wine_df_decomp.seasonal, wine_df_decomp.trend, end_long, method='walk_forward', index='wine', freq='M')

# WATCH
# Initial Split into train and test (for after split cross validation)
watch_residuals = watch_df_decomp.resid.dropna() # Remove 6 NaN values at the start + end
watch_residuals_train = watch_residuals[:int(0.8*len(watch_residuals))]
watch_residuals_test = watch_residuals[int(0.8*len(watch_residuals)):]

# Are the watch residuals stationary ? Yes so set d=0 in ARIMA model

# Determine good ARIMA Model candidates using the ACF and PACF Plots and choose the best one
# p, d, q = [0,1,2,14,26], [0], [0,1,4]
# candidates = generate_arima_candidates(p, d, q)
# eval_df = evaluate_model_with_Plots(watch_residuals, candidates, eval_df, index='watch')
# print(eval_df)

# Evaluate Watch Residual ARIMA model with Box-Jenkins model diagnostic
arima_resid_watch = (2,0,0) 
# check_model_with_BoxJenkins(watch_residuals, arima_resid_watch, seasonal_start_cd=None, index='watch')
# Residuals are white noise

# Save optimal model
# watch_model_resid = create_model(watch_residuals_train, arima_resid_watch, seasonal_order=None, index='watch_residuals') # Only run once to save the optimal model

# Now that model is trained + evaluated, use it to forecast
# Forecast residual
# long_term = watch_residuals_train.shape[0]
# ref_start = watch_residuals.index[-1] # 2023-06-01
# end_long = "2033-02-01"
# watch_resid_prediction = forecast_model(watch_residuals, watch_residuals_test, long_term, "Long", end_date=end_long, model=None, seasonal=False, index='watch_residuals')
# forecast_decomp_recomb_strategy(watch_df_decomp.observed, watch_residuals_test, watch_resid_prediction, watch_df_decomp.seasonal, watch_df_decomp.trend, end_long, method='mean', index='watch', freq='MS')

# ART
# Initial Split into train and test (for after split cross validation)
art_residuals = art_df_decomp.resid.dropna() # Remove 6 NaN values at the start + end
art_residuals_train = art_residuals[:int(0.8*len(art_residuals))]
art_residuals_test = art_residuals[int(0.8*len(art_residuals)):]

# Are the art residuals stationary ? Yes so set d=0 in ARIMA model

# Determine good ARIMA Model candidates using the ACF and PACF Plots and choose the best one
# p, d, q = [0,1,2,4,6,7,18,19,31], [0], [0,1,2,3,4,6,10,12,42]
# candidates = generate_arima_candidates(p, d, q)
# eval_df = evaluate_model_with_Plots(art_residuals, candidates, eval_df, index='art')
# print("Head")
# print(eval_df.head(40))
# print("Tail")
# print(eval_df.tail(41))

# Evaluate Art Residual ARIMA model with Box-Jenkins model diagnostic
arima_resid_art = (6,0,10) 
# check_model_with_BoxJenkins(art_residuals, arima_resid_art, seasonal_start_cd=None, index='art')
# Residuals are white noise

# Save optimal model
# art_model_resid = create_model(art_residuals_train, arima_resid_art, seasonal_order=None, index='art_residuals') # Only run once to save the optimal model

# Now that model is trained + evaluated, use it to forecast
# Forecast residual
# long_term = art_residuals_train.shape[0]
# ref_start = art_residuals.index[-1] # 2023-03-01
# end_long = "2049-12-01"
# art_resid_prediction = forecast_model(art_residuals, art_residuals_test, long_term, "Long", end_date=end_long, model=None, seasonal=False, index='art_residuals')
# forecast_decomp_recomb_strategy(art_df_decomp.observed, art_residuals_test, art_resid_prediction, art_df_decomp.seasonal, art_df_decomp.trend, end_long, method='walk_forward', index='art', freq='MS')
