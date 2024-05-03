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

# TODO : Justify the choice of SARIMA parameters for each asset in the report
# TODO : Continue Fine-tuning Parameters
# TODO : Find a way to remotely train or train faster the heavy SARIMA model for each asset

# Links to understand more about SARIMA Parameters : 

# https://en.wikipedia.org/w<iki/Box%E2%80%93Jenkins_method
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
    plt.plot(yhat_test, color="green", label="predicted") # Comment this when evaluating multiple models
    plt.plot(y_test, color="blue", label="observed") # Comment this when evaluating multiple models
    plt.plot(baseline, color="red", label="baseline") # Comment this when evaluating multiple models 
    plt.plot(baseline_mean, color="purple", label="mean") # Comment this when evaluating multiple models 
    plt.legend(loc='best') # Comment this when evaluating multiple models
    plt.title(f'Compare forecasted and observed {index} index values for test set') # Comment this when evaluating multiple models
    plt.xticks([0, len(y_test)/2, len(y_test)-1]) # Comment this when evaluating multiple models 
    plt.xlabel('Time') # Comment this when evaluating multiple models 
    plt.ylabel('Index value') # Comment this when evaluating multiple models
    plt.show() # Comment this when evaluating multiple models

    return yhat_test, mae, mse, mae_baseline, mse_baseline, mae_baseline_mean, mse_baseline_mean, rmse, rmse_baseline, rmse_baseline_mean, mape, mape_baseline, mape_baseline_mean

def evaluate_model_with_Plots(train, test, candidates, eval_df, seasonal=False, seasonal_order=None, index='wine'): 
    # Take the model with the lowest eval metrics and errors
    for candidate in candidates:
        if seasonal == False:
            # Fit candidate model
            cd_fit_results, _ = create_model(train, candidate, seasonal_order, index) 
        
            # Test candidate model on test set
            _, cd_mae, cd_mse, mae_bas, mse_bas, mae_mean, mse_mean, cd_rmse, rmse_bas, rmse_mean, cd_mape, mape_bas, mape_mean = test_model(test, cd_fit_results, seasonal, index)

            # Store evaluation information
            eval_df.loc[len(eval_df)] = [candidate, seasonal_order, cd_fit_results.aic, cd_fit_results.bic, cd_mae, cd_mse, cd_rmse, cd_mape]
            print("MAE Baseline:", mae_bas)
            print("MSE Baseline:", mse_bas)
            print("RMSE Baseline:", rmse_bas)
            print("MAPE % Baseline:", mape_bas)
            print("MAE Mean:", mae_mean)
            print("MSE Mean:", mse_mean)
            print("RMSE Mean:", rmse_mean)
            print("MAPE % Mean:", mape_mean)
            
        else:
            cd_fit_results = ARIMAResults.load(f'models\{index}_sarima.pkl')

            # Test candidate model on test set
            _, cd_mae, cd_mse, mae_bas, mse_bas, mae_mean, mse_mean, cd_rmse, rmse_bas, rmse_mean, cd_mape, mape_bas, mape_mean = test_model(wine_test, cd_fit_results, seasonal, index)

            # Store evaluation information
            eval_df.loc[len(eval_df)] = [candidate, seasonal_order, cd_fit_results.aic, cd_fit_results.bic, cd_mae, cd_mse, cd_rmse, cd_mape]
            print("MAE Baseline:", mae_bas)
            print("MSE Baseline:", mse_bas)
            print("RMSE Baseline:", rmse_bas)
            print("MAPE % Baseline:", mape_bas)
            print("MAE Mean:", mae_mean)
            print("MSE Mean:", mse_mean)
            print("RMSE Mean:", rmse_mean)
            print("MAPE % Mean:", mape_mean)
        
    return eval_df

def evaluate_model_with_BoxJenkins(train, test, start_cd, eval_df, seasonal_start_cd, seasonal=False, index='wine'):
    if seasonal == False:
        # Create ARIMA Model
        start = start_cd[0]
        seasonal_start = None
        fit_results, _ = create_model(train, start, seasonal_start, index)
    
    else:
        start = start_cd[0]
        seasonal_start = seasonal_start_cd[0]
        fit_results, _ = create_model(train, start, seasonal_start, index)

    # Test ARIMA Model
    yhat_test, mae, mse, mae_baseline, mse_baseline, mae_baseline_mean, mse_baseline_mean, rmse, rmse_baseline, rmse_baseline_mean, mape, mape_baseline, mape_baseline_mean = test_model(test, fit_results, seasonal, index)

    # Get Evaluation Metrics for this model:
    eval_df.loc[len(eval_df)] = [start, seasonal_start, fit_results.aic, fit_results.bic, mae, mse, rmse, mape]

    # Compute Test Residuals
    y_test = test
    residuals = y_test - yhat_test

    # Plot Residuals - Does it follow a white noise pattern ?
    plt.plot(residuals, color="blue", label="residuals", linestyle=":")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.legend(loc='best')
    plt.title(f'Model residuals on {index} index test set')
    plt.xticks([0, len(residuals)/2, len(residuals)-1])
    plt.xlabel('Time')
    plt.ylabel('Residual value')
    plt.show()

    # Check ACF and PACF of Residuals
    if index=='wine':
        fig = plot_acf(residuals, color = "blue", lags=50)
        plt.title(f'Index {index} model residuals ACF')
        plt.show()

        fig = plot_pacf(residuals, color = "green", lags=26) # PACF cannot be longer than 50% of the data
        plt.title(f'Index {index} model residuals PACF')
        plt.show()

    elif index=='watch':
        fig = plot_acf(residuals, color = "blue", lags=41) # ACF cannot be longer than testing data.
        plt.title('Watch Index Model Residuals ACF 50 lags')
        plt.show()

        fig = plot_pacf(residuals, color = "green", lags=20) # PACF cannot be longer than 50% of the data
        plt.title('Watch Index Model Residuals PACF 26 lags')
        plt.show()

    else: # index=='art'
        fig = plot_acf(residuals, color = "blue", lags=109) # ACF cannot be longer than testing data.
        plt.title('Art Index Model Residuals ACF 100+ lags')
        plt.show()

        fig = plot_pacf(residuals, color = "green", lags=55) # PACF cannot be longer than 50% of the data
        plt.title('Art Index Model Residuals PACF 55 lags')
        plt.show()

    # Perform Ljung-Box Test on Residuals to test if they are white noise/independently distributed
    # Null Hypothesis : The residuals are independently distributed
    # Alternative Hypothesis : The residuals are not independently distributed
    # If p-value < 0.05, reject the null hypothesis thus we want to see a p-value > 0.05
    if index=='wine':
        is_white_noise = is_white_noise_with_LjungBox(residuals, significance_level=0.05)
        print(f"Are the residuals white noise? {is_white_noise}")

    elif index=='watch':
        is_white_noise = is_white_noise_with_LjungBox(residuals, significance_level=0.05, lags=41)
        print(f"Are the residuals white noise? {is_white_noise}")

    else: # index=='art'
        is_white_noise = is_white_noise_with_LjungBox(residuals, significance_level=0.05, lags=41)
        print(f"Are the residuals white noise? {is_white_noise}")

def forecast_model(data, train, test, forecast_steps, length, end_date, model=None, seasonal=False, index='wine'):
    if model == None and seasonal == False: # ARIMA Model
        model = ARIMAResults.load(f'models\{index}_arima.pkl')
    elif model == None and seasonal == True: # SARIMA Model
        model = ARIMAResults.load(f'models\{index}_sarima.pkl')

    forecast = model.get_forecast(steps=forecast_steps)
    forecast_ci = forecast.conf_int()
    yhat = forecast.predicted_mean.values # Apply the exp transformation if you used log transform during fit before to invert scales back

    if index=='wine':
        x_axis = pd.date_range(start=data.index[0], end=data.index[-1], freq = 'M')
        x_axis_forecast = pd.date_range(start=test.index[0], end = end_date, freq = 'M')

    elif index=='watch':
        x_axis = pd.date_range(start=data.index[0], end=data.index[-1], freq = 'MS')
        x_axis_forecast = pd.date_range(start=test.index[0], end = end_date, freq = 'MS')

    else: # index=='art'
        x_axis = pd.date_range(start=data.index[0], end=data.index[-1], freq = 'MS')
        x_axis_forecast = pd.date_range(start=test.index[0], end = end_date, freq = 'MS')

    plt.plot(x_axis, data.values, color="blue", label="observed data")
    plt.plot(x_axis_forecast, yhat, color="red", label="forecast", linestyle="--")
    plt.legend(loc='best')
    plt.title(f'{length} term forecast of {index} index values')
    plt.xlabel('Time')
    plt.ylabel('Index value')
    plt.show()

def cross_validation(data, order, seasonal_order=None, seasonal=False, index='wine'):
    pass

##### MAIN #####

## Load the data from global pre-processing.py ##

# Data is adjusted for inflation and decomposed into trend, seasonality and residuals
wine_df_decomp, watch_df_decomp, art_dfdecomp = preprocessing.main(univariate=True)

## Evaluating stationarity of the data and the ARIMA Parameters ##

# # Data is non-stationary, so we apply first order differencing
wine_df_diff = wine_df_decomp.observed.diff().dropna()
watch_df_diff = watch_df_decomp.observed.diff().dropna()
art_df_diff = art_dfdecomp.observed.diff().dropna()

# NB The data exhibits WAY better stationary after first order differencing
# Smoothing the data with a 30 day moving average messes (for some reason) the stationarity of the data.
# Increasing the window size makes it worse.

# Evaluating stationarity of the data using KPSS and ADF tests 
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

## SARIMA (p,d,q)*(P,D,Q)**M Model Forecasting ##

# First order differencing makes the data stationary so I will set my d = 1 as confirmed by ADF + KPSS tests

# Methodology : 
# First determine a good ARIMA Model using the ACF and PACF Plots
# Then use the Box-Jenkins Methodology to determine the optimal ARIMA model by choosing an autocorrellation lag close to 0 for residuals
# Finally if there is an underlying complex seasonal pattern, use SARIMA to capture it
# If it improves accuracy, then optimize the SARIMA model with the Box-Jenkins Methodology on the residual + seasonal component

# WINE INDEX DATA FORECASTING

# Initial Split into train and test
wine_train = wine_df_decomp.observed[:int(0.8*len(wine_df_decomp.observed))]
wine_test = wine_df_decomp.observed[int(0.8*len(wine_df_decomp.observed)):]
wine_seasonal = wine_df_decomp.seasonal
eval_df = pd.DataFrame(columns=["ARIMA", "SEASONAL", "AIC", "BIC", "MAE", "MSE", "RMSE", "MAPE %"]) # To store the most important evaluation metrics

# Box Jenkins Methodology to determine the optimal ARIMA model
# Evaluate Wine ARIMA model with ACF + PACF plots
# Candidates are chosen based on the ACF and PACF plots
# candidates = [(17,1,0), (3,1,0), (0,1,20), (0,1,12), (0,1,3), (17,1,20)]
# eval_df = evaluate_model_with_Plots(wine_train, wine_test, candidates, eval_df, index='wine')
# print(eval_df)

# Best model seems to be (17,1,0) if we want a simpler model 
# (17,1,20) is the best in terms of MAE and (R)MSE, but it is more complex and thus penalized by AIC and BIC
# We still do manage to be better than the baseline and the mean so this at least one success

# However this suggests that the optimal cannot be precisely determined by the ACF and PACF plots 
# Because in this case, both the MA and AR components are active, i.e p,q > 0
# Thus we need to look at the residuals and the Box-Jenkins model diagnostic to determine the optimal model 

# Evaluate Wine ARIMA model with Box-Jenkins model diagnostic
# Starting point : previous best model (17,1,20) by combining previous best AR and MA orders
start_cd = [(17,1,20)] 
# evaluate_model_with_BoxJenkins(wine_train, wine_test, start_cd, eval_df, seasonal_start_cd=None, seasonal=False, index='wine')
# The residual of this model (17,1,20) indicates a significant value of 0 at lag 12 in the ACF
# As well as a significant value of 0 at lag 17 in the PACF of the residuals which is a good sign telling us that the AR order is optimal
# Thus I will try a model with (17,1,12) to see if we improve the performance
# Best model yet : (17,1,12)
# Best model yet 2 : (17,1,20) is more complex but does seem to have better performance
# NB : Log-transformation improves the AIC (goodness of fit) and BIC (model complexity) a lot -1000 points
# But it increases MAE and MSE error on the test set. This is a trade-off between goodness of fit and predictive power

# Seasonal decomposition suggests underlying complex seasonal pattern 
# Additionally: AR and MA orders are pretty high, suggesting the need for a SARIMA model

# Evaluate Stationarity of the seasonal component 
# stationary = is_stationary_with_KPSS(wine_seasonal, significance_level=0.05)
# print(f"Is the data stationary according to the KPSS Test? {stationary}") # True
# stationary = is_stationary_with_ADF(wine_seasonal, significance_level=0.05)
# print(f"Is the data stationary according to the ADF Test? {stationary}") # True
# We can set our order D to 0 since the seasonal component is stationary

# By looking at the ACF and PACF plots of the seasonal component, there is a significant lag at 13 but none beyond
# in the PACF, thus we can set our seasonal AR order P to 13.
# In the ACF, there is a significant lag at 71, but none beyond, thus we can set Q to 71.
# However, 71 is a pretty big value, thus we can also set Q to 12, which is the maximum positive value for a lag
# outside the significance region. Or 6 which is the maximum negative value for a lag outside the significance region.
# The period in the ACF seems to repeat itself every 9 lags, thus we can set our M to be 9.
# But we will put just above the AR and MA order of the ARIMA to avoid duplicate lags
seasonal_start_cd = [(), (13,0,6,9)] # Seasonal order needs to be > to AR and MA order of ARIMA
# evaluate_model_with_BoxJenkins(wine_train, wine_test, start_cd, eval_df, seasonal_start_cd, seasonal=True, index='wine')

# Create optimal (S)ARIMA model
optimal = start_cd[0]
optimal_seasonal = seasonal_start_cd[0]
optimal_seasonal_order = seasonal_start_cd[1]
# wine_model = create_model(wine_train, optimal, seasonal_order=None, index='wine') # Only run once to save the optimal model
# wine_model = create_model(wine_train, optimal_seasonal, optimal_seasonal_order, index='wine') # Only run once to save the optimal model

# Now that the optimal has been found, use it to forecast
short_term = wine_test.shape[0] + 12 # 1 year
medium_term = wine_test.shape[0] + 12*5 # 5 years
long_term = wine_train.shape[0] # Full training set can go beyond that but it would be extrapolation, so less reliable

# Short, medium and long term forecasts
ref_start = wine_df_decomp.observed.index[-1] # "2023-12-31"
end_short = "2024-12-31"
end_medium = "2028-12-31"
end_long = "2037-06-30"
# forecast_model(wine_df_decomp.observed, wine_train, wine_test, long_term, "Long", end_date=end_long, model=None, seasonal=False, index='wine')

# WATCH INDEX DATA FORECASTING
# Split data into train and test
watch_train = watch_df_decomp.observed[:int(0.8*len(watch_df_decomp.observed))]
watch_test = watch_df_decomp.observed[int(0.8*len(watch_df_decomp.observed)):]
watch_seasonal = watch_df_decomp.seasonal
eval_df = pd.DataFrame(columns=["ARIMA", "SEASONAL", "AIC", "BIC", "MAE", "MSE", "RMSE", "MAPE %"]) # To store the most important evaluation metrics

# Box Jenkins Methodology to determine the optimal ARIMA model
# Evaluate Watch ARIMA model with ACF + PACF plots
# Candidates are chosen based on the ACF and PACF plots
# candidates = [(0,1,1), (0,1,6), (2,1,0), (37,1,0), (37,1,1), (37,1,6), (2,1,6), (2,1,1)]
# eval_df = evaluate_model_with_Plots(watch_train, watch_test, candidates, eval_df, index='watch')
# print(eval_df)

# The best model seems to be (37,1,6) if we want a simpler model (37,1,0)
# They are both penalized by the high AR order, however those with low AR order < 10 really are not good

# Evaluate Watch ARIMA model with Box-Jenkins model diagnostic
# Starting point : previous best model (37,1,6)
start_cd = [(37,1,9)] 
# evaluate_model_with_BoxJenkins(watch_train, watch_test, start_cd, eval_df, seasonal_start_cd=None, seasonal=False, index='watch')
# The residual of this model exhibits a value of 0 at lag 9 in the ACF
# Since this is the residual of the model, we want to take the residual closest to 0.
# Thus set my MA order to 9. Same logic for the AR order by setting it to 15 when looking at the PACF.
# There are conflicting results: The ACF plot show that some of the autocorrelation has been captured
# The residual analysis shows that its a bit closer to random noise
# However it increased the test set error and model complexity.
# Best model yet : (15,1,9) exhibits the best performance while keeping the model simple
# Best model yet 2 : (37,1,9) is more complex but does seem to have better performance with residuals being closer to random noise.
# NB : Log transformation improves all the evaluation metrics but makes residuals less white noise.
# Thus seeing how the model seems to capture less the structure of the data with the log-transformation
# Improvement on the test set might just be due to chance.

# Seasonal decomposition suggests underlying complex seasonal pattern 
# Additionally: AR and MA orders are pretty high, suggesting the need for a SARIMA model

# Evaluate Stationarity of the seasonal component 
# stationary = is_stationary_with_KPSS(watch_seasonal, significance_level=0.05)
# print(f"Is the data stationary according to the KPSS Test? {stationary}") # True
# stationary = is_stationary_with_ADF(watch_seasonal, significance_level=0.05)
# print(f"Is the data stationary according to the ADF Test? {stationary}") # True
# We can set our order D to 0 since the seasonal component is stationary

# By looking at the ACF and PACF plots of the seasonal component, there is a significant lag at 14
# but none beyond 14 in the PACF thus we can set our P to 14
# In the ACF, there is a significant lag at 72, but none beyond, thus we can set Q to 72
# However, 72 is a pretty big value, thus we can also set Q to 12 or 4, which is the maximum positive or negative value exhibited by a lag.
# We can set our seasonal order to be 38 since it is the minimum value > to the AR and MA order of the ARIMA model
# Although the period in the ACF of the seasonal component seems to repeat itself every 12 lags and the fact we have monthly data.
# There is evidence for a seasonal order of 12 lags, however we cannot set it lower than the AR and MA order of the ARIMA model to avoid duplicates.
seasonal_start_cd = [(14,0,12,38)] # Seasonal order needs to be > to AR and MA order
# evaluate_model_with_BoxJenkins(watch_train, watch_test, start_cd, eval_df, seasonal_start_cd, seasonal=True, index='watch')

# Create optimal (S)ARIMA model
optimal = start_cd[0]
optimal_seasonal = seasonal_start_cd[0]
# watch_model = create_model(watch_train, optimal, seasonal_order=None, index='watch') # Only run once to save the optimal model
# watch_model = create_model(watch_train, optimal_seasonal, optimal_seasonal_order, index='watch') # Only run once to save the optimal model

# Now that model is trained + evaluated, use it to forecast
short_term = watch_test.shape[0] + 12 # 1 year
medium_term = watch_test.shape[0] + 12*5 # 5 years
long_term = watch_train.shape[0] # Full training set can go beyond that but it would be extrapolation, so less reliable

# Short, medium and long term forecasts
ref_start = watch_df_decomp.observed.index[-1] # "2023-12-01"
end_short = "2024-12-01"
end_medium = "2028-12-01"
end_long = "2034-02-01"
# forecast_model(watch_df_decomp.observed, watch_train, watch_test, long_term, "Long", end_date=end_long, model=None, seasonal=False, index='watch')

# ART INDEX DATA FORECASTING
# Split data into train and test
art_train = art_dfdecomp.observed[:int(0.8*len(art_dfdecomp.observed))]
art_test = art_dfdecomp.observed[int(0.8*len(art_dfdecomp.observed)):]
art_seasonal = art_dfdecomp.seasonal
eval_df = pd.DataFrame(columns=["ARIMA", "SEASONAL", "AIC", "BIC", "MAE", "MSE", "RMSE", "MAPE %"]) # To store the most important evaluation metrics

# Box Jenkins Methodology to determine the optimal ARIMA model
# Evaluate Art ARIMA model with ACF + PACF plots
# Candidates are chosen based on the ACF and PACF plots
# candidates = [(0,1,6), (0,1,42), (0,1,96), (42,1,0), (30,1,0), (30,1,6), (30,1,42), (30,1,96), (42,1,6), (42,1,42), (42,1,96)]
# eval_df = evaluate_model_with_Plots(art_train, art_test, candidates, eval_df, index='art')
# print(eval_df)

# The best model seems to be (30,1,6)

# Evaluate Art ARIMA model with Box-Jenkins model diagnostic
# Starting point : previous best model (30,1,6)
start_cd = [(32,1,13)] 
# evaluate_model_with_BoxJenkins(art_train, art_test, start_cd, eval_df, seasonal_start_cd=None, seasonal=False, index='art')
# In the PACF of the residuals, there are a few lags before 30 that have value 0: lag 12,13,24
# There is also lag 32 just above 30 that has a value of 0, thus I will set my AR order to 32 or 24 and see which improves most.
# In the ACF of the residuals, lag 13,14 are closest to 0, thus I will set my MA to whichever improves most the model on the test set.
# There is also lag 39, but 39 seems pretty high and will increase the complexity of the model while setting a minimum constraint for the seasonal order later.
# Best model yet : (32,1,13) exhibits the best performance reducing the MAE and MSE on the test set and captures the trend in the test set
# NB: Log - transformation stabilizes the variance as such the residuals are closer to white noise, however there is less volatility in the forecast, thus less accuracy on the test set
# Since the original data is pretty volatile, thus discard the log-transformation again.

# Seasonal decomposition suggests underlying complex seasonal pattern so we will now optimize the SARIMA model

# Evaluate Stationarity of the seasonal component

# stationary = is_stationary_with_KPSS(art_seasonal, significance_level=0.05)
# print(f"Is the data stationary according to the KPSS Test? {stationary}") # True
# stationary = is_stationary_with_ADF(art_seasonal, significance_level=0.05)
# print(f"Is the data stationary according to the ADF Test? {stationary}") # True
# We can set our order D to 0 since the seasonal component is stationary

# By looking at the ACF and PACF plots of the seasonal component, there is a significant lag in the PACF
# 19, 24 and 30. 19 is the last lag out of a cluster of significant lags, so we start with 19
# and then try 24 and 30 to see if it improves performance. In the ACF, there is a significant lag at 180
# but knowing 180 is a pretty large value, we can set it to 4 or 12 which are the lags with the highest positive
# or negative value outside the significance region.
# ACF pattern repeats itself every 7 lags, and since the data is sampled monthly, either 7 or 12 seems good
# for M being the seasonal order. However M > max(p,q) from ARIMA(p,d,q) to avoid duplicate lags. Thus we
# can set it to the minimum value possible: 33 since p = 32 and q = 13.
seasonal_start_cd = [(19,0,12,33)] # Seasonal order needs to be > to AR and MA order
# evaluate_model_with_BoxJenkins(art_train, art_test, start_cd, eval_df, seasonal_start_cd, seasonal=True, index='art')

# Create optimal (S)ARIMA model
optimal = start_cd[0]
optimal_seasonal = seasonal_start_cd[0]
# art_model = create_model(art_train, optimal, seasonal_order=None, index='art') # Only run once to save the optimal model
# art_model = create_model(art_train, optimal_seasonal, optimal_seasonal_order, index='art') # Only run once to save the optimal model

# Now that model is trained + evaluated, use it to forecast
short_term = art_test.shape[0] + 12 # 1 year
medium_term = art_test.shape[0] + 12*5 # 5 years
long_term = art_train.shape[0] # Full training set can go beyond that but it would be extrapolation, so less reliable

# Short, medium and long term forecasts
ref_start = art_dfdecomp.observed.index[-1] # "2023-09-01"
end_short = "2024-09-01"
end_medium = "2028-09-01"
end_long = "2051-02-01"
# forecast_model(art_dfdecomp.observed, art_train, art_test, long_term, "Long", end_date=end_long, model=None, seasonal=False, index='art')
      
##### VISUALIZATION / HELPER PLOTS #####

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
















