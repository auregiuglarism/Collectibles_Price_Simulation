import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import preprocessing   

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMAResults
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import statsmodels.api as sm

# Links to understand more about correlation and (S)ARIMAX

# TODO : Read paper : time series analysis to modeling to forecast 
# TODO : Start correlation analysis

##### CORRELATION #####

def compute_covariance(cov_df, index_df, variables):
    covariances =  []
    for variable in variables:
        cov = np.ma.cov(index_df, variable, rowvar=False)
        covariances.append(cov)

    cov_df = pd.concat(cov_df, pd.DataFrame([covariances]), ignore_index=True)
    return cov_df
        

##### MODELS #####

def create_model(train, order, exogenous_var, seasonal_order=None, index='wine'):
    if seasonal_order == None: # ARIMAX Model
        model = ARIMA(train, trend='n', order=order, 
                      exog=exogenous_var,  
                      enforce_stationarity=True,
                      enforce_invertibility=True) 
        
        fit_results = model.fit()
        fit_results.save(f'models\{index}_arimax.pkl') # Comment this when evaluating multiple models
        
    else: # SARIMAX Model
        model = ARIMA(train, trend='n', order=order,  
                exog=exogenous_var,
                seasonal_order=seasonal_order,
                enforce_stationarity=True,
                enforce_invertibility=True) 
        
        model.initialize_approximate_diffuse() # Avoid LU Decomposition error when searching for optimal parameters
        
        fit_results = model.fit()
        fit_results.save(f'models\{index}_sarimax.pkl') # Comment this when evaluating multiple models

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

##### MAIN #####

## Load the data from pre-processing ##
wine_df, watch_df, art_df, crypto_df, gold_df, sp500_df, cpi_df, bond_yield_df = preprocessing.main(univariate=False)

# Compute covariance matrix between each pair of correlated variables and index
wine_cov = pd.DataFrame({"Crypto", "Gold", "SP500", "CPI", "Bond Yield"})
wine_cov = compute_covariance(wine_cov, wine_df.observed, [crypto_df.observed, gold_df.observed, sp500_df.observed, cpi_df.observed, bond_yield_df.observed])
print(wine_cov)
# Compute Pearson correlation coefficient between each pair of correlated variables and index

# Test the significance of the correlation coefficient with a t-test

# Select the most correlated variable for each index to use with ARIMAX



