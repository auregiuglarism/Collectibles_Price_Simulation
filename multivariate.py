import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import preprocessing   

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMAResults
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import statsmodels.api as sm

from scipy.stats import ttest_ind

# Links to understand more about correlation, (S)ARIMAX, covariance

# https://www.statology.org/how-to-read-covariance-matrix/#:~:text=The%20values%20along%20the%20diagonals%20of%20the%20matrix,matrix%20represent%20the%20covariances%20between%20the%20various%20subjects.
# https://builtin.com/data-science/covariance-matrix

# https://datagy.io/t-test-python/

# TODO : Read paper : time series analysis to modeling to forecast 
# TODO : Include an exogenous variable about the real estate market in the US and other exogenous twins about non-US markets for each exogenous variable that already exists.
# TODO : Write down accuracy for each asset class

##### CORRELATION #####

def compute_covariance(cov_df, index_df, variables):
    covariances =  []

    start_date = index_df.index[0].split("-")
    year_df_first = start_date[0]
    month_df_first = start_date[1]
    end_date = index_df.index[-1].split("-")
    year_df_end = end_date[0]
    month_df_end = end_date[1]

    for variable in variables:

        last_date = variable.index[-1].split("-")
        year_var_end = last_date[0]
        month_var_end = last_date[1]
        first_date = variable.index[0].split("-")
        year_var_first = first_date[0]
        month_var_first = first_date[1]

        if (len(variable) > len(index_df)): # If the variable has more data than the index and ends later
            if int(year_var_end) < int(year_df_end): # If the variable ends before the index
                last_row = variable.loc[year_var_end+"-"+month_var_end:].index[0]
                variable = variable.loc[year_df_first+"-"+month_df_first:last_row]

                last_row = index_df.loc[year_var_end+"-"+month_var_end:].index[0]
                index_df_cov = index_df.loc[year_df_first+"-"+month_df_first:last_row]
                cov = np.cov(index_df_cov, variable)[0][1]
            else:
                last_row = index_df.loc[year_df_end+"-"+month_df_end:].index[0]
                variable = variable.loc[year_df_first+"-"+month_df_first:last_row]
                cov = np.cov(index_df, variable)[0][1]

        if len(variable) < len(index_df): # If the variable has less data than the index
            if int(year_df_end) <= int(year_var_end) and int(month_df_end) < int(month_var_end): # If the variable ends after the index
                last_row = index_df.loc[year_df_end+"-"+month_df_end:].index[0]
                variable = variable.loc[year_var_first+"-"+month_var_first:last_row]

                last_row = index_df.loc[year_df_end+"-"+month_df_end:].index[0]
                index_df_cov = index_df.loc[year_var_first+"-"+month_var_first:last_row] 

                if index_df_cov.index[-1].split("-")[1] != variable.index[-1].split("-")[1]: # If slice ends in different month
                    index_df_cov = index_df_cov[:-1]
                cov = np.cov(index_df_cov, variable)[0][1]

            else: # If the variable ends before the index
                last_row = index_df.loc[year_var_end+"-"+month_var_end:].index[0]
                index_df_cov = index_df.loc[year_var_first+"-"+month_var_first:last_row]
                cov = np.cov(index_df_cov, variable)[0][1]

        covariances.append(cov)

    cov_df.loc[len(cov_df)] = [covariances[0], covariances[1], covariances[2], covariances[3]]
    return cov_df

def compute_pearson_coeff(pearson_df, index_df, variables):
    pearson_coeffs = []

    start_date = index_df.index[0].split("-")
    year_df_first = start_date[0]
    month_df_first = start_date[1]
    end_date = index_df.index[-1].split("-")
    year_df_end = end_date[0]
    month_df_end = end_date[1]

    for variable in variables:

        last_date = variable.index[-1].split("-")
        year_var_end = last_date[0]
        month_var_end = last_date[1]
        first_date = variable.index[0].split("-")
        year_var_first = first_date[0]
        month_var_first = first_date[1]

        if (len(variable) > len(index_df)): # If the variable has more data than the index and ends later
            if int(year_var_end) < int(year_df_end): # If the variable ends before the index
                last_row = variable.loc[year_var_end+"-"+month_var_end:].index[0]
                variable = variable.loc[year_df_first+"-"+month_df_first:last_row]

                last_row = index_df.loc[year_var_end+"-"+month_var_end:].index[0]
                index_df_coef = index_df.loc[year_df_first+"-"+month_df_first:last_row]

                # Substract the sample mean
                index_df_coef = index_df_coef - index_df_coef.mean()
                variable = variable - variable.mean()
                coef = np.corrcoef(index_df_coef, variable)[0][1]
            else:
                last_row = index_df.loc[year_df_end+"-"+month_df_end:].index[0]
                variable = variable.loc[year_df_first+"-"+month_df_first:last_row]

                # Substract the sample mean 
                index_df_coef = index_df - index_df.mean() 
                variable = variable - variable.mean()
                coef = np.corrcoef(index_df_coef, variable)[0][1]

        if len(variable) < len(index_df): # If the variable has less data than the index
            if int(year_df_end) <= int(year_var_end) and int(month_df_end) < int(month_var_end): # If the variable ends after the index
                last_row = index_df.loc[year_df_end+"-"+month_df_end:].index[0]
                variable = variable.loc[year_var_first+"-"+month_var_first:last_row]

                last_row = index_df.loc[year_df_end+"-"+month_df_end:].index[0]
                index_df_cov = index_df.loc[year_var_first+"-"+month_var_first:last_row] 

                if index_df_cov.index[-1].split("-")[1] != variable.index[-1].split("-")[1]: # If slice ends in different month
                    index_df_cov = index_df_cov[:-1]
                
                # Substract the sample mean 
                index_df_cov = index_df_cov - index_df_cov.mean()
                variable = variable - variable.mean()
                coef = np.corrcoef(index_df_cov, variable)[0][1]

            else: # If the variable ends before the index
                last_row = index_df.loc[year_var_end+"-"+month_var_end:].index[0]
                index_df_cov = index_df.loc[year_var_first+"-"+month_var_first:last_row]
                # Substract the sample mean 
                index_df_cov = index_df_cov - index_df_cov.mean()
                variable = variable - variable.mean()
                coef = np.corrcoef(index_df_cov, variable)[0][1]

        pearson_coeffs.append(coef)

    pearson_df.loc[len(pearson_df)] = [pearson_coeffs[0], pearson_coeffs[1], pearson_coeffs[2], pearson_coeffs[3]]
    return pearson_df

def compute_t_test(index_df, variable, significance_level=0.05):
    t_stat, p_val = ttest_ind(index_df, variable)
    print(f"t-statistic: {t_stat}")
    print(f"P-value: {p_val}")

    if p_val < significance_level:
        # Reject the null hypothesis
        print("The means differ significantly, correlation is significant and tells us the direction")
    else:
        # Accept the null hypothesis
        print("There is so significant difference between the means, correlation is not significant")
    return t_stat, p_val


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
        model = SARIMAX(train, trend='n', order=order,  
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

def test_model(test, test_exog, model=None, seasonal=False, index='wine'): # Testing data
    if model == None and seasonal == False: # ARIMA Model
        model = ARIMAResults.load(f'models\{index}_arimax.pkl')
    
    elif model == None and seasonal == True: # SARIMA Model
        model = ARIMAResults.load(f'models\{index}_sarimax.pkl')

    # Testing Forecast make sure test_exog has same length as test.shape[0]
    forecast_steps = test.shape[0]
    forecast = model.get_forecast(steps=forecast_steps, exog=test_exog)
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

def split_cross_validation(data, order, exog, index='wine', seasonal_order=None, seasonal=False):
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

        split_exog = exog[:int(split*len(exog))]
        exog_train = split_exog[:int(0.8*len(split_exog))]
        exog_test = split_exog[int(0.8*len(split_exog)):]

        fit_results, _ = create_model(train, order, exog_train, seasonal_order, index)
        _, mae, mse, mae_baseline, mse_baseline, mae_baseline_mean, mse_baseline_mean, rmse, rmse_baseline, rmse_baseline_mean, mape, mape_baseline, mape_baseline_mean = test_model(test, exog_test, fit_results, seasonal, index)

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

def evaluate_model_with_Plots(data, candidates, eval_df, exog, seasonal=False, index='wine', arima_order=None): 
    # Take the model with the lowest eval metrics and errors
    for candidate in candidates:
        if seasonal == False:
            # Split cross validation
            aic, bic, mae, mse, rmse, mape, mae_bas, mse_bas, rmse_bas, mape_bas, mae_mean, mse_mean, rmse_mean, mape_mean = split_cross_validation(data, candidate, exog, index, None, seasonal)
            
            # Store evaluation information (those are already avg calculated in the split cross validation function)
            eval_df.loc[len(eval_df)] = [candidate, None, aic, bic, mae, mse, rmse, mape]
        
        else:
            # Split cross validation
            aic, bic, mae, mse, rmse, mape, mae_bas, mse_bas, rmse_bas, mape_bas, mae_mean, mse_mean, rmse_mean, mape_mean = split_cross_validation(data, order=arima_order, exog=exog, index=index, seasonal_order=candidate, seasonal=seasonal)

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

def align_data(index_df, exog):
    start_date = index_df.index[0].split("-")
    year_df_first = start_date[0]
    month_df_first = start_date[1]
    end_date = index_df.index[-1].split("-")
    year_df_end = end_date[0]
    month_df_end = end_date[1]

    last_date = exog.index[-1].split("-")
    year_var_end = last_date[0]
    month_var_end = last_date[1]
    first_date = exog.index[0].split("-")
    year_var_first = first_date[0]
    month_var_first = first_date[1]

    if (len(exog) > len(index_df)): # If the variable has more data than the index and ends later
        if int(year_var_end) < int(year_df_end): # If the variable ends before the index
            last_row = exog.loc[year_var_end+"-"+month_var_end:].index[0]
            exog = exog.loc[year_df_first+"-"+month_df_first:last_row]

            last_row = index_df.loc[year_var_end+"-"+month_var_end:].index[0]
            index_df = index_df.loc[year_df_first+"-"+month_df_first:last_row]
            
        else:
            last_row = index_df.loc[year_df_end+"-"+month_df_end:].index[0]
            exog = exog.loc[year_df_first+"-"+month_df_first:last_row]

    if len(exog) < len(index_df): # If the variable has less data than the index
        if int(year_df_end) <= int(year_var_end) and int(month_df_end) < int(month_var_end): # If the variable ends after the index
            last_row = index_df.loc[year_df_end+"-"+month_df_end:].index[0]
            exog = exog.loc[year_var_first+"-"+month_var_first:last_row]

            last_row = index_df.loc[year_df_end+"-"+month_df_end:].index[0]
            index_df = index_df.loc[year_var_first+"-"+month_var_first:last_row] 

            if index_df.index[-1].split("-")[1] != exog.index[-1].split("-")[1]: # If slice ends in different month
                index_df = index_df[:-1]   

        else: # If the variable ends before the index
            last_row = index_df.loc[year_var_end+"-"+month_var_end:].index[0]
            index_df = index_df.loc[year_var_first+"-"+month_var_first:last_row]
            
    
    # They now have the same length, but not exactly the same dates day for day
    exog.index = index_df.index

    return index_df, exog

def forecast_model(data, test, exog_data, forecast_steps, length, end_date, model=None, seasonal=False, index='wine'):
    if model == None and seasonal == False: # ARIMAX Model
        model = ARIMAResults.load(f'models\{index}_arimax.pkl')
    elif model == None and seasonal == True: # SARIMAX Model
        model = ARIMAResults.load(f'models\{index}_sarimax.pkl')

    forecast = model.get_forecast(steps=forecast_steps, exog=exog_data)
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
    plt.title(f'{length} term forecast of {index} index values using exogenous variable')
    plt.xlabel('Time')
    plt.ylabel('Index value')
    plt.show()

    return yhat

##### MAIN #####

## Load the data from pre-processing ##
wine_df, watch_df, art_df, gold_df, sp500_df, cpi_df, bond_yield_df = preprocessing.main(univariate=False)

# Compute covariance matrix between each pair of correlated variables and index
# Log transform the data to stabilize variance and get more accurate covariance
wine_cov = pd.DataFrame(columns = ["Gold", "SP500", "CPI", "Bond Yield"])
wine_cov = compute_covariance(wine_cov, np.log(wine_df.observed), [np.log(gold_df.observed), np.log(sp500_df.observed), np.log(cpi_df.observed), bond_yield_df.observed])
watch_cov = pd.DataFrame(columns = ["Gold", "SP500", "CPI", "Bond Yield"])
watch_cov = compute_covariance(watch_cov, np.log(watch_df.observed), [np.log(gold_df.observed), np.log(sp500_df.observed), np.log(cpi_df.observed), bond_yield_df.observed])
art_cov = pd.DataFrame(columns = ["Gold", "SP500", "CPI", "Bond Yield"])
art_cov = compute_covariance(art_cov, np.log(art_df.observed), [ np.log(gold_df.observed), np.log(sp500_df.observed), np.log(cpi_df.observed), bond_yield_df.observed])
# print(wine_cov)
# print(watch_cov)
# print(art_cov)

# Big covariance between Gold and Wine, Crypto and Watch, SP500 and Art
# Bond Yield has a negative covariance with all indexes, and is a bit biased, because I cannot log transform it since it has negative values

# Compute Pearson correlation coefficient between each pair of correlated variables and index
# Log transform the data to stabilize variance and get more accurate coefficient
wine_pearson_coeff = pd.DataFrame(columns = ["Gold", "SP500", "CPI", "Bond Yield"])
wine_pearson_coeff = compute_pearson_coeff(wine_pearson_coeff, np.log(wine_df.observed), [np.log(gold_df.observed), np.log(sp500_df.observed), np.log(cpi_df.observed), bond_yield_df.observed])
watch_pearson_coeff = pd.DataFrame(columns = ["Gold", "SP500", "CPI", "Bond Yield"])
watch_pearson_coeff = compute_pearson_coeff(watch_pearson_coeff, np.log(watch_df.observed), [np.log(gold_df.observed), np.log(sp500_df.observed), np.log(cpi_df.observed), bond_yield_df.observed])
art_pearson_coeff = pd.DataFrame(columns = ["Gold", "SP500", "CPI", "Bond Yield"])
art_pearson_coeff = compute_pearson_coeff(art_pearson_coeff, np.log(art_df.observed), [np.log(gold_df.observed), np.log(sp500_df.observed), np.log(cpi_df.observed), bond_yield_df.observed])
# print(wine_pearson_coeff)
# print(watch_pearson_coeff)
# print(art_pearson_coeff)

# Big correlation between Gold and Wine, watch and CPI, SP500+CPI and Art
# Bond Yield has a negative correlation with all indexes, and is a bit biased, because I cannot log transform it since it has negative values

# Test the significance of the correlation coefficient with a t-test (two sample), alternative: Mann-Whitney U test (which is non-parametric)
# compute_t_test(np.log(wine_df.observed), np.log(gold_df.observed)) # Significant correlation
# compute_t_test(np.log(art_df.observed), np.log(sp500_df.observed)) # Significant correlation
# compute_t_test(np.log(wine_df.observed), np.log(cpi_df.observed)) # significant correlation
# compute_t_test(np.log(watch_df.observed), np.log(cpi_df.observed)) # significant correlation

# NB forecasting using exogenous variables requires future exog data to be known in advance which is not the case.
# Thus I need to forecast the exogenous variables as well using ARIMA, and use that forecast as input for the main forecast of our index.

# WINE
arima_wine = (3,1,3)
wine_adjusted, exog_wine = align_data(wine_df.observed, gold_df.observed)

wine_train = wine_adjusted[:int(0.8*len(wine_adjusted))]
wine_test = wine_adjusted[int(0.8*len(wine_adjusted)):]

wine_train_exog = exog_wine[:int(0.8*len(exog_wine))]
wine_test_exog = exog_wine[int(0.8*len(exog_wine)):]

# Evaluate the model
# eval_df = pd.DataFrame(columns=["ARIMA", "SEASONAL", "AIC", "BIC", "MAE", "MSE", "RMSE", "MAPE %"]) # To store the most important evaluation metrics
# eval_df = evaluate_model_with_Plots(wine_adjusted, [arima_wine], eval_df, exog_wine, seasonal=False, index='wine')
# print(eval_df)

# Save optimal model
# wine_model_exog = create_model(wine_adjusted, arima_wine, exog_wine, index='wine') 

# Forecast
# Now that the optimal has been found, use it to forecast
long_term = wine_train.shape[0] # Full training set can go beyond that but it would be extrapolation, so less reliable

# Long term forecasts
ref_start = wine_adjusted.index[-1] # "2022-07-31"
end_long = "2035-02-28"
# Not applicable yet
# forecast_model(wine_adjusted, wine_test, wine_train_exog, long_term, "Long", end_date=end_long, model=None, seasonal=False, index='wine')

# WATCH
arima_watch = (2,1,3)
watch_adjusted, exog_watch = align_data(watch_df.observed, cpi_df.observed)

watch_train = watch_adjusted[:int(0.8*len(watch_adjusted))]
watch_test = watch_adjusted[int(0.8*len(watch_adjusted)):]
                   
watch_train_exog = exog_watch[:int(0.8*len(exog_watch))]
watch_test_exog = exog_watch[int(0.8*len(exog_watch)):]

# Evaluate the model
# eval_df = pd.DataFrame(columns=["ARIMA", "SEASONAL", "AIC", "BIC", "MAE", "MSE", "RMSE", "MAPE %"]) # To store the most important evaluation metrics
# eval_df = evaluate_model_with_Plots(watch_adjusted, [arima_watch], eval_df, exog_watch, seasonal=False, index='watch')
# print(eval_df)

# Save optimal model
# watch_model_exog = create_model(watch_adjusted, arima_watch, exog_watch, index='watch')

# Forecast
# Now that the optimal has been found, use it to forecast
long_term = watch_train.shape[0] # Full training set can go beyond that but it would be extrapolation, so less reliable

# Long term forecasts
ref_start = watch_adjusted.index[-1] # "2023-12-01"
end_long = "2034-02-01"
# Not applicable yet
# forecast_model(watch_adjusted, watch_test, watch_train_exog, long_term, "Long", end_date=end_long, model=None, seasonal=False, index='watch')

# ART
arima_art = (13,1,6)
art_adjusted, exog_art = align_data(art_df.observed, cpi_df.observed) 

# Evaluate the model
# eval_df = pd.DataFrame(columns=["ARIMA", "SEASONAL", "AIC", "BIC", "MAE", "MSE", "RMSE", "MAPE %"]) # To store the most important evaluation metrics
# eval_df = evaluate_model_with_Plots(art_adjusted, [arima_art], eval_df, exog_art, seasonal=False, index='art')
# print(eval_df) # Best variable for ARIMAX art is CPI

# Since SARIMA > ARIMA for Art, evaluate SARIMAX 
sarima_art = [(4,1,2),(5,0,6,6)]
# art_adjusted, exog_art = align_data(art_df.observed, sp500_df.observed)
# eval_df = pd.DataFrame(columns=["ARIMA", "SEASONAL", "AIC", "BIC", "MAE", "MSE", "RMSE", "MAPE %"]) # To store the most important evaluation metrics
# eval_df = evaluate_model_with_Plots(art_adjusted, [sarima_art[1]], eval_df, exog_art, seasonal=True, index='art', arima_order=sarima_art[0])
# print(eval_df) # Best variable for SARIMAX art is SP500

art_train = art_adjusted[:int(0.8*len(art_adjusted))]
art_test = art_adjusted[int(0.8*len(art_adjusted)):]
               
forecast_exog = exog_art[int(0.8*len(exog_art)):]
exog_mean = forecast_exog.mean()
fill = [exog_mean for i in range(len(art_train) - len(forecast_exog))]
forecast_exog = pd.concat([forecast_exog, pd.Series(fill)])

# Save optimal model
# art_model_exog = create_model(art_adjusted, arima_art, exog_art, index='art')
# seasonal_art_model_exog = create_model(art_adjusted, sarima_art[0], exog_art, seasonal_order=sarima_art[1], index='art')

# Forecast
# Now that the optimal has been found, use it to forecast
long_term = art_train.shape[0] # Full training set can go beyond that but it would be extrapolation, so less reliable
# Long term forecasts
ref_start = art_adjusted.index[-1] # "2023-09-01"
end_long = "2051-02-01"
# Not applicable yet
# forecast_model(art_adjusted, art_test, forecast_exog, long_term, "Long", end_date=end_long, model=None, seasonal=True, index='art')