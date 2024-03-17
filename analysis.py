import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import preprocessing 
from statsmodels.tsa.stattools import kpss

# TODO : Evaluate the stationarity of the data using correlogram

##### STATISTICAL TESTS #####

# KPSS Test for Stationarity
# Null Hypothesis: the data is stationary around a constant mean, exhibiting a linear trend component
# The Null Hypothesis is rejected if the p-value is less than the significance level
# The Null Hypotheis is rejected if the test statistic is greater than the critical value (5% significance level)
def is_stationary_with_kpss(data, significance_level=0.05):
    test = kpss(data, regression='ct') # 'ct' for constant and trend component more appropriate for financial time series than 'c' for constant only
    test_statistic = test[0]
    p_value = test[1]
    critical_value_ref = test[3]
    print("KPSS p-value: {:0.5f}".format(p_value))
    print("KPSS test statistic: {:0.5f}".format(test_statistic))
    print("KPSS critical value: ", critical_value_ref)
    return p_value > significance_level

##### MAIN #####

## Load the data from pre-processing ##
wine_df, watch_df, art_df, crypto_df, gold_df, sp500_df, cpi_df, bond_yield_df = preprocessing.main()

## Check for stationarity using Statistical Test and Correlogram ##
# Statistical Test : All data points are Non-Stationary

# Asset Data
print("TEST FOR STATIONARITY:", is_stationary_with_kpss(wine_df['Index Value'])) # Not stationary at 5% significance level
print("TEST FOR STATIONARITY:", is_stationary_with_kpss(watch_df['Index Value'])) # Not stationary at 5% significance level
print("TEST FOR STATIONARITY:", is_stationary_with_kpss(art_df['Index Value'])) # Not stationary at 5% significance level

# Correlated Variables
print("TEST FOR STATIONARITY:", is_stationary_with_kpss(crypto_df['Index Value'])) # Not stationary at 5% significance level
print("TEST FOR STATIONARITY:", is_stationary_with_kpss(gold_df['Index Value'])) # Not stationary at 5% significance level
print("TEST FOR STATIONARITY:", is_stationary_with_kpss(sp500_df['Real'])) # Not stationary at 5% significance level
print("TEST FOR STATIONARITY:", is_stationary_with_kpss(cpi_df['CPI_Index'])) # Not stationary at 5% significance level


