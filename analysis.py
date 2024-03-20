import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import preprocessing   
from statsmodels.tsa.seasonal import seasonal_decompose 

# TODO : Find paper to support the fact stationarity is irrelevant for correlation analysis
# TODO : Start correlation analysis

##### MAIN #####

## Load the data from pre-processing ##
wine_df, watch_df, art_df, crypto_df, gold_df, sp500_df, cpi_df, bond_yield_df = preprocessing.main()




