import csv
import pandas as pd
import matplotlib.pyplot as plt
import json
import math

# TODO : Log transform the data and the correlated variables
# TODO : Might need to adjust data from inflation, especially correlated variables

##### INDEX DATA #####

# Open watch data (done with pd.dataframe but you can also do it with arrays)
def get_watch_data(path):
    # Create a dataframe
    df = pd.read_csv(path)
    return df

# Open wine data (done with pd.dataframe but you can also do it with arrays)
def get_wine_data(path):
    # Create a dataframe
    df = pd.read_csv(path)
    return df

# Open art data
# Index Methodology : https://www.artmarketresearch.com/all-art-methodology/
# The art index is already inflation-adjusted
def get_art_data(path): # pd.read_json not working due to complex json file
    with open(path, 'r') as f:
        data = json.load(f)

        # Extract columns and rows from the JSON data
        cols_data = data['cols']
        rows_data = data['rows']

        # Extract labels from columns data
        column_labels = [col['label'] for col in cols_data]

        # Extract data from rows
        row_values = [[row['c'][i]['v'] for i in range(len(cols_data))] for row in rows_data]

        # Create pandas DataFrame
        df = pd.DataFrame(row_values, columns=column_labels)
        return df.drop(columns=['Drop this column'])
    
# Open GBP and convert to yearly rates   
def get_EURGBP_rates_yearly(path):
    df = pd.read_csv(path)

    year_list = []
    rate_list = []
    
    temp_year = []
    temp_year_values = []

    df.set_index('Date', inplace=True) # set date as index

    for date, value in df.iterrows():
        year = date.split('-')[0]
        if (year not in temp_year) and (temp_year != []):
                rate_list.append(sum(temp_year_values)/len(temp_year_values)) # Average rate for the year
                year_list.append(temp_year[0])

                temp_year = []
                temp_year_values = []

        temp_year.append(year)
        temp_year_values.append(value)

    yearly_average_df = pd.DataFrame({'Year': year_list, 'Rate': rate_list})
    return yearly_average_df

# Open EUR USD and convert to yearly rates
def get_EURUSD_rates_yearly(path):
    df = pd.read_csv(path)

    year_list = []
    rate_list = []
    
    temp_year = []
    temp_year_values = []

    df.set_index('Date', inplace=True) # set date as index

    for date, value in df.iterrows():
        year = date.split('-')[0]
        if (year not in temp_year) and (temp_year != []):
                rate_list.append(sum(temp_year_values)/len(temp_year_values)) # Average rate for the year
                year_list.append(temp_year[0])

                temp_year = []
                temp_year_values = []

        temp_year.append(year)
        temp_year_values.append(value)

    yearly_average_df = pd.DataFrame({'Year': year_list, 'Rate': rate_list})
    return yearly_average_df

# Open GBP USD and convert to yearly rates
def get_GBPUSD_rates_yearly(path):
    df = pd.read_csv(path)

    year_list = []
    rate_list = []
    
    temp_year = []
    temp_year_values = []

    df.set_index('Date', inplace=True) # set date as index

    for date, value in df.iterrows():
        year = date.split('-')[0]
        if (year not in temp_year) and (temp_year != []):
                rate_list.append(sum(temp_year_values)/len(temp_year_values)) # Average rate for the year
                year_list.append(temp_year[0])

                temp_year = []
                temp_year_values = []

        temp_year.append(year)
        temp_year_values.append(value)

    yearly_average_df = pd.DataFrame({'Year': year_list, 'Rate': rate_list})
    return yearly_average_df

##### CURRENCY CONVERSION #####

def convert_to_GBP(df, currency_rates):
    gbp_values = []
    gbp_dates = []

    for index, value in df.iterrows():
        date = str(value.iloc[0])
        year = date.split('-')[0]
        value = value.iloc[1]

        # Take current GPB-EUR rate
        rate_index = 0
        for i, rate in currency_rates.iterrows():
            if rate.iloc[0] == year:
                rate_index = i
                break   
        rate = currency_rates.iloc[rate_index, 1]

        # Take EUR-GBP rate for the current iterating year and multiply the value by it
        gbp_val = value * rate

        gbp_values.append(gbp_val)
        gbp_dates.append(date)
        
    df_gbp = pd.DataFrame({'Date': gbp_dates, 'Index Value': gbp_values})
    return df_gbp

def convert_to_USD(df, currency_rates):
    usd_values = []
    usd_dates = []

    for index, value in df.iterrows():
        date = str(value.iloc[0])
        year = date.split('-')[0]
        value = value.iloc[1]

        # Take current GPB-EUR rate
        rate_index = 0
        for i, rate in currency_rates.iterrows():
            if rate.iloc[0] == year:
                rate_index = i
                break   
        rate = currency_rates.iloc[rate_index, 1]

        # Take EUR-GBP rate for the current iterating year and multiply the value by it
        usd_val = value * rate

        usd_values.append(usd_val)
        usd_dates.append(date)
        
    df_usd = pd.DataFrame({'Date': usd_dates, 'Index Value': usd_values})
    return df_usd

##### CORRELATED VARIABLES #####

def get_CPI_Index(path):
    df = pd.read_csv(path)
    return df

def get_monthly_global_crypto_market_cap(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['Drop_column'])

    average_monthly_marketcap = []
    date = []

    temp_month = []
    temp_month_values = []

    for index, value in df.iterrows():
        iter_date = str(value.iloc[0])
        current_month = str(iter_date).split('-')[1]
        market_cap = value.iloc[1]

        if (current_month not in temp_month) and (temp_month != []):
            amcp = (sum(temp_month_values)/len(temp_month_values))
            date.append(previous_date) # if we took iter_date, the month would have already changed
            average_monthly_marketcap.append(amcp)

            temp_month = []
            temp_month_values = []

        temp_month.append(current_month) 
        temp_month_values.append(market_cap)
        previous_date = iter_date # to make sure we get the right month

    df_average = pd.DataFrame({'Date': date, 'Market cap (monthly)': average_monthly_marketcap})
    return df_average

def get_monthly_gold_prices(path):
    df = pd.read_csv(path)

    average_monthly_prices = []
    date = []

    temp_month = []
    temp_month_values = []

    for index, value in df.iterrows():
        iter_date = str(value.iloc[0])
        current_month = str(iter_date).split('-')[1]
        price = value.iloc[1]

        if math.isnan(price) == True:
            continue

        if (current_month not in temp_month) and (temp_month != []):
            agp = (sum(temp_month_values)/len(temp_month_values))
            date.append(previous_date) # if we took iter_date, the month would have already changed
            average_monthly_prices.append(agp)

            temp_month = []
            temp_month_values = []
        
        temp_month.append(current_month)
        temp_month_values.append(price)
        previous_date = iter_date # to make sure we get the right month

    df_average = pd.DataFrame({'Date': date, 'Gold Price (monthly)': average_monthly_prices})
    return df_average

def get_sp500(path):
    df = pd.read_csv(path)
    return df

def get_US10_year_bond_yield(path):
    df = pd.read_csv(path)
    return df

##### ADJUST FOR INFLATION #####

def adjust_inflation(df, cpi_df):
    # Average CPI to Yearly rates instead of monthly rates
    year_list = []
    rate_list = []
    
    temp_year = []
    temp_year_values = []

    cpi_df.set_index('Date', inplace=True) # set date as index

    for date, value in cpi_df.iterrows():
        year = date.split('-')[0]
        if (year not in temp_year) and (temp_year != []):
                rate_list.append(sum(temp_year_values)/len(temp_year_values)) # Average rate for the year
                year_list.append(temp_year[0])

                temp_year = []
                temp_year_values = []

        temp_year.append(year)
        temp_year_values.append(value)

    yearly_average_df = pd.DataFrame({'Year': year_list, 'CPI_Index': rate_list})

    # Now adjust the data for inflation
    
    return 0

##### MAIN #####

# Retrieve all data and convert them to monthly time frames under the same currency (USD)

# Get GBP to EUR rates
GBP_rates = get_EURGBP_rates_yearly('data\GBP_EUR_Historical_Rates.csv')
# plt.plot(GBP_rates['Year'], GBP_rates['Rate'])
# plt.xlabel('Year')
# plt.ylabel('Rate')
# plt.title('EUR to GBP (Yearly Average)')
# plt.show()

# Get EUR to USD rates
USD_rates = get_EURUSD_rates_yearly('data\EUR_USD_Historical_Rates.csv')
# plt.plot(USD_rates['Year'], USD_rates['Rate'])
# plt.xlabel('Year')
# plt.ylabel('Rate')
# plt.title('EUR to USD (Yearly Average)')
# plt.show()

# Get GBP to USD rates
USD2_rates = get_GBPUSD_rates_yearly('data\GBP_USD_Historical_Rates.csv')
# plt.plot(USD2_rates['Year'], USD2_rates['Rate'])
# plt.xlabel('Year')
# plt.ylabel('Rate')
# plt.title('GBP to USD (Yearly Average)')
# plt.show()

# Watch EUR
watch_df_EUR = get_watch_data('data\Watches\Watch_Index.csv')
# plt.plot(watch_df_EUR['Date'], watch_df_EUR['Index Value'])
# plt.xlabel('Date')
# plt.ylabel('Index Value (Monthly Average)')
# plt.title('(Custom Weighted) Watch Index Monthly Average EUR')
# plt.show()

# EUR was created in 1999, so we need to convert to USD (Not inflation adjusted)
watch_df = convert_to_USD(watch_df_EUR, USD_rates)
# plt.plot(watch_df['Date'], watch_df['Index Value'])
# plt.xlabel('Date')
# plt.ylabel('Index Value (Monthly Average)')
# plt.title('(Custom Weighted) Watch Index Monthly Average USD')
# plt.show()

# Wine GBP
wine_df_GBP = get_wine_data('data\Wine\Liv-Ex 100 Index.csv')
# plt.plot(wine_df_GBP['Date'], wine_df_GBP['Index Value'])
# plt.xlabel('Date')
# plt.ylabel('Index Value')
# plt.title('Liv-Ex 100 Index (Monthly Average)')
# plt.show()

# Wine needs to be converted to USD (Not inflation adjusted)
wine_df = convert_to_USD(wine_df_GBP, USD2_rates)
# plt.plot(wine_df['Date'], wine_df['Index Value'])
# plt.xlabel('Date')
# plt.ylabel('Index Value')
# plt.title('Liv-Ex 100 Index (Monthly Average) USD')
# plt.show()

# Art GBP (already inflation adjusted)
art_df_GBP = get_art_data('data\Art\All Art Index Family\index_values.json')
# plt.plot(art_df_GBP['Date'], art_df_GBP['Index Value'])
# plt.xlabel('Date')
# plt.ylabel('Index Value')
# plt.title('All Art Index Family (Monthly Average)')
# plt.show()

# Art needs to be converted to USD
art_df = convert_to_USD(art_df_GBP, USD2_rates)
# plt.plot(art_df['Date'], art_df['Index Value'])
# plt.xlabel('Date')
# plt.ylabel('Index Value')
# plt.title('All Art Index Family (Monthly Average) USD')
# plt.show()

# CPI Index United States (USD) # Seasonally adjusted
cpi_df = get_CPI_Index(r'data\Correlated Variables\CPI Index\United States\CPI Data all items United States.csv')
# plt.plot(cpi_df['Date'], cpi_df['CPI_Index'])
# plt.xlabel('Date')
# plt.ylabel('Index Value')
# plt.title('Monthly CPI_Index (Inflation) United States')
# plt.show()

# Global Crypto Market Cap (USD)
crypto_df = get_monthly_global_crypto_market_cap(r'data\Correlated Variables\Crypto\global_marketcap.csv')
# plt.plot(crypto_df['Date'], crypto_df['Market cap (monthly)'])
# plt.xlabel('Date')
# plt.ylabel('Market cap (USD)')
# plt.title('Global Crypto Marketcap (Monthly Average)')
# plt.show()

# Gold Prices (USD)
gold_df = get_monthly_gold_prices(r'data\Correlated Variables\Gold Prices\data.csv')
# plt.plot(gold_df['Date'], gold_df['Gold Price (monthly)'])
# plt.xlabel('Date')
# plt.ylabel('Gold Price (USD)')
# plt.title('Gold Prices (Monthly Average)')
# plt.show()

# SP500 (USD) 
sp500_df = get_sp500(r'data\Correlated Variables\S&P 500\data.csv')
# plt.plot(sp500_df['Date'], sp500_df['Real'])
# plt.xlabel('Date')
# plt.ylabel('S&P 500 Index Value (REAL)')
# plt.title('S&P 500 Monthly (USD) Inflation Adjusted')
# plt.show()
# Now plot SP500 nominal (not inflation adjusted)
# plt.plot(sp500_df['Date'], sp500_df['Nominal'])
# plt.xlabel('Date')
# plt.ylabel('S&P 500 Index Value (NOMINAL)')
# plt.title('S&P 500 Monthly (USD) NOT Inflation Adjusted')
# plt.show()

# United States 10-Year Bond Yield (USD) in %
bond_yield_df = get_US10_year_bond_yield(r'data\Correlated Variables\United States 10 Years Government Bond Yield\data.csv')
# plt.plot(bond_yield_df['Date'], bond_yield_df['Rate_in_Percent'])
# plt.xlabel('Date')
# plt.ylabel('10 Year Bond Yield (%)')
# plt.title('United States 10-Year Bond Yield (Monthly) USD (%)')
# plt.show()

# Now adjust all the data for inflation
# NB : Art index is already adjusted for inflation
# cpi_df = adjust_inflation(cpi_df)


