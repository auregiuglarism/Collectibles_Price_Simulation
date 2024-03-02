import csv
import pandas as pd
import matplotlib.pyplot as plt
import json

# TODO: Make sure all data is on the same monetary GBP unit

##### GETTER FUNCTIONS #####

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
def get_GBP_rates_yearly(path):
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
           
##### MAIN FUNCTION #####

if __name__ == "__main__":
    # Get GBP to EUR rates
    GBP_rates = get_GBP_rates_yearly('Data_retrieval\data\GBP_EUR_Historical_Rates.csv')
    # plt.plot(GBP_rates['Year'], GBP_rates['Rate'])
    # plt.xlabel('Year')
    # plt.ylabel('Rate')
    # plt.title('GBP to EUR (Yearly Average)')
    # plt.show()
    
    # Watch EUR
    watch_df_EUR = get_watch_data('Data_retrieval\data\Watches\Watch_Index.csv')
    # plt.plot(watch_df_EUR['Date'], watch_df_EUR['Index Value'])
    # plt.xlabel('Date')
    # plt.ylabel('Index Value (Monthly Average)')
    # plt.title('(Custom Weighted) Watch Index Monthly Average EUR')
    # plt.show()

    # EUR was created in 1999, so we need to convert to GBP
    watch_df_GBP = convert_to_GBP(watch_df_EUR, GBP_rates)
    # plt.plot(watch_df_GBP['Date'], watch_df_GBP['Index Value'])
    # plt.xlabel('Date')
    # plt.ylabel('Index Value (Monthly Average)')
    # plt.title('(Custom Weighted) Watch Index Monthly Average GBP')
    # plt.show()

    # Wine GBP
    wine_df = get_wine_data('Data_retrieval\data\Wine\Liv-Ex 100 Index.csv')
    # plt.plot(wine_df['Date'], wine_df['Index Value'])
    # plt.xlabel('Date')
    # plt.ylabel('Index Value')
    # plt.title('Liv-Ex 100 Index (Monthly Average)')
    # plt.show()

    # Art (GBP)
    art_df = get_art_data('Data_retrieval\data\Art\All Art Index Family\index_values.json')
    # plt.plot(art_df['Date'], art_df['Index Value'])
    # plt.xlabel('Date')
    # plt.ylabel('Index Value')
    # plt.title('All Art Index Family (Monthly Average)')
    # plt.show()



    

    


