import csv
import pandas as pd
import matplotlib.pyplot as plt
import json

# TODO: Determine the currency of the art index
# TODO: Make sure all data is on the same monetary EUR unit

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


if __name__ == "__main__":
    # Watch EUR
    watch_df = get_watch_data('Data_retrieval\data\Watches\Watch_Index.csv')
    # plt.plot(watch_df['Date'], watch_df['Index Value'])
    # plt.xlabel('Date')
    # plt.ylabel('Index Value (Monthly Average)')
    # plt.title('(Custom Weighted) Watch Index Monthly Average')
    # plt.show()

    # Wine GBP
    wine_df = get_wine_data('Data_retrieval\data\Wine\Liv-Ex 100 Index.csv')
    # plt.plot(wine_df['Date'], wine_df['Index Value'])
    # plt.xlabel('Date')
    # plt.ylabel('Index Value')
    # plt.title('Liv-Ex 100 Index (Monthly Average)')
    # plt.show()

    # Art (Currency unknown for the moment)
    art_df = get_art_data('Data_retrieval\data\Art\All Art Index Family\index_values.json')
    # plt.plot(art_df['Date'], art_df['Index Value'])
    # plt.xlabel('Date')
    # plt.ylabel('Index Value')
    # plt.title('All Art Index Family (Monthly Average)')
    # plt.show()


