import csv
import pandas as pd

# NB: Make sure the csv file is still in complete unprocessed form before running file.

# Changes month names into numbers
def month_numbering(data):
    for row in data['date']:
        current_date = row
        # Replace month names with numbers
        if current_date[0:3] == 'Jan':
            current_date = current_date.replace('Jan', '01')
            current_date = current_date.replace(' ', '-')

        elif current_date[0:3] == 'Feb':
            current_date = current_date.replace('Feb', '02')
            current_date = current_date.replace(' ', '-')
        
        elif current_date[0:3] == 'Mar':
            current_date = current_date.replace('Mar', '03')
            current_date = current_date.replace(' ', '-')

        elif current_date[0:3] == 'Apr':
            current_date = current_date.replace('Apr', '04')
            current_date = current_date.replace(' ', '-')

        elif current_date[0:3] == 'May':
            current_date = current_date.replace('May', '05')
            current_date = current_date.replace(' ', '-')

        elif current_date[0:3] == 'Jun':
            current_date = current_date.replace('Jun', '06')
            current_date = current_date.replace(' ', '-')

        elif current_date[0:3] == 'Jul':
            current_date = current_date.replace('Jul', '07')
            current_date = current_date.replace(' ', '-')

        elif current_date[0:3] == 'Aug':
            current_date = current_date.replace('Aug', '08')
            current_date = current_date.replace(' ', '-')

        elif current_date[0:3] == 'Sep':
            current_date = current_date.replace('Sep', '09')
            current_date = current_date.replace(' ', '-')

        elif current_date[0:3] == 'Oct':
            current_date = current_date.replace('Oct', '10')
            current_date = current_date.replace(' ', '-')

        elif current_date[0:3] == 'Nov':
            current_date = current_date.replace('Nov', '11')
            current_date = current_date.replace(' ', '-')

        elif current_date[0:3] == 'Dec':
            current_date = current_date.replace('Dec', '12')
            current_date = current_date.replace(' ', '-')

        # Write the new date back to the DataFrame
        data['date'] = data['date'].replace(row, current_date)

    # Save the updated DataFrame back to a CSV file
    data.to_csv('Data_retrieval\data\Watches\Rolex Daytona 116520 Price History.csv', index=False) # Change path if needed

# Reorders the date from 'mm-dd-yyyy' to 'yyyy-mm-dd'
def date_reordering(data):
    for row in data['date']:
        current_date = row
        # Reorder the date from 'mm-dd-yyyy' to 'yyyy-mm-dd'
        current_date = current_date.split('-')
        current_date = current_date[2] + '-' + current_date[0] + '-' + current_date[1]

        # Write the new date back to the DataFrame
        data['date'] = data['date'].replace(row, current_date)

    # Save the updated DataFrame back to a CSV file
    data.to_csv('Data_retrieval\data\Watches\Rolex Daytona 116520 Price History.csv', index=False) # Change path if needed
        

"""Filling in Rolex Daytona 116500""" ## DONE
# data = pd.read_csv('Data_retrieval\data\Watches\Rolex Daytona 116500 Price History.csv')
# month_numbering(data)
# data = pd.read_csv('Data_retrieval\data\Watches\Rolex Daytona 116500 Price History.csv')
# date_reordering(data)

# # Read the existing CSV file into a pandas DataFrame
# existing_data = pd.read_csv('Data_retrieval\data\Watches\Rolex Daytona 116500 Price History.csv')

# # Create a new DataFrame or series with the missing dates and the desired price
# desired_price_1 = 11171  
# missing_dates_1 = pd.date_range(start='2016-03-16', end='2018-05-14', freq='D')
# missing_data_1 = pd.DataFrame({'date': missing_dates_1, 'Rolex 116508 (EUR)': desired_price_1})

# desired_price_2 = 17767
# missing_dates_2 = pd.date_range(start='2018-05-15', end='2019-02-05', freq='D')
# missing_data_2 = pd.DataFrame({'date': missing_dates_2, 'Rolex 116508 (EUR)': desired_price_2})

# # Concatenate the missing data points together : 
# missing_data = pd.concat([missing_data_1, missing_data_2], ignore_index=True)

# Concatenate the existing DataFrame with the new one
# updated_data = pd.concat([missing_data, existing_data], ignore_index=True)
    
# Sort the DataFrame by date
# # updated_data.sort_values(by='date', inplace=True)

# # Save the updated DataFrame back to a CSV file
# updated_data.to_csv('Data_retrieval\data\Watches\Rolex Daytona 116500 Price History.csv', index=False)

"""Filling in Patek Philippe 5711/1A""" ## DONE
# data = pd.read_csv('Data_retrieval\data\Watches\Patek Philippe 5711-1A Price History.csv')
# month_numbering(data)
# data = pd.read_csv('Data_retrieval\data\Watches\Patek Philippe 5711-1A Price History.csv')
# date_reordering(data)

# # Read the existing CSV file into a pandas DataFrame
# existing_data = pd.read_csv('Data_retrieval\data\Watches\Patek Philippe 5711-1A Price History.csv')

# # Create a new DataFrame or series with the missing dates and the desired price
# desired_price_1 = 17605  
# missing_dates_1 = pd.date_range(start='2012-03-11', end='2012-06-13', freq='D')
# missing_data_1 = pd.DataFrame({'date': missing_dates_1, 'Rolex 116508 (EUR)': desired_price_1})

# desired_price_2 = 16997
# missing_dates_2 = pd.date_range(start='2012-06-14', end='2012-11-27', freq='D')
# missing_data_2 = pd.DataFrame({'date': missing_dates_2, 'Rolex 116508 (EUR)': desired_price_2})

# desired_price_3 = 19903
# missing_dates_3 = pd.date_range(start='2012-11-28', end='2013-06-22', freq='D')
# missing_data_3 = pd.DataFrame({'date': missing_dates_3, 'Rolex 116508 (EUR)': desired_price_3})

# desired_price_4 = 19608
# missing_dates_4 = pd.date_range(start='2013-06-23', end='2014-06-11', freq='D')
# missing_data_4 = pd.DataFrame({'date': missing_dates_4, 'Rolex 116508 (EUR)': desired_price_4})

# desired_price_5 = 20313
# missing_dates_5 = pd.date_range(start='2014-06-12', end='2015-11-09', freq='D')
# missing_data_5 = pd.DataFrame({'date': missing_dates_5, 'Rolex 116508 (EUR)': desired_price_5})

# desired_price_6 = 34715
# missing_dates_6 = pd.date_range(start='2015-11-10', end='2016-03-16', freq='D')
# missing_data_6 = pd.DataFrame({'date': missing_dates_6, 'Rolex 116508 (EUR)': desired_price_6})

# desired_price_7 = 22530
# missing_dates_7 = pd.date_range(start='2016-03-17', end='2016-12-06', freq='D')
# missing_data_7 = pd.DataFrame({'date': missing_dates_7, 'Rolex 116508 (EUR)': desired_price_7})

# desired_price_8 = 44253
# missing_dates_8 = pd.date_range(start='2016-12-07', end='2017-05-15', freq='D')
# missing_data_8 = pd.DataFrame({'date': missing_dates_8, 'Rolex 116508 (EUR)': desired_price_8})

# desired_price_9 = 51404
# missing_dates_9 = pd.date_range(start='2017-05-16', end='2017-12-07', freq='D')
# missing_data_9 = pd.DataFrame({'date': missing_dates_9, 'Rolex 116508 (EUR)': desired_price_9})

# desired_price_10 = 37111
# missing_dates_10 = pd.date_range(start='2017-12-08', end='2018-03-22', freq='D')
# missing_data_10 = pd.DataFrame({'date': missing_dates_10, 'Rolex 116508 (EUR)': desired_price_10})

# desired_price_11 = 38583
# missing_dates_11 = pd.date_range(start='2018-03-23', end='2018-06-13', freq='D')
# missing_data_11 = pd.DataFrame({'date': missing_dates_11, 'Rolex 116508 (EUR)': desired_price_11})

# desired_price_12 = 42423
# missing_dates_12 = pd.date_range(start='2018-06-14', end='2018-12-06', freq='D')
# missing_data_12 = pd.DataFrame({'date': missing_dates_12, 'Rolex 116508 (EUR)': desired_price_12})

# desired_price_13 = 41770
# missing_dates_13 = pd.date_range(start='2018-12-07', end='2019-02-05', freq='D')
# missing_data_13 = pd.DataFrame({'date': missing_dates_13, 'Rolex 116508 (EUR)': desired_price_13})

# # Concatenate the missing data points together : 
# missing_data = pd.concat([missing_data_1, missing_data_2, missing_data_3, 
#                           missing_data_4, missing_data_5, missing_data_6, missing_data_7,
#                           missing_data_8, missing_data_9, missing_data_10, missing_data_11,
#                           missing_data_12, missing_data_13], ignore_index=True)

# # Concatenate the existing DataFrame with the new one
# updated_data = pd.concat([missing_data, existing_data], ignore_index=True)

# # Sort the DataFrame by date
# # updated_data.sort_values(by='date', inplace=True)

# # Save the updated DataFrame back to a CSV file
# updated_data.to_csv('Data_retrieval\data\Watches\Patek Philippe 5711-1A Price History.csv', index=False)

"""Filling in Rolex Datejust 126334""" ## DONE
# data = pd.read_csv('Data_retrieval\data\Watches\Rolex DateJust 126334 Price History.csv')
# month_numbering(data)
# data = pd.read_csv('Data_retrieval\data\Watches\Rolex DateJust 126334 Price History.csv')
# date_reordering(data)

# # Read the existing CSV file into a pandas DataFrame
# existing_data = pd.read_csv('Data_retrieval\data\Watches\Rolex DateJust 126334 Price History.csv')

# # Create a new DataFrame or series with the missing dates and the desired price
# desired_price_1 = 5586
# missing_dates_1 = pd.date_range(start='2017-03-26', end='2019-02-05', freq='D')
# missing_data_1 = pd.DataFrame({'date': missing_dates_1, 'Rolex 116508 (EUR)': desired_price_1})

# # Concatenate the missing data points together : 
# missing_data = missing_data_1

# # Concatenate the existing DataFrame with the new one
# updated_data = pd.concat([missing_data, existing_data], ignore_index=True)
   
# # Sort the DataFrame by date
# # updated_data.sort_values(by='date', inplace=True)

# # Save the updated DataFrame back to a CSV file
# updated_data.to_csv('Data_retrieval\data\Watches\Rolex DateJust 126334 Price History.csv', index=False)

"""Filling in Rolex Daytona 116508""" ## DONE
# data = pd.read_csv('Data_retrieval\data\Watches\Rolex Daytona 116508 Price History.csv')
# month_numbering(data)
# data = pd.read_csv('Data_retrieval\data\Watches\Rolex Daytona 116508 Price History.csv')
# date_reordering(data)

# # Read the existing CSV file into a pandas DataFrame
# existing_data = pd.read_csv('Data_retrieval\data\Watches\Rolex Daytona 116508 Price History.csv')

# # Create a new DataFrame or series with the missing dates and the desired price
# desired_price_1 = 20478 
# missing_dates_1 = pd.date_range(start='2016-03-17', end='2017-01-01', freq='D')
# missing_data_1 = pd.DataFrame({'date': missing_dates_1, 'Rolex 116508 (EUR)': desired_price_1})

# desired_price_2 = 20721 
# missing_dates_2 = pd.date_range(start='2017-01-02', end='2018-01-01', freq='D')
# missing_data_2 = pd.DataFrame({'date': missing_dates_2, 'Rolex 116508 (EUR)': desired_price_2})

# desired_price_3 = 22939 
# missing_dates_3 = pd.date_range(start='2018-01-02', end='2019-02-05', freq='D')
# missing_data_3 = pd.DataFrame({'date': missing_dates_3, 'Rolex 116508 (EUR)': desired_price_3})

# # Concatenate the missing data points together : 
# missing_data = pd.concat([missing_data_1, missing_data_2, missing_data_3], ignore_index=True)

# # Concatenate the existing DataFrame with the new one
# updated_data = pd.concat([missing_data, existing_data], ignore_index=True)
  
# # Sort the DataFrame by date
# # updated_data.sort_values(by='date', inplace=True)# Save the updated DataFrame back to a CSV file

# updated_data.to_csv('Data_retrieval\data\Watches\Rolex Daytona 116508 Price History.csv', index=False)

"""Filling in Rolex Daytona 116520""" 
data = pd.read_csv('Data_retrieval\data\Watches\Rolex Daytona 116520 Price History.csv')
month_numbering(data)
data = pd.read_csv('Data_retrieval\data\Watches\Rolex Daytona 116520 Price History.csv')
date_reordering(data)

# Read the existing CSV file into a pandas DataFrame
existing_data = pd.read_csv('Data_retrieval\data\Watches\Rolex Daytona 116520 Price History.csv')

# Create a new DataFrame or series with the missing dates and the desired price
desired_price_1 = 10541 
missing_dates_1 = pd.date_range(start='2006-10-12', end='2007-05-31', freq='D')
missing_data_1 = pd.DataFrame({'date': missing_dates_1, 'Rolex 116520 (EUR)': desired_price_1})

desired_price_2 = 10313
missing_dates_2 = pd.date_range(start='2007-06-1', end='2007-12-11', freq='D')
missing_data_2 = pd.DataFrame({'date': missing_dates_2, 'Rolex 116520 (EUR)': desired_price_2})

desired_price_3 = 9055
missing_dates_3 = pd.date_range(start='2007-12-12', end='2008-05-12', freq='D')
missing_data_3 = pd.DataFrame({'date': missing_dates_3, 'Rolex 116520 (EUR)': desired_price_3})

# CONTINUE ADDING AUCTION DATA TO FILL UNTIL 2019

# Concatenate the missing data points together : 
missing_data = pd.concat([missing_data_1, missing_data_2, missing_data_3], ignore_index=True)

# Concatenate the existing DataFrame with the new one
updated_data = pd.concat([missing_data, existing_data], ignore_index=True)

# Sort the DataFrame by date
# updated_data.sort_values(by='date', inplace=True)# Save the updated DataFrame back to a CSV fil

updated_data.to_csv('Data_retrieval\data\Watches\Rolex Daytona 116520 Price History.csv', index=False)