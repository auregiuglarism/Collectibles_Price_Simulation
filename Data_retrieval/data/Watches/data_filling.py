import csv
import pandas as pd

# to delete later
# Wed Mar 16 2016,11171 # Additional
# Mon May 14 2018,17767 # Additional

# Chat GPT Example
# Read the existing CSV file into a pandas DataFrame
existing_data = pd.read_csv('existing_data.csv')

# Create a new DataFrame or series with dates from 2016 to 2018 and fill it with the desired price
desired_price = 100  # Change this to your desired price
missing_dates = pd.date_range(start='2016-01-01', end='2018-01-01', freq='D')
missing_data = pd.DataFrame({'Date': missing_dates, 'Price': desired_price})

# Concatenate the existing DataFrame with the new one
updated_data = pd.concat([existing_data, missing_data], ignore_index=True)

# Sort the DataFrame by date
updated_data.sort_values(by='Date', inplace=True)

# Save the updated DataFrame back to a CSV file
updated_data.to_csv('updated_data.csv', index=False)
