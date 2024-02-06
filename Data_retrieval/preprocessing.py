import csv
import pandas as pd

# Open the file
with open('Data_retrieval\data\Watches\Rolex Daytona 116520 Price History.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    prices_data = []
    date_data = []

    row_count = 0
    for row in csv_reader:
        if row_count == 0:
            print(f'Column names are {", ".join(row)}')
            row_count += 1
        else:
            date_data.append(row[0])
            prices_data.append(row[1])

            row_count += 1
    print(f'Processed {row_count} rows.')

# Create a dataframe
df = pd.DataFrame({'date': date_data, 'price (USD)': prices_data})
print(df)

### Cars ###

### Art ###


