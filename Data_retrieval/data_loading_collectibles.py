import csv
import pandas as pd

### Watches ###

# Rolex Daytona 116506 was first introduced in 2013, I've got daily prices until 2018

# Open the file
with open('data/Watches/Daytona 116506 Watch Market Prices - 2018-2023.csv') as csv_file:
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


