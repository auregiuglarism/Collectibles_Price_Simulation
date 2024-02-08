import csv
import matplotlib.pyplot as plt

# TODO : MAKE SURE TO ITERATE EXACTLY AT THE END OF THE MONTH AND NOT JUST +30 DAYS

# NB make sure the csv files* all have the same length and dates. This is important for the index to be accurate.
# *: All csv files except Watch_Index.csv 

# Index Weights retrieved from watchcharts.com
w1_116500 = 0.28
w2_126334 = 0.22
w3_116508 = 0.17
w4_57111A = 0.17
w5_116520 = 0.16 

# Open all the files
def open_files():
    # Rolex Daytona 116520 Price History
    with open('Data_retrieval\data\Watches\Rolex Daytona 116520 Price History.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        prices_data_rolex_116520 = []
        date_data_rolex_116520 = []

        row_count = 0
        for row in csv_reader:
            if row_count == 0:
                # print(f'Column names for rolex daytona 116520 are {", ".join(row)}')
                row_count += 1
            else:
                date_data_rolex_116520.append(row[0])
                prices_data_rolex_116520.append(row[1])

                row_count += 1
        # print(f'Processed {row_count} rows for rolex daytona 116520.')

    # Rolex Daytona 116508 Price History
    with open('Data_retrieval\data\Watches\Rolex Daytona 116508 Price History.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        prices_data_rolex_116508 = []
        date_data_rolex_116508 = []

        row_count = 0
        for row in csv_reader:
            if row_count == 0:
                # print(f'Column names for rolex daytona 116508 are {", ".join(row)}')
                row_count += 1
            else:
                date_data_rolex_116508.append(row[0])
                prices_data_rolex_116508.append(row[1])

                row_count += 1
        # print(f'Processed {row_count} rows for rolex daytona 116508.')

    # Rolex Daytona 116500 Price History
    with open('Data_retrieval\data\Watches\Rolex Daytona 116500 Price History.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        prices_data_rolex_116500 = []
        date_data_rolex_116500 = []

        row_count = 0
        for row in csv_reader:
            if row_count == 0:
                # print(f'Column names for rolex daytona 116500 are {", ".join(row)}')
                row_count += 1
            else:
                date_data_rolex_116500.append(row[0])
                prices_data_rolex_116500.append(row[1])

                row_count += 1
        # print(f'Processed {row_count} rows for rolex daytona 116500.')

    # Rolex Datejust 126334 Price History
    with open('Data_retrieval\data\Watches\Rolex DateJust 126334 Price History.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        prices_data_rolex_126334 = []
        date_data_rolex_126334 = []

        row_count = 0
        for row in csv_reader:
            if row_count == 0:
                # print(f'Column names for rolex datejust 126334 are {", ".join(row)}')
                row_count += 1
            else:
                date_data_rolex_126334.append(row[0])
                prices_data_rolex_126334.append(row[1])

                row_count += 1
        # print(f'Processed {row_count} rows for rolex datejust 126334.')

    # Patek Philippe 5711/1A Price History
    with open('Data_retrieval\data\Watches\Rolex DateJust 126334 Price History.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        prices_data_patek_57111A = []
        date_data_patek_57111A = []

        row_count = 0
        for row in csv_reader:
            if row_count == 0:
                # print(f'Column names for patek philippe 57111A are {", ".join(row)}')
                row_count += 1
            else:
                date_data_patek_57111A.append(row[0])
                prices_data_patek_57111A.append(row[1])

                row_count += 1
        # print(f'Processed {row_count} rows for patek philippe 57111A.')
                
    if len(prices_data_patek_57111A) == len(prices_data_rolex_116500) == len(prices_data_rolex_116508) == len(prices_data_rolex_116500) == len(prices_data_rolex_126334):
        return prices_data_rolex_116520, prices_data_rolex_116508, prices_data_rolex_116500, prices_data_rolex_126334, prices_data_patek_57111A, date_data_rolex_116520
    else:
        return "Error: Length of the csv files are not the same"
    

## Make index ##
if __name__ == "__main__":
    prices_data_rolex_116520, prices_data_rolex_116508, prices_data_rolex_116500, prices_data_rolex_126334, prices_data_patek_57111A, dates = open_files()
    
    # Create the index
    index_prices = []
    for val in range(0, len(prices_data_rolex_116500)): # All have the same lengths and same dates
        index_value = (w1_116500 * float(prices_data_rolex_116500[val])) + (w2_126334 * float(prices_data_rolex_126334[val])) + (w3_116508 * float(prices_data_rolex_116508[val])) + (w4_57111A * float(prices_data_patek_57111A[val])) + (w5_116520 * float(prices_data_rolex_116520[val]))
        index_prices.append(index_value)
    # print(index_prices)

    # Plot the index daily (Seems to be correct when compared to the graph on the watchcharts website)
    # plt.plot(dates, index_prices)
    # plt.xlabel('Date')
    # plt.ylabel('Index Value')
    # plt.title('Custom Weighted Watch Index')
    # plt.show()

    # Now average the index to reflect monthly changes.
    # TODO HERE
    monthly_index = []
    monthly_dates = []
    for value in range(0, len(index_prices), 30):
        monthly_dates.append(dates[value]) # Take the first date of the month for this month without the day name
        monthly_index.append(sum(index_prices[value:value+30])/30) 
    # print(monthly_index)

    # Plot the monthly index (Seems to be correct when compared visually to the daily graph)
    # plt.plot(monthly_dates, monthly_index)
    # plt.xlabel('Date')
    # plt.ylabel('Index Value (Monthly Average)')
    # plt.title('Custom Weighted Watch Index Monthly Average')
    # plt.show()

    # Save the monthly index to a csv file
    print("Saving the monthly index to a csv file")
    with open('Data_retrieval\data\Watches\Watch_Index.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Date', 'Index Value'])
        for i in range(0, len(monthly_dates)):
            writer.writerow([monthly_dates[i], monthly_index[i]])


    
    


