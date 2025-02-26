import pandas as pd

# Your corrected input data
data = {
    'Contract': ['IBRN1', 'IBRN1', 'IBRN1', 'IBRN1', 'IBRN1', 'IBRN1', 'IBRN1', 'IBRN1', 'IBRN1', 'IBRN1', 'IBRN1', 'IBRN1',
                 'IECF1', 'IECF1', 'IECF1', 'IECF1', 'IECF1', 'IECF1', 'IECF1', 'IECF1', 'IECF1',
                 'IHNG1', 'IHNG1', 'IHNG1', 'IHNG1', 'IHNG1', 'IHNG1', 'IHNG1'],
    'Date': ['02-Jan-25', '02-Jan-25', '03-Jan-25', '06-Jan-25', '07-Jan-25', '07-Jan-25', '08-Jan-25', '09-Jan-25', '10-Jan-25',
             '13-Jan-25', '14-Jan-25', '15-Jan-25', '08-Jan-25', '09-Jan-25', '10-Jan-25', '13-Jan-25', '14-Jan-25',
             '14-Jan-25', '15-Jan-25', '16-Jan-25', '17-Jan-25', '03-Jan-25', '06-Jan-25', '07-Jan-25', '07-Jan-25',
             '08-Jan-25', '09-Jan-25', '10-Jan-25'],
    'Value': [98, 100, 101, 104, 102, 103, 97, 99, 101, 100, 101, 97, 83, 81, 77, 79, 78, 77, 81, 80, 83, 120, 121, 124,
              126, 126, 125, 127]
}

# Check if the length of all columns is the same
print(len(data['Contract']))
print(len(data['Date']))
print(len(data['Value']))

# Create the DataFrame
df = pd.DataFrame(data)

# Convert 'Date' to datetime for proper sorting and comparison
df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')

# Create an empty list to store the cleaned data
cleaned_data = []

# Loop through each unique contract
for contract in df['Contract'].unique():
    # Filter the data for the current contract
    contract_data = df[df['Contract'] == contract]
    # print(contract_data)

    # Variable to store the previous value for the 'Prior' column
    prev_value = None

    # Loop through each unique date for this contract
    for date in contract_data['Date'].unique():
        # Get the data for the current date and contract
        date_data = contract_data[contract_data['Date'] == date]
        # print(date_data)

        # Find the maximum value (least negative) for this contract and date
        min_value = date_data['Value'].max()
        # print(min_value)

        # Add the 'Prior' column to the contract_data (NaN for the first row)
        prior_value = prev_value if prev_value is not None else min_value

        date_data = date_data.copy()
        date_data.loc[:, 'r2'] = (date_data['Value'] - prior_value)**2
        # print(date_data)

        # Find the minimum r2 for this contract and date
        min_r2 = date_data['r2'].min()
        # print(min_r2)

        # Get the row where 'r2' is equal to 'min_r2'
        row = date_data[date_data['r2'] == min_r2].iloc[0]

        # Access the 'Value' column from the row
        final_value = row['Value']
        # print(final_value)

        # Append the result to the cleaned_data list
        cleaned_data.append(
            [contract, date, final_value, prior_value])
        # print(cleaned_data)

        prev_value = min_value

# Convert cleaned data to a DataFrame
cleaned_df = pd.DataFrame(cleaned_data, columns=[
                          'Contract', 'Date', 'Value', 'Prior'])

# Sort the data by 'Contract' and 'Date'
cleaned_df = cleaned_df.sort_values(by=['Contract', 'Date'])

cleaned_df = cleaned_df.drop(columns=['Prior'])

# Display the cleaned DataFrame
print(cleaned_df)
