import pandas as pd

# Read the Excel file and create a pandas DataFrame object
My_Data = pd.read_excel("My_Data.xlsx")

# Print the column names of the DataFrame
print(My_Data.columns)
