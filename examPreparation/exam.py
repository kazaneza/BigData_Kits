import pandas as pd
import numpy as np

# Step 1: Read the CSV file
df = pd.read_csv('malaria_clinical_data.csv')

# Step 2: Replace empty cells with median along axis 0
numeric_columns = df.select_dtypes(include=np.number).columns
median_values = df[numeric_columns].median(axis=0)
df[numeric_columns] = df[numeric_columns].fillna(median_values)

# Highlight 
df['is_duplicate'] = df.duplicated()

# Delete the duplicate rows
df = df.drop_duplicates()

# Step 5: Save 
df.to_csv('CleanData.csv', index=False)
