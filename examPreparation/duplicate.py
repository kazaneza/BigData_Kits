import pandas as pd
import numpy as np

# Step 1: Read the CSV file
df = pd.read_csv('malaria_clinical_data.csv')

# Step 2: Replace empty cells with median along axis 0
numeric_columns = df.select_dtypes(include=np.number).columns
median_values = df[numeric_columns].median(axis=0)
df[numeric_columns] = df[numeric_columns].fillna(median_values)

# Step 3: Highlight duplicates in the DataFrame
df['is_duplicate'] = df.duplicated()

# Step 4: Save the duplicate rows to a new CSV file
duplicate_df = df[df['is_duplicate']]
duplicate_df.to_csv('DuplicateData.csv', index=False)

# Print the duplicate rows
print(duplicate_df)
