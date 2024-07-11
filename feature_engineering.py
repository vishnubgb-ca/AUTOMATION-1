import pandas as pd

# Load data from csv file
df = pd.read_csv('data.csv')

# Check if 'lead_time' column exists in the dataframe
if 'lead_time' in df.columns:
    # Delete rows with NaN values in 'lead_time'
    df = df[df['lead_time'].notna()]

# Save the cleaned dataframe to a new csv file
df.to_csv('cleaned_data.csv', index=False)

# Print the top 5 rows of the cleaned dataframe
print(df.head(5))