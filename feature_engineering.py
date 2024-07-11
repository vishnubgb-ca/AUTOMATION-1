import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('data.csv')

# Feature deletion
df = df.drop(['lead_time'], axis=1)

# Check if there are columns with categorical data
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# Encoding
if len(categorical_features) > 0:
    le = LabelEncoder()
    for feature in categorical_features:
        df[feature] = le.fit_transform(df[feature])

# Save the cleaned data
df.to_csv('cleaned_data.csv', index=False)

# Print the top 5 rows of the cleaned dataframe
print(df.head())