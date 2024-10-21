import os
import pandas as pd

df = pd.read_pickle('data.pkl')

# Create the training data DataFrame
train_data = pd.DataFrame()

# Fill the Question column with a generic extraction request
train_data['Question'] = ["Extract Key Issues, Relevant Statutes, Legal Precedents, Legal Doctrines, Jurisdiction, and Date from the following case:"] * len(df)

# The Context column contains the full case text
train_data['Context'] = df['Summary']

# Create the JSON structure for the Answer column
train_data['Answer'] = df.apply(lambda row: {
    "Key Issues": row['Key Issues'],
    "Relevant Statutes": row['Relevant Statutes and Provisions'],
    "Legal Precedents": row['Precedents Cited'],
    "Legal Doctrines": row['Legal Doctrines'],
    "Jurisdiction": row['Jurisdiction'],
    "Date": row['Date']
}, axis=1)

# Convert the dictionary to a JSON string format for each row
train_data['Answer'] = train_data['Answer'].apply(lambda x: str(x))


# Save the training data to a JSON file
train_data.to_json('train_data_with_summary.json', orient='records', lines=True)
train_data.to_pickle('train_data_with_summary.pkl')