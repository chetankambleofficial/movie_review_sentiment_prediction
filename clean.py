import pandas as pd
import re

# Load the CSV file
df = pd.read_csv('imdb_dataset.csv')  # Replace with your actual file name

# Function to clean text
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = re.sub(r'<.*?>', '', text)        # Remove HTML tags
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)         # Remove extra spaces
    return text.strip()

# Apply the cleaning function
df['text'] = df['text'].apply(clean_text)

# Filter for minimum text length (e.g., at least 50 characters)
df = df[df['text'].str.len() >= 50]

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Select 1,000 records
df_small = df.head(1000)

# Export to a **human-readable** CSV file (without compression)
df_small.to_csv('cleaned_data_1000.csv', index=False, encoding='utf-8')

print("✅ Data cleaned, shuffled, and saved as 'cleaned_data_1000.csv' (Human-readable)")
