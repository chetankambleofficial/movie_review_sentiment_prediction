from datasets import load_dataset
import pandas as pd

# Step 1: Load the IMDB dataset
print("Loading IMDB dataset...")
dataset = load_dataset("imdb")

# Step 2: Convert to pandas DataFrame
train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

# Step 3: Combine train and test datasets
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Step 4: Save to CSV
csv_filename = "imdb_dataset.csv"
combined_df.to_csv(csv_filename, index=False)

print(f"IMDB dataset saved to {csv_filename}.")
print(f"Total rows: {len(combined_df)}")