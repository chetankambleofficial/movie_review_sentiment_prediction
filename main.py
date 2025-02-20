import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import pandas as pd

# Step 1: Load the CSV Dataset (only 2000 rows)
csv_filename = "data.csv"
df = pd.read_csv(csv_filename)

# Take only the first 2000 rows
df = df.head(100)  # Update to 2000 rows as per the comment

# Display the first few rows
print("Dataset Preview:")
print(df.head())

# Step 2: Split into Training and Validation Sets
texts = df["text"].tolist()
labels = df["label"].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Step 3: Tokenize the Dataset
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }

# Create datasets and dataloaders
train_dataset = CustomDataset(train_texts, train_labels, tokenizer)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Step 4: Load Pre-trained Model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Step 5: Set Up Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Step 6: Training Loop
def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    correct_predictions = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Compute loss
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions.double() / len(dataloader.dataset)
    return avg_loss, accuracy

# Step 7: Validation Loop
def eval_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Compute loss
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            # Compute accuracy
            preds = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions.double() / len(dataloader.dataset)
    return avg_loss, accuracy

# Step 8: Train the Model
epochs = 3
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
    val_loss, val_acc = eval_epoch(model, val_loader, loss_fn, device)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
    print("-" * 50)

# Step 9: Save the Model and Tokenizer
model.save_pretrained("imdb_sentiment_model_2000")
tokenizer.save_pretrained("imdb_sentiment_model_2000")

# Step 10: Make Predictions
def predict_sentiment(text, model, tokenizer, device):
    model.eval()
    encoding = tokenizer(
        text, return_tensors="pt", max_length=128, padding="max_length", truncation=True
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    return "Positive" if preds[0] == 1 else "Negative"

# Test the model with a sample review
sample_review = "This movie was boring! The acting was worst and the storyline was boring."
prediction = predict_sentiment(sample_review, model, tokenizer, device)
print(f"Review: {sample_review}")
print(f"Predicted Sentiment: {prediction}")
