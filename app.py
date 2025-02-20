from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Load the pre-trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.load_state_dict(torch.load("imdb_sentiment_model_2000.pth"))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function to predict sentiment
def predict_sentiment(text):
    encoding = tokenizer(text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    return "Positive" if preds[0] == 1 else "Negative"

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    review_text = None

    if request.method == "POST":
        review_text = request.form["review"]
        prediction = predict_sentiment(review_text)

    return render_template("index.html", prediction=prediction, review=review_text)

if __name__ == "__main__":
    app.run(debug=True)
