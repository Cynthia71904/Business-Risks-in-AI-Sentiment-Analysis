import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from keybert import KeyBERT

MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 2e-5
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_ROWS = None

df = pd.read_csv("Reviews.csv").head(1000)
if MAX_ROWS:
    df = df.head(MAX_ROWS)
def score_to_label(score):
    if score >= 4:
        return 2#pos
    elif score == 3:
        return 1#neu
    else:
        return 0#neg
df["label"] = df["Score"].apply(score_to_label)
df["text"] = df["Summary"].astype(str) + " " + df["Text"].astype(str)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
class ReviewDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenizer(
            text, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }
train_dataset = ReviewDataset(train_texts, train_labels)
val_dataset = ReviewDataset(val_texts, val_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
model.to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * EPOCHS
)
train_losses, val_losses, train_accs, val_accs = [], [], [], []
for epoch in range(EPOCHS):
    model.train()
    total_loss, total_correct = 0, 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        optimizer.zero_grad()
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        total_correct += (preds == inputs["labels"]).sum().item()

    train_losses.append(total_loss / len(train_loader))
    train_accs.append(total_correct / len(train_dataset))

    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            val_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            val_correct += (preds == inputs["labels"]).sum().item()

    val_losses.append(val_loss / len(val_loader))
    val_accs.append(val_correct / len(val_dataset))

    print(f"Epoch {epoch+1}: Train Loss {train_losses[-1]:.4f}, Val Loss {val_losses[-1]:.4f}")
    print(f"Train Acc {train_accs[-1]:.4f}, Val Acc {val_accs[-1]:.4f}")

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.title("Loss")
plt.subplot(1,2,2)
plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Validation Accuracy")
plt.legend()
plt.title("Accuracy")
plt.tight_layout()
plt.savefig("training_AL.png")
plt.close()

model.eval()
tokenized = tokenizer(df["text"].tolist(), truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
input_ids, attention_mask = tokenized["input_ids"].to(DEVICE), tokenized["attention_mask"].to(DEVICE)
with torch.no_grad():
  outputs = model(input_ids, attention_mask=attention_mask)
  preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
df["Predicted_Sentiment"] = preds
df["Predicted_Sentiment"] = df["Predicted_Sentiment"].map({0: "Negative", 1: "Neutral", 2: "Positive"})

sentiment_counts = df["Predicted_Sentiment"].value_counts()
plt.figure(figsize=(6,6))
plt.pie(
    sentiment_counts.values,
    labels=sentiment_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    explode=[0.05]*len(sentiment_counts),
    shadow=True
)
plt.title("Sentiment Distribution")
plt.savefig("sentiment_pie_chart.png")
plt.close()

kw_model = KeyBERT(model="distilbert-base-nli-mean-tokens")
keywords_list = []
for text in tqdm(df["text"].tolist(), desc="Keyword Extraction"):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1,2), stop_words="english", top_n=3)
    keywords_list.append(", ".join([kw for kw, _ in keywords]))
df["Keywords"] = keywords_list
df.to_csv("Reviews_Sentiment_Keywords.csv", index=False)
print("已保存")
