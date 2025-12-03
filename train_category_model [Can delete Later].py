import os
import random
import pickle
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW

# ============== CONFIG ==============
CSV_PATH = "train_category.csv"      # your CSV
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
OUTPUT_DIR = "./category_model_simple"

MAX_LENGTH = 128
BATCH_SIZE = 4
EPOCHS = 2
LR = 2e-5
SEED = 42
# ====================================

# Fix seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 1. Load CSV (with encoding safety)
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Could not find {CSV_PATH} in current directory")

# Try latin1 first because of Windows, if error then try utf-8
try:
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv(CSV_PATH, encoding="utf-8")

# Strip and normalize column names
clean_cols = {c.strip().lower(): c for c in df.columns}
if "text" not in clean_cols or "category" not in clean_cols:
    raise ValueError(
        f"CSV must have 'text' and 'category' columns (any case, trimmed). Found: {df.columns}"
    )

text_col = clean_cols["text"]
cat_col = clean_cols["category"]

df = df[[text_col, cat_col]].rename(columns={text_col: "text", cat_col: "category"})
df = df.dropna(subset=["text", "category"])
df["text"] = df["text"].astype(str)
df["category"] = df["category"].astype(str).str.strip()

print("\nSample of loaded data:")
print(df.head())

# 2. Encode labels
label_encoder = LabelEncoder()
df["label_id"] = label_encoder.fit_transform(df["category"])

print("\nLabel mapping:")
for idx, cat in enumerate(label_encoder.classes_):
    print(f"{idx} -> {cat}")

# 3. Train / Validation split (80 / 20)
train_df, val_df = train_test_split(
    df[["text", "label_id"]],
    test_size=0.2,
    random_state=SEED,
    stratify=df["label_id"],
)

print(f"\nTrain samples: {len(train_df)}  |  Validation samples: {len(val_df)}")

# 4. Dataset class
class CategoryDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        return item

# 5. Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_encoder.classes_),
)
model.to(device)

# 6. DataLoaders
train_dataset = CategoryDataset(
    train_df["text"], train_df["label_id"], tokenizer, MAX_LENGTH
)
val_dataset = CategoryDataset(
    val_df["text"], val_df["label_id"], tokenizer, MAX_LENGTH
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 7. Optimizer
optimizer = AdamW(model.parameters(), lr=LR)

# 8. Training loop
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def eval_epoch(model, dataloader, device):
    model.eval()
    preds = []
    labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
            batch_labels = batch["labels"].cpu().numpy()

            preds.extend(batch_preds)
            labels.extend(batch_labels)

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(labels, preds)
    report = classification_report(
        labels,
        preds,
        target_names=label_encoder.classes_,
        digits=3,
    )
    return avg_loss, acc, report

print("\n===== Training LegalBERT for Category Classification (simple loop) =====")
for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_loss, val_acc, val_report = eval_epoch(model, val_loader, device)

    print(f"\nEpoch {epoch}/{EPOCHS}")
    print(f"Train loss: {train_loss:.4f}")
    print(f"Val   loss: {val_loss:.4f}")
    print(f"Val   acc : {val_acc:.4f}")
    print("Validation classification report:")
    print(val_report)

# 9. Save model & label encoder
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)

print(f"\n✅ Model and label encoder saved to: {OUTPUT_DIR}")

# 10. Quick prediction function
def predict_category(text: str) -> str:
    model.eval()
    with torch.no_grad():
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(**enc)
        pred_id = torch.argmax(outputs.logits, dim=-1).item()
        label = label_encoder.inverse_transform([pred_id])[0]
        return label

print("\n===== Quick Test Predictions =====")
test_queries = [
    "My landlord is not returning my deposit after I vacated the flat",
    "My employer is forcing me to work overtime without extra pay",
    "Police refused to register an FIR for theft of my bike",
    "I want to file for divorce from my husband",
    "My online order was not delivered and refund is not processed",
    "Someone hacked my Instagram and is posting from my account",
]

for q in test_queries:
    print(f"Q: {q}")
    print(f" → Predicted category: {predict_category(q)}\n")
