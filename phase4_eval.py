import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
from tqdm import tqdm

# ── 1. Load saved model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading saved model...")
model = AutoModelForSequenceClassification.from_pretrained("./saved_model")
tokenizer = AutoTokenizer.from_pretrained("./saved_model")
model.to(device)
model.eval()

# ── 2. Load & tokenize test set 
print("Loading dataset...")
dataset = load_dataset("imdb")

def tokenize_fn(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

tokenized = dataset.map(tokenize_fn, batched=True)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_loader = DataLoader(tokenized["test"], batch_size=16)

# ── 3. Collect all predictions
print("Running predictions...")
all_preds  = []
all_labels = []
all_probs  = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=-1)
        preds   = torch.argmax(outputs.logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

# ── 4. Classification report
print("\n--- Classification Report ---")
print(classification_report(all_labels, all_preds, target_names=["Negative", "Positive"]))

# ── 5. Confusion matrix 
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
plt.title("Confusion Matrix — RoBERTa IMDb Sentiment")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Confusion matrix saved to confusion_matrix.png")

# ── 6. Confidence distribution 
correct_mask = all_preds == all_labels
correct_conf = all_probs[correct_mask].max(axis=1)
wrong_conf   = all_probs[~correct_mask].max(axis=1)

plt.figure(figsize=(8, 4))
plt.hist(correct_conf, bins=30, alpha=0.6, label="Correct predictions", color="green")
plt.hist(wrong_conf,   bins=30, alpha=0.6, label="Wrong predictions",   color="red")
plt.xlabel("Model confidence")
plt.ylabel("Count")
plt.title("Confidence Distribution — Correct vs Wrong Predictions")
plt.legend()
plt.tight_layout()
plt.savefig("confidence_dist.png", dpi=150)
print("Confidence chart saved to confidence_dist.png")

# ── 7. Error analysis — find interesting mistakes 
raw_test   = dataset["test"]
wrong_idxs = np.where(all_preds != all_labels)[0]

print(f"\n--- Error Analysis (showing 5 mistakes out of {len(wrong_idxs)} total) ---")
label_map = {0: "Negative", 1: "Positive"}

# Sort by confidence — show the most confidently wrong predictions
wrong_confs     = all_probs[wrong_idxs].max(axis=1)
most_conf_wrong = wrong_idxs[np.argsort(wrong_confs)[::-1][:5]]

for idx in most_conf_wrong:
    text       = raw_test[int(idx)]["text"][:300]
    true_label = label_map[all_labels[idx]]
    pred_label = label_map[all_preds[idx]]
    confidence = all_probs[idx].max() * 100
    print(f"\n[True: {true_label} | Predicted: {pred_label} | Confidence: {confidence:.1f}%]")
    print(text)
    print("...")