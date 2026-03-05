import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm  # progress bar

# ── 1. Setup 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── 2. Load & tokenize data 
print("Loading and tokenizing dataset...")
# As in phase 2
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
dataset = load_dataset("imdb")

def tokenize_fn(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

tokenized = dataset.map(tokenize_fn, batched=True)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Use a small subset first to make sure everything runs
small_train = tokenized["train"]
small_test  = tokenized["test"]

train_loader = DataLoader(small_train, batch_size=16, shuffle=True)
test_loader  = DataLoader(small_test,  batch_size=16)

# ── 3. Load the model 
print("Loading RoBERTa model...")
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
model.to(device)  # move model to GPU

# ── 4. Optimizer & scheduler 
"""
### Terminologies

Epoch: One full pass through the training dataset
        - More epochs, model learn more/fits date better, but may be overfitting

Loss: how wrong is the model 

Knob: weight/parameters 

Learning Rate(lr): How big a step we take when turning knob

Step: Take the current weight and change them a little to reduce loss.
        - One update is called a step

Gradient: which direction should we turn each knob to reduce loss
        - It indicates how to adjust a parameter
        - Each parameter has a gradient

Warmup_step: In the set number of steps, the lr will increase from about 0 to lr
        - Then it usually decreasing in "linear" mode
        - Usually 5-10% of total training steps
        - Directly jumping to full lr may make training unstable especially for the 
          first few hundreds
        
"""
optimizer = AdamW(model.parameters(), lr=1e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_loader)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=num_training_steps
)

# ── 5. Training loop 
print("\nStarting training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        # Move batch to GPU
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        # By deafult the grad is cumulative
        optimizer.zero_grad()

        # PyTorch computes gradients, then each param.grad is filled
        loss.backward()

        # Optimizer updates the model weight, referring the gradient
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} — avg training loss: {avg_loss:.4f}")

# ── 6. Evaluation 
print("\nEvaluating...")
model.eval()
correct = 0
total = 0

with torch.no_grad():  # no gradient tracking needed during eval
    for batch in tqdm(test_loader, desc="Evaluating"):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"\nTest accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

# ── 7. Save the model 
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")
print("\nModel saved to ./saved_model")