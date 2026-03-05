from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# 1. Load the IMDb dataset
print("Loading IMDb dataset...")
dataset = load_dataset("imdb")
print(dataset)


# ── 2. Peek at the raw data 
print("\n--- Sample review (label 0 = negative, 1 = positive) ---")
print("Text:  ", dataset["train"][0]["text"][:300])
print("Label: ", dataset["train"][0]["label"])


# ── 3. Load the RoBERTa tokenizer 
# It has its own rules to split the inputs, also trained by sth
tokenizer = AutoTokenizer.from_pretrained("roberta-base")


# ── 4. Tokenize a single example manually 

# The first line in the trainning data, then the first 300 chars
# in the text(the comment)
sample_text = dataset["train"][0]["text"][:300]

# Tokenizer automatically creates "attention_mask" to deal with 
# scenarios that some inputs is shorter than 128 chars
tokens = tokenizer(
    sample_text,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"  # return PyTorch tensors, not Py lists
)

print("\n--- Tokenized output ---")
print("input_ids shape:      ", tokens["input_ids"].shape)
print("attention_mask shape: ", tokens["attention_mask"].shape)
print("First 20 token IDs:   ", tokens["input_ids"][0][:20])
print("Decoded back to text: ", tokenizer.decode(tokens["input_ids"][0][:20]))


# ── 5. Tokenize the full dataset efficiently 
def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

print("\nTokenizing full dataset (this may take a minute)...")
# Passing an function object to .map(), not ran by myself
# This is purely the use of .map() method which I don't get clear with.
tokenized_dataset = dataset.map(tokenize_fn, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
print("Done")
print(tokenized_dataset)


# ── 6. Create DataLoaders
# To serve the patches to the model
train_loader = DataLoader(tokenized_dataset["train"], batch_size=16, shuffle=True)
test_loader  = DataLoader(tokenized_dataset["test"],  batch_size=16)

# Peek at one batch
batch = next(iter(train_loader))
print("\n--- One batch from DataLoader ---")
print("input_ids shape:      ", batch["input_ids"].shape)
print("attention_mask shape: ", batch["attention_mask"].shape)
print("labels shape:         ", batch["label"].shape)
print("Sample labels:        ", batch["label"])