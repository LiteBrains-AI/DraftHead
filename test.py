import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from collections import Counter

from draft_model import DraftHeadModel


# --- Device --- 

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# --- Load AG News --- 

dataset = load_dataset("fancyzhx/ag_news")

train_data: Dataset = dataset["train"]
test_data: Dataset = dataset["test"]

# --- Hyperparameters --- 


NUM_TRAIN_SAMPLES = 4000   # None = use full training set
NUM_TEST_SAMPLES  = 2000    # None = use full test set

BATCH_SIZE = 40
MAX_LEN = 50

# Select number of samples


if NUM_TRAIN_SAMPLES is not None:
    NUM_TRAIN_SAMPLES = min(
        NUM_TRAIN_SAMPLES,
        len(train_data)
    )

    train_data = train_data.select(
        range(NUM_TRAIN_SAMPLES)
    )


if NUM_TEST_SAMPLES is not None:
    NUM_TEST_SAMPLES = min(
        NUM_TEST_SAMPLES,
        len(test_data)
    )

    test_data = test_data.select(
        range(NUM_TEST_SAMPLES)
    )

# --- Tokenizer --- 

def tokenize(text: str):
    return text.lower().split()

# --- Vocabulary --- 

counter = Counter()

for i in range(len(train_data)):
    counter.update(
        tokenize(train_data[i]["text"])
    )


vocab = {
    word: idx + 2
    for idx, (word, _) in enumerate(
        counter.most_common()
    )
}

vocab["<unk>"] = 0
vocab["<pad>"] = 1

# --- Encoding --- 

def encode(text: str, max_len: int = MAX_LEN):

    tokens = tokenize(text)

    ids = [
        vocab.get(tok, vocab["<unk>"])
        for tok in tokens[:max_len]
    ]

    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (
            max_len - len(ids)
        )

    return torch.tensor(
        ids,
        dtype=torch.long
    )

# --- Collate --- 

def collate_batch(batch):

    texts = torch.stack([
        encode(x["text"])
        for x in batch
    ])

    labels = torch.tensor(
        [x["label"] for x in batch],
        dtype=torch.long
    )

    return texts, labels

# --- DataLoaders --- 

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_batch
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_batch
)

# --- Info --- 

print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")


# --- Model ---

model = DraftHeadModel(
    vocab_size=len(vocab),
    hidden_size=128,
    num_layers=4,
    num_heads=4,
    num_drafts=16,
    max_length=MAX_LEN,
).to(device)


# --- Replace LM Head for Classification ---

model.lm_head = nn.Linear(
    128,
    4  # AG News has 4 classes
).to(device)


# --- Loss & Optimizer ---

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=5e-4
)


# --- Training Function ---

def train_epoch():

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for texts, labels in train_loader:

        texts = texts.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits, _ = model(texts)

        # Pool sequence outputs
        logits = logits[:, 0]

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = logits.argmax(dim=-1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    accuracy = 100 * correct / total

    return (
        total_loss / len(train_loader),
        accuracy
    )


# --- Evaluation Function ---

@torch.no_grad()
def evaluate():

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for texts, labels in test_loader:

        texts = texts.to(device)
        labels = labels.to(device)

        logits, _ = model(texts)

        logits = logits[:, 0]

        loss = criterion(
            logits,
            labels
        )

        total_loss += loss.item()

        predictions = logits.argmax(dim=-1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    accuracy = 100 * correct / total

    return (
        total_loss / len(test_loader),
        accuracy
    )


# --- Training Loop ---

EPOCHS = 20

for epoch in range(EPOCHS):

    train_loss, train_acc = train_epoch()

    test_loss, test_acc = evaluate()

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_acc:.2f}%"
    )

# Train Accuracy is ~ 98%. Model is clearly overfitting!
# Test Accuracy is ~ 66% only. (1 drafts)
# Test Accuracy is ~ 63% only. (4 drafts)
# Test Accuracy is ~ 73% (16 drafts)
# ✅ Reason found: Model is overfitting for fewer samples, So, generalization is weak!

# Open question: DraftHead assigns confidence scores to intermediate drafts. However, confidence is learned implicitly through task loss and may not correspond to actual correctness.
