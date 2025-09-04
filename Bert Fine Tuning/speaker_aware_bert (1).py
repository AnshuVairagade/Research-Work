# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import torch.nn as nn

"""1. Tokenizer & Special Tokens"""

MODEL_NAME = "bert-base-uncased"
LABEL_LIST = ["neutral", "positive", "negative"]   # Example: sentiment labels
MAX_SPEAKERS = 10
CONTEXT_SIZE = 3

SPECIAL_SPK_TOKENS = [f"[SPK{i}]" for i in range(MAX_SPEAKERS)]
speaker2id = {f"spk{i}": i for i in range(MAX_SPEAKERS)}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
extra = [SPECIAL_SPK_TOKENS[i] for i in range(len(speaker2id))]
tokenizer.add_special_tokens({"additional_special_tokens": extra})
print("Added speaker tokens:", extra)

"""2. Dataset"""

class SpeakerContextDataset(Dataset):
    def __init__(self, conversations, tokenizer, speaker2id, max_len=128, context_size=3):
        self.samples = []
        self.tokenizer = tokenizer
        self.speaker2id = speaker2id
        self.max_len = max_len
        self.context_size = context_size

        for conv in conversations:
            for i in range(len(conv)):
                context = conv[max(0, i - context_size): i + 1]
                target_label = conv[i]["label"]

                input_text = " ".join([f"[SPK{self.speaker2id[u['speaker']]}] {u['text']}" for u in context])
                self.samples.append({
                    "text": input_text,
                    "speaker_ids": [self.speaker2id[u["speaker"]] for u in context],
                    "turn_ids": list(range(len(context))),
                    "label": LABEL_LIST.index(target_label),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample["text"], truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "speaker_ids": torch.tensor(sample["speaker_ids"], dtype=torch.long),
            "turn_ids": torch.tensor(sample["turn_ids"], dtype=torch.long),
            "labels": torch.tensor(sample["label"], dtype=torch.long)
        }

"""# 3. Collator"""

class PadCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids = torch.stack([x["input_ids"] for x in batch])
        attention_mask = torch.stack([x["attention_mask"] for x in batch])
        speaker_ids = torch.nn.utils.rnn.pad_sequence(
            [x["speaker_ids"] for x in batch], batch_first=True, padding_value=0
        )
        turn_ids = torch.nn.utils.rnn.pad_sequence(
            [x["turn_ids"] for x in batch], batch_first=True, padding_value=0
        )
        labels = torch.stack([x["labels"] for x in batch])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "speaker_ids": speaker_ids,
            "turn_ids": turn_ids,
            "labels": labels
        }

"""# 4. Model"""

class SpeakerAwareBERT(nn.Module):
    def __init__(self, model_name, num_labels, max_speakers, max_turns):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert.resize_token_embeddings(len(tokenizer))
        hidden_size = self.bert.config.hidden_size

        self.speaker_embeddings = nn.Embedding(max_speakers, hidden_size)
        self.turn_embeddings = nn.Embedding(max_turns, hidden_size)

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, speaker_ids, turn_ids, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token

        spk_emb = self.speaker_embeddings(speaker_ids[:, -1])  # last speaker
        turn_emb = self.turn_embeddings(turn_ids[:, -1])      # last turn

        pooled_output = pooled_output + spk_emb + turn_emb
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}

"""# 5. Example Conversations"""

conversations = [
    [
        {"speaker": "spk0", "text": "Hello, how are you?", "label": "neutral"},
        {"speaker": "spk1", "text": "I'm good, thanks!", "label": "positive"},
        {"speaker": "spk0", "text": "Glad to hear that.", "label": "positive"},
    ],
    [
        {"speaker": "spk2", "text": "This is frustrating.", "label": "negative"},
        {"speaker": "spk3", "text": "Calm down please.", "label": "neutral"},
    ],
]

"""# 6. Build Dataset & Dataloader"""

from sklearn.model_selection import train_test_split

train_convs, val_convs = train_test_split(conversations, test_size=0.3, random_state=42)

train_dataset = SpeakerContextDataset(train_convs, tokenizer, speaker2id)
val_dataset = SpeakerContextDataset(val_convs, tokenizer, speaker2id)

collator = PadCollator(tokenizer.pad_token_id)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collator)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collator)

"""# 7. Training"""

from sklearn.metrics import accuracy_score

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                speaker_ids=batch["speaker_ids"],
                turn_ids=batch["turn_ids"],
                labels=batch["labels"]
            )

            loss = outputs["loss"]
            logits = outputs["logits"]

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            speaker_ids=batch["speaker_ids"],
            turn_ids=batch["turn_ids"],
            labels=batch["labels"]
        )

        loss = outputs["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    train_eval_loss, train_eval_acc = evaluate(model, train_dataloader, device)
    val_loss, val_acc = evaluate(model, val_dataloader, device)

    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"  Train | Loss: {train_eval_loss:.4f}, Acc: {train_eval_acc:.4f}")
    print(f"  Val   | Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

"""# Model training curve

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

def smooth_curve(x, y, step=0.05):
    """Smooth curve using spline interpolation with fine step"""
    x_new = np.arange(min(x), max(x) + step, step)  # finer spacing on x-axis
    spline = make_interp_spline(x, y, k=3)  # cubic spline
    y_smooth = spline(x_new)
    return x_new, y_smooth

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, num_epochs):
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    # ---- Loss Curve ----
    plt.subplot(1, 2, 1)
    x_smooth, y_smooth = smooth_curve(list(epochs), train_losses, step=0.1)
    plt.plot(x_smooth, y_smooth, 'b-', label='Training Loss')
    x_smooth, y_smooth = smooth_curve(list(epochs), val_losses, step=0.1)
    plt.plot(x_smooth, y_smooth, 'r-', label='Validation Loss')
    plt.scatter(epochs, train_losses, c='b', marker='o')  # original points
    plt.scatter(epochs, val_losses, c='r', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)

    # ---- Accuracy Curve ----
    plt.subplot(1, 2, 2)
    x_smooth, y_smooth = smooth_curve(list(epochs), train_accs, step=0.1)
    plt.plot(x_smooth, y_smooth, 'b-', label='Training Accuracy')
    x_smooth, y_smooth = smooth_curve(list(epochs), val_accs, step=0.1)
    plt.plot(x_smooth, y_smooth, 'r-', label='Validation Accuracy')
    plt.scatter(epochs, train_accs, c='b', marker='o')
    plt.scatter(epochs, val_accs, c='r', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

def smooth_curve(x, y, step=0.05):
    """Smooth curve using cubic spline interpolation with 0.05 step"""
    x_new = np.arange(min(x), max(x) + step, step)  
    spline = make_interp_spline(x, y, k=3) 
    y_smooth = spline(x_new)
    return x_new, y_smooth

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, num_epochs):
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(14, 6)) 

    # ---- Loss Curve ----
    plt.subplot(1, 2, 1)
    x_smooth, y_smooth = smooth_curve(list(epochs), train_losses)
    plt.plot(x_smooth, y_smooth, 'b-', label='Training Loss')
    x_smooth, y_smooth = smooth_curve(list(epochs), val_losses)
    plt.plot(x_smooth, y_smooth, 'r-', label='Validation Loss')
    plt.scatter(epochs, train_losses, c='b', marker='o') 
    plt.scatter(epochs, val_losses, c='r', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    y_min = min(min(train_losses), min(val_losses))
    y_max = max(max(train_losses), max(val_losses))
    plt.yticks(np.arange(round(y_min, 2), round(y_max + 0.05, 2), 0.05))

    # ---- Accuracy Curve ----
    plt.subplot(1, 2, 2)
    x_smooth, y_smooth = smooth_curve(list(epochs), train_accs)
    plt.plot(x_smooth, y_smooth, 'b-', label='Training Accuracy')
    x_smooth, y_smooth = smooth_curve(list(epochs), val_accs)
    plt.plot(x_smooth, y_smooth, 'r-', label='Validation Accuracy')
    plt.scatter(epochs, train_accs, c='b', marker='o')
    plt.scatter(epochs, val_accs, c='r', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()

    # Set y-axis ticks with 0.05 interval
    y_min = min(min(train_accs), min(val_accs))
    y_max = max(max(train_accs), max(val_accs))
    plt.yticks(np.arange(round(y_min, 2), round(y_max + 0.05, 2), 0.05))

    plt.tight_layout()
    plt.show()

plot_training_curves(train_losses, val_losses, train_accs, val_accs, num_epochs=10)

"""# 8. Save Model"""

torch.save(model.state_dict(), "speaker_aware_bert.pt")
print("Model saved as speaker_aware_bert.pt")