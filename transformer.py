import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np

 ############################################################
 ############### This code belongs to ex2.py ################
 ############################################################

ex2_category_dict = {'comp.graphics': 'computer graphics',
                 'rec.sport.baseball': 'baseball',
                 'sci.electronics': 'science, electronics',
                 'talk.politics.guns': 'politics, guns'
                     }


def ex2_get_data(categories=None, portion=1.):
    """
    Get data for given categories and portion.
    :param portion: portion of the data to use.
    """
    # get data
    from sklearn.datasets import fetch_20newsgroups
    data_train = fetch_20newsgroups(categories=categories, subset='train', remove=('headers', 'footers', 'quotes'),
                                    random_state=21)
    data_test = fetch_20newsgroups(categories=categories, subset='test', remove=('headers', 'footers', 'quotes'),
                                   random_state=21)

    # train
    train_len = int(portion * len(data_train.data))
    x_train = np.array(data_train.data[:train_len])
    y_train = data_train.target[:train_len]
    # remove empty entries
    non_empty = x_train != ""
    x_train, y_train = x_train[non_empty].tolist(), y_train[non_empty].tolist()

    # test
    x_test = np.array(data_test.data)
    y_test = data_test.target
    non_empty = np.array(x_test) != ""
    x_test, y_test = x_test[non_empty].tolist(), y_test[non_empty].tolist()
    return x_train, y_train, x_test, y_test


############################################################
############################################################
############################################################


class TransformerDataset(torch.utils.data.Dataset):
    """
    Dataset class for handling text and labels
    """
    def __init__(self, texts, labels, tokenizer, max_length):
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
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

def train_epoch(model, data_loader, optimizer, device):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0
    total_preds, total_labels = [], []

    for batch in tqdm(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        total_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(total_labels, total_preds)
    return total_loss / len(data_loader), acc

def evaluate(model, data_loader, device):
    """
    Evaluate the model.
    """
    model.eval()
    total_loss = 0
    total_preds, total_labels = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            total_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            total_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(total_labels, total_preds)
    return total_loss / len(data_loader), acc

def transformer_classification(portion=0.1):
    x_train, y_train, x_test, y_test = ex2_get_data(categories=ex2_category_dict.keys(), portion=portion)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=len(ex2_category_dict))

    # Prepare datasets and data loaders
    max_length = 128
    train_dataset = TransformerDataset(x_train, y_train, tokenizer, max_length)
    test_dataset = TransformerDataset(x_test, y_test, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Set device and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0)

    # Training loop
    n_epochs = 2
    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {test_loss:.4f}, Validation Accuracy: {test_acc:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    portions = [0.1, 0.2]
    for portion in portions:
        print(f"\nTraining with portion {portion}")
        transformer_classification(portion=portion)
