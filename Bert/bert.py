import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig,BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_data(file_path, test_size=0.2, random_state=42):
    data = pd.read_csv(file_path)
    X = data['sentence_str'].tolist()
    y = LabelEncoder().fit_transform(data['school'].tolist())

    # Split into train+validation and test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # Split train+validation into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=test_size, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

def initialize_model(train_loader, epochs=4):
    config = BertConfig.from_pretrained(
        'bert-base-uncased',
        num_labels=4,  # Number of classes in your dataset
        hidden_dropout_prob=0.44,  # Dropout probability for hidden layers
        attention_probs_dropout_prob=0.44  # Dropout probability for attention layers
    )

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=3e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    return model, tokenizer, optimizer, scheduler, device


def train(model, train_loader, val_loader, optimizer, scheduler, device, epochs=4):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        print(f"Epoch {epoch+1}/{epochs}:")

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct_train += (predictions == batch['labels']).sum().item()
            total_train += batch['labels'].size(0)

        train_loss = total_train_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        eval_loss, eval_acc = evaluate(model, val_loader, device)
        val_losses.append(eval_loss)
        val_accuracies.append(eval_acc)
        print(f"Validation Loss: {eval_loss:.4f}, Validation Accuracy: {eval_acc:.4f}")
    return train_losses, train_accuracies, val_losses, val_accuracies

def evaluate(model, loader, device):
    all_predictions = []
    all_labels = []

    model.eval()
    total_eval_loss = 0
    correct_eval = 0
    total_eval = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            total_eval_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct_eval += (predictions == batch['labels']).sum().item()
            total_eval += batch['labels'].size(0)

    eval_loss = total_eval_loss / len(loader)
    eval_acc = correct_eval / total_eval
    return eval_loss, eval_acc

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # Plot Loss
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Train vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Train vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def final_evaluation(model, test_loader, device):
    print("\nFinal Model Evaluation on Test Data:")
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

    # Calculate test accuracy
    test_accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Generate and print classification report
    report = classification_report(all_labels, all_predictions, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
    print("\nClassification Report:")
    print(report)

def main():
    file_path = 'philosophy_data_edit.csv'
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(file_path)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = TextDataset(X_train, y_train, tokenizer, max_length=128)
    val_dataset = TextDataset(X_val, y_val, tokenizer, max_length=128)
    test_dataset = TextDataset(X_test, y_test, tokenizer, max_length=128)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=3)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=3)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=3)

    model, tokenizer, optimizer, scheduler, device = initialize_model(train_loader)

    train_losses, train_accuracies, val_losses, val_accuracies = train(model,
                                                                        train_loader, val_loader, optimizer, scheduler, device)
    final_evaluation(model, test_loader, device)
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

if __name__ == "__main__":
    main()
