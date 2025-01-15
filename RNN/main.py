import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score,classification_report, accuracy_score
import pandas as pd
from text_processing import TextProcessor, TextDataset, load_data
from model import RNNModel

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0, 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
        train_acc = correct / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss, val_correct = 0, 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_acc = val_correct / len(val_loader.dataset)

        # Calculate precision, recall, and F1-score
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

def evaluate_model(model, val_loader, label_encoder, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate classification report
    report = classification_report(
        all_labels, all_preds, target_names=label_encoder.classes_, output_dict=True
    )
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.round(2)

    # Display the classification report as a table
    print("\nClassification Report:")
    print(df_report)

    # Test accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {accuracy:.2%}")

def main():
    # Load and preprocess data
    file_path = 'philosophy_data_edit.csv'
    X_train, X_val, y_train, y_val, label_encoder = load_data(file_path)
    processor = TextProcessor()
    X_train_processed = processor.fit_transform(X_train)
    X_val_processed = processor.transform(X_val)
    
    train_dataset = TextDataset(X_train_processed, y_train)
    val_dataset = TextDataset(X_val_processed, y_val)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    
    # Define model and training parameters
    vocab_size = 10000
    embed_dim = 128
    hidden_dim = 256
    output_dim = len(label_encoder.classes_)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = RNNModel(vocab_size, embed_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10, device=device)

    # Evaluate the model
    evaluate_model(model, val_loader, label_encoder, device)

if __name__ == "__main__":
    main()
