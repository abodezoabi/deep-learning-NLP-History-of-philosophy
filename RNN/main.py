import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,classification_report
import pandas as pd
from text_processing import TextProcessor, TextDataset, load_data
from model import RNNModelWithAttention
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs):
    model.to(device)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss, correct = 0, 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
        train_acc = correct / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
        val_acc = val_correct / len(val_loader.dataset)

        scheduler.step()

        # Check for improvement
      #  if val_loss < best_val_loss:
      #      best_val_loss = val_loss
       #     torch.save(model.state_dict(), checkpoint_path)
       #     print(f"Epoch {epoch+1}: Validation loss improved to {val_loss:.4f}. Saved model.")
       # else:
        #    epochs_no_improve += 1
        #    print(f"Epoch {epoch+1}: Validation loss did not improve from {best_val_loss:.4f}.")

        # Early stopping
     #   if epochs_no_improve >= patience:
     #       print(f"Early stopping triggered after {epoch+1} epochs.")
     #       break

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


def evaluate_model(model, test_loader, label_encoder, device):
    # Load the best model weights
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.round(2)

    print("\nClassification Report:")
    print(df_report)

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {accuracy:.2%}")

def main():
    file_path = 'philosophy_data_edit.csv'
    X_train, X_temp, y_train, y_temp, label_encoder = load_data(file_path)
    from sklearn.model_selection import train_test_split
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    processor = TextProcessor()
    X_train_processed = processor.fit_transform(X_train)
    X_val_processed = processor.transform(X_val)
    X_test_processed = processor.transform(X_test)

    train_dataset = TextDataset(X_train_processed, y_train)
    val_dataset = TextDataset(X_val_processed, y_val)
    test_dataset = TextDataset(X_test_processed, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    vocab_size = processor.tokenizer.num_words
    embed_dim = 128
    hidden_dim = 256
    output_dim = len(label_encoder.classes_)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RNNModelWithAttention(vocab_size, embed_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=0)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0010, max_lr=0.002, step_size_up=2000)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=1)
    
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=10)

    print("\nFinal Model Evaluation on Test Data:")
    evaluate_model(model, test_loader, label_encoder, device)

if __name__ == "__main__":
    main()