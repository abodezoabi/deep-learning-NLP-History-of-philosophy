import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import joblib  # To save the model, although PyTorch has its own method of saving
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)  # One output node for binary classification

    def forward(self, x):
        return torch.sigmoid(self.fc(x))  # Sigmoid for binary classification

def train_logistic_regression(x_data, y_data, model_path, epochs=100, lr=0.001, validation_split=0.2):
    """
    Train a Logistic Regression model with a validation set to evaluate its performance.
    
    Parameters:
        x_data (numpy.ndarray): Input features.
        y_data (numpy.ndarray): Labels.
        model_path (str): Path where the trained model will be saved.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        validation_split (float): Proportion of data to be used as validation set.
        
    Returns:
        model (LogisticRegressionModel): The trained Logistic Regression model.
    """
    # Split data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_data, y_data, test_size=validation_split, random_state=42
    )
    
    # Convert to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    # Create model instance
    input_dim = x_train.shape[1]
    model = LogisticRegressionModel(input_dim)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate Train Accuracy
        y_pred_binary_train = (y_pred >= 0.5).float()
        train_accuracy = (y_pred_binary_train == y_train).sum().item() / len(y_train)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = criterion(val_pred, y_val).item()
            val_pred_binary = (val_pred >= 0.5).float()
            val_accuracy = (val_pred_binary == y_val).sum().item() / len(y_val)
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Train Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy * 100:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%")
    
    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return model
def train_with_cross_validation(x_data, y_data, model_path, epochs=100, lr=0.001, k=5):
    """
    Train a Logistic Regression model using K-Fold Cross Validation.
    
    Parameters:
        x_data (numpy.ndarray): Input features.
        y_data (numpy.ndarray): Labels.
        model_path (str): Path where the trained model will be saved.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        k (int): Number of folds for cross-validation.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold = 1
    val_accuracies = []

    for train_index, val_index in kf.split(x_data):
        print(f"Fold {fold}/{k}")
        x_train, x_val = x_data[train_index], x_data[val_index]
        y_train, y_val = y_data[train_index], y_data[val_index]
        
        # Convert to PyTorch tensors
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        x_val = torch.tensor(x_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        
        # Create model instance
        input_dim = x_train.shape[1]
        model = LogisticRegressionModel(input_dim)
        
        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            model.train()
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate Train Accuracy
            y_pred_binary_train = (y_pred >= 0.5).float()
            train_accuracy = (y_pred_binary_train == y_train).sum().item() / len(y_train)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_pred_binary = (val_pred >= 0.5).float()
            val_accuracy = (val_pred_binary == y_val).sum().item() / len(y_val)
            val_accuracies.append(val_accuracy)
        
        print(f"Fold {fold} Epoch {epoch + 1}/{epochs}, Train Accuracy: {train_accuracy * 100:.2f}%, "
              f"Validation Accuracy: {val_accuracy * 100:.2f}%")
        fold += 1

    # Print average validation accuracy
    avg_accuracy = sum(val_accuracies) / k
    print(f"Average Validation Accuracy: {avg_accuracy * 100:.2f}%")
    
    return model
