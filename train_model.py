import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import joblib  # To save the model, although PyTorch has its own method of saving

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)  # One output node for binary classification

    def forward(self, x):
        return torch.sigmoid(self.fc(x))  # Sigmoid for binary classification

def train_logistic_regression(train_vectors, train_labels, model_path="logistic_model.pth", tune=False, validation_data=None, epochs=10, lr=0.001):
    """
    Train a Logistic Regression model with optional hyperparameter tuning using PyTorch.
    
    Parameters:
        train_vectors (torch.Tensor): Input features for training.
        train_labels (torch.Tensor): Labels for training data.
        model_path (str): Path where the trained model will be saved.
        tune (bool): Hyperparameter tuning flag.
        validation_data (tuple): Validation data (vectors, labels).
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        
    Returns:
        model (LogisticRegressionModel): The trained Logistic Regression model.
    """

    # Create model instance
    input_dim = train_vectors.shape[1]  # Number of features in the dataset
    model = LogisticRegressionModel(input_dim)

    # Loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross-Entropy loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer
    
    # Convert to PyTorch tensors
    train_vectors = torch.tensor(train_vectors, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32).view(-1, 1)  # Shape needs to be [N, 1]
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(train_vectors)
        loss = criterion(outputs, train_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate training accuracy
        predicted = (outputs >= 0.5).float()  # Threshold of 0.5 for binary classification
        train_accuracy = (predicted == train_labels).float().mean()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Training Accuracy: {train_accuracy.item():.4f}")
    
    # If validation data is provided, evaluate the model on validation set
    if validation_data:
        valid_vectors, valid_labels = validation_data
        model.eval()
        valid_vectors = torch.tensor(valid_vectors, dtype=torch.float32)
        valid_labels = torch.tensor(valid_labels, dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            valid_outputs = model(valid_vectors)
            predicted = (valid_outputs >= 0.5).float()  # Threshold of 0.5 for binary classification
            valid_accuracy = (predicted == valid_labels).float().mean()
            print(f"Validation Accuracy: {valid_accuracy.item():.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), model_path)  # Save only model's state_dict
    print(f"Model saved to {model_path}")
    
    return model

