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

def train_logistic_regression(x_train, y_train, model_path="logistic_model.pth", epochs=100, lr=0.001):
    """
    Train a Logistic Regression model with optional hyperparameter tuning using PyTorch.
    
    Parameters:
        x_train (torch.Tensor): Input features for training.
        y_train (torch.Tensor): Labels for training data.
        model_path (str): Path where the trained model will be saved.
        tune (bool): Hyperparameter tuning flag.
        validation_data (tuple): Validation data (vectors, labels).
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        
    Returns:
        model (LogisticRegressionModel): The trained Logistic Regression model.
    """

    # Create model instance
    input_dim = x_train.shape[1]  # Number of features in the dataset
    model = LogisticRegressionModel(input_dim)

    # Loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross-Entropy loss for binary classification
    optimizer = optim.SGD(model.parameters(), lr=lr)  # Adam optimizer
    
    # Convert to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Shape needs to be [N, 1]
    
    # Training loop
    for epoch in range(epochs):
        # Forward pass: compute the predicted y
        y_pred = model(x_train)
        
        # Compute the loss
        loss = criterion(y_pred, y_train)
        
        # Backward pass and optimization
        optimizer.zero_grad()  # Clear gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weight

        # Print the loss and current weight
        print(f'Epoch {epoch}: Loss = {loss.item()}, w = {model.w.item()}')
    
    # Save the trained model
    torch.save(model.state_dict(), model_path)  # Save only model's state_dict
    print(f"Model saved to {model_path}")
    
    return model

