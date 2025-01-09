import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

def train_fnn(
    x_data, 
    y_data, 
    model, 
    epochs=100, 
    lr=0.001, 
    batch_size=32, 
    device='cpu'
):
    """
    Train a Fully Connected Neural Network (FNN).
    """
    # Split into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42
    )

    # Convert to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            y_val_pred = model(x_val)
            val_loss = criterion(y_val_pred, y_val).item()
            val_accuracy = (torch.argmax(y_val_pred, dim=1) == y_val).sum().item() / len(y_val)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%")

    print("Training complete!")
    return model
