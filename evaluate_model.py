import torch

def evaluate_model(model, test_vectors, test_labels):
    """
    Evaluate the trained model on the test data.

    Parameters:
        model (LogisticRegressionModel): The trained model.
        test_vectors (numpy.array): The test input features.
        test_labels (numpy.array): The true labels of the test data.
    """
    model.eval()  # Set the model to evaluation mode
    test_vectors = torch.tensor(test_vectors, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32).view(-1, 1)
    
    with torch.no_grad():
        outputs = model(test_vectors)
        predicted = (outputs >= 0.5).float()  # Threshold at 0.5 for binary classification
        correct = (predicted == test_labels).float().sum()
        accuracy = correct / len(test_labels)
        print(f"Test Accuracy: {accuracy.item():.4f}")
