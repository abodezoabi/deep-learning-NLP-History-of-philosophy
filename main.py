import sys
from text_preprocessing import load_data, process_data
from train_model import train_logistic_regression
from evaluate_model import evaluate_model

# Configurations
data_path = "Datasets/Combined_News_DJIA.csv"
stock_path = "Datasets/upload_DJIA_table.csv"
model_path = "Model/logistic_model.pth"  


"""
Parameters:
train_file (str): Path to the training data CSV file.
test_file (str): Path to the testing data CSV file.
model_path (str): Path to save the trained model (default is "logistic_model.pth").
tune (bool): If True, perform hyperparameter tuning (default is False).
"""

# Step 1: Load the data (train and test)
print("[INFO] Loading data...")
train_data, test_data, valid_data, data = load_data(data_path)

# Step 2: Process the training and test data to get TF-IDF vectors
print("[INFO] Processing training data...")
train_matrix = process_data(data, train_data)
train_labels = train_matrix[:, 0]  # First column is the label
train_vectors = train_matrix[:, 1:]  # Remaining columns are the TF-IDF features

print("[INFO] Processing validation data...")
valid_matrix = process_data(data, valid_data)  # Fixed this from train_data to valid_data
valid_labels = valid_matrix[:, 0]  # First column is the label
valid_vectors = valid_matrix[:, 1:]  # Remaining columns are the TF-IDF features

print("[INFO] Processing test data...")
test_matrix = process_data(data, test_data)
test_labels = test_matrix[:, 0]  # First column is the label
test_vectors = test_matrix[:, 1:]  # Remaining columns are the TF-IDF features

# Step 3: Train the Logistic Regression model using PyTorch
print("[INFO] Training the model...")
model = train_logistic_regression(
    train_vectors, train_labels, model_path, False, 
    validation_data=(valid_vectors, valid_labels)
)

# Step 4: Evaluate the model on the test data
print("[INFO] Evaluating the model...")
evaluate_model(model, test_vectors, test_labels)
