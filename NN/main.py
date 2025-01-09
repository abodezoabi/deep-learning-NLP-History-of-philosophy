import torch
from text_processor import TextProcessor
from nn_model import FullyConnectedNN
from train_nn import train_fnn
from evaluate_nn import evaluate_fnn

# Paths and configurations
data_path = "Dataset/philosophy_data_edit.csv"  # Path to your dataset
model_save_path = "Model/fnn_model.pth"         # Path to save the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_dim = 100         # Adjust based on vectorization output
hidden_dim = 1024        # Number of neurons in the hidden layer
num_classes = 3         # Adjust based on your dataset
dropout_rate = 0.6      # Dropout rate for regularization
epochs = 200             # Number of training epochs
learning_rate = 0.01  # Learning rate for optimizer
batch_size = 64         # Batch size for training

# Step 1: Data Preprocessing
print("[INFO] Processing data...")
processor = TextProcessor()

# Choose vectorization method (1 - BoW, 2 - TF-IDF, 3 - Word2Vec)
vectorizer_choice = 3  # Adjust based on your preference

if vectorizer_choice == 1:
    train_data, test_data, y_train, y_test, class_names = processor.read_data(data_path)
    train_vectors = processor.convert_to_vector(train_data, train_data, y_train, vectorizer='bow')
    test_vectors = processor.convert_to_vector(train_data, test_data, y_test, vectorizer='bow')
elif vectorizer_choice == 2:
    train_data, test_data, y_train, y_test, class_names = processor.read_data(data_path)
    train_vectors = processor.convert_to_vector(train_data, train_data, y_train, vectorizer='tfidf')
    test_vectors = processor.convert_to_vector(train_data, test_data, y_test, vectorizer='tfidf')
elif vectorizer_choice == 3:
    train_data, test_data, y_train, y_test, class_names = processor.read_data(data_path)
    word2vec_model = processor.train_word2vec(train_data['sentence_str'].tolist())
    train_vectors = processor.convert_to_word2vec(word2vec_model, train_data['sentence_str'].tolist())
    test_vectors = processor.convert_to_word2vec(word2vec_model, test_data['sentence_str'].tolist())
else:
    raise ValueError("Invalid vectorizer choice. Please select 1, 2, or 3.")

# Prepare data for training
x_train = train_vectors
y_train = y_train.argmax(axis=1)  # Convert one-hot labels to indices
x_test = test_vectors
y_test = y_test.argmax(axis=1)    # Convert one-hot labels to indices

# Step 2: Define the Model
print("[INFO] Defining the neural network model...")
model = FullyConnectedNN(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout_rate=dropout_rate)
model.to(device)

# Step 3: Train the Model
print("[INFO] Training the model...")
model = train_fnn(x_train, y_train, model, epochs=epochs, lr=learning_rate, batch_size=batch_size, device=device)

# Save the trained model
torch.save(model.state_dict(), model_save_path)
print(f"[INFO] Model saved to {model_save_path}")

# Step 4: Evaluate the Model
print("[INFO] Evaluating the model...")
accuracy = evaluate_fnn(model, x_test, y_test, class_names, device=device)

print(f"[INFO] Final Test Accuracy: {accuracy * 100:.2f}%")
