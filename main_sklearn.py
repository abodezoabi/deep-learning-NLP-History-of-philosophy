from sklearn.linear_model import LogisticRegression
from text_processor import load_data, prepare_text, compute_bow, compute_tfidf
from sklearn.metrics import accuracy_score, precision_score, recall_score

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
train, y_train, test, y_test = load_data(data_path)

# Step 2: Prepare text for processing
train_corpus, test_corpus = prepare_text(train, test)

# Step 3: Compute bag-of-words (BoW) features
x_train, x_test = compute_bow(train_corpus, test_corpus)

# Step 4: Train the Logistic Regression classifier
lr_classifier = LogisticRegression()
lr_classifier.fit(x_train, y_train)

# Step 5: Predict on the training set
lr_y_pred = lr_classifier.predict(x_test)

# Step 6: Calculate metrics for the test set
test_accuracy = accuracy_score(y_test, lr_y_pred)
test_precision = precision_score(y_test, lr_y_pred)
test_recall = recall_score(y_test, lr_y_pred)
print("Test Accuracy score is: {}%".format(round(test_accuracy * 100, 2)))
print("Test Precision score is: {}".format(round(test_precision, 2)))
print("Test Recall score is: {}".format(round(test_recall, 2)))
