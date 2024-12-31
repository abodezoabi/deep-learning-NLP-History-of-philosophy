from train_model import train_logistic_regression
from evaluate_model import evaluate_model
from text_processor import load_data, prepare_text, compute_bow, compute_tfidf

# Configurations
data_path = "Datasets/Combined_News_DJIA.csv"
stock_path = "Datasets/upload_DJIA_table.csv"
model_path = "Model/logistic_model.pth"  

# Step 1: Load the data (train and test)
print("[INFO] Loading data...")
train, train_label, test, test_label = load_data(data_path)

# Step 2: Prepare text for processing
print("[INFO] Prepare text data for vectorize")
train_corpus, test_corpus = prepare_text(train, test)

# Step 3: Compute bag-of-words (BoW) features
print("[INFO] Computing bag of bow")
x_train, x_test = compute_bow(train_corpus, test_corpus)

# Step 4: Train the Logistic Regression classifier
y_train = train_label.iloc[:, 1].values
y_test = test_label.iloc[:, 1].values

print("[INFO] Training with logistic regression")
model = train_logistic_regression(x_train, y_train, model_path)

# Step: Evaluate the test on the new model
evaluate_model(model, x_test, y_test)

