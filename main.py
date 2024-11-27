import text_processing
import train_model
import evaluate_model

# Configurations
data_path = "Combined_News_DJIA.csv"
stock_path = "upload_DJIA_table.csv"
model_path = "logistic_model.pkl"

# Step 1: Load and process data
train, test = text_processing.load_and_prepare_data(data_path, stock_filepath=stock_path)
train_headlines = text_processing.combine_headlines(train)
test_headlines = text_processing.combine_headlines(test)

# Step 2: Vectorize text
vectorizer, train_vectors, test_vectors = text_processing.vectorize_text(train_headlines, test_headlines)

# Step 3: Add stock price features
train_combined, test_combined = text_processing.add_stock_features(train, test, train_vectors, test_vectors)

# Step 4: Train the model
model = train_model.train_logistic_regression(train_combined, train["Label"], model_path, tune=True)

# Step 5: Evaluate the model
evaluate_model.evaluate_model(model_path, test_combined, test["Label"])
