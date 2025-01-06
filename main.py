from evaluate_model import evaluate_model
from text_processor import TextProcessor
from train_model import train_logistic_regression, train_with_cross_validation


data_path = "Dataset/philosophy_data.csv"
model_path = "Model/model.pth"

processor = TextProcessor()

print("[INFO] Read Data .....")
train_data, test_data, y_train, y_test = processor.read_data(data_path)

print("[INFO] Convert train Data to vectors .....")
train_matrix = processor.convert_to_vector(train_data, train_data, y_train)

x_train = train_matrix[:, 1:]  # Remaining columns are the bag of bow features
y_train = train_matrix[:, 0]  # First column is the label

print("[INFO] Convert test Data to vectors .....")
test_matrix = processor.convert_to_vector(train_data, test_data, y_test)

x_test = test_matrix[:, 1:]  # Remaining columns are the bag of bow features
y_test = test_matrix[:, 0]  # First column is the label

print("[INFO] Training the model...")
model = train_logistic_regression(x_train, y_train, model_path)

print("[INFO] Evaluate model...")
evaluate_model(model, x_test, y_test)

print("[INFO] Training the model with cross validations...")
model = train_with_cross_validation(x_train, y_train, model_path)

print("[INFO] Evaluate model...")
evaluate_model(model, x_test, y_test)









