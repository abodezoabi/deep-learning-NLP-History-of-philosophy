from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from text_processor import TextProcessor

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

lr_classifier = LogisticRegression()
lr_classifier.fit(x_train, y_train)

lr_y_pred = lr_classifier.predict(x_train)

train_accuracy = accuracy_score(y_train, lr_y_pred)
train_precision = precision_score(y_train, lr_y_pred)
train_recall = recall_score(y_train, lr_y_pred)
print("Train Accuracy score is: {}%".format(round(train_accuracy * 100, 2)))
print("Train Precision score is: {}".format(round(train_precision, 2)))

print("[INFO] Evaluate model...")

lr_y_pred = lr_classifier.predict(x_test)

test_accuracy = accuracy_score(y_test, lr_y_pred)
test_precision = precision_score(y_test, lr_y_pred)
test_recall = recall_score(y_test, lr_y_pred)
print("Test Accuracy score is: {}%".format(round(test_accuracy * 100, 2)))
print("Test Precision score is: {}".format(round(test_precision, 2)))

