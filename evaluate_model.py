from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

def evaluate_model(model_path, test_vectors, test_labels):
    """Load a trained model and evaluate it on the test data."""
    model = joblib.load(model_path)
    predictions = model.predict(test_vectors)
    pred_prob = model.predict_proba(test_vectors)[:, 1]

    # Classification Report
    print("Classification Report:")
    print(classification_report(test_labels, predictions))

    # ROC-AUC Score
    roc_auc = roc_auc_score(test_labels, pred_prob)
    print(f"ROC-AUC Score: {roc_auc:.2f}")

    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, predictions))
