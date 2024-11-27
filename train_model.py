from sklearn.linear_model import LogisticRegression  # For Logistic Regression model
from sklearn.model_selection import GridSearchCV  # GridSearchCV for hyperparameter tuning
import joblib  # For saving and loading the trained model

def train_logistic_regression(train_vectors, train_labels, model_path="logistic_model.pkl", tune=False):
    """
    Train a Logistic Regression model with optional hyperparameter tuning and save it to disk.
    
    Parameters:
        train_vectors (sparse matrix): Input features for training, e.g., TF-IDF vectors.
        train_labels (array-like): Labels corresponding to the training data (0 or 1).
        model_path (str): Path where the trained model will be saved. Default is "logistic_model.pkl".
        tune (bool): If True, perform hyperparameter tuning using GridSearchCV. Default is False.
        
    Returns:
        model (LogisticRegression): The trained Logistic Regression model.
    """
    
    if tune:
        # Define the hyperparameter grid to search over
        param_grid = {
            'C': [0.01, 0.1, 1, 10],  # Regularization strength (smaller values mean stronger regularization)
            'max_iter': [100, 200, 500]  # Maximum number of iterations for optimization
        }
        
        # Set up GridSearchCV with Logistic Regression and cross-validation (cv=3)
        grid = GridSearchCV(
            LogisticRegression(class_weight='balanced'),  # Use balanced class weights to handle imbalanced data
            param_grid,  # Specify the parameter grid
            cv=3,  # Use 3-fold cross-validation
            scoring='roc_auc'  # Evaluate using the ROC-AUC metric
        )
        
        # Fit GridSearchCV to find the best parameters
        grid.fit(train_vectors, train_labels)
        
        # Get the best model based on cross-validation results
        model = grid.best_estimator_
        
        # Print the best parameters found during the search
        print(f"Best Parameters: {grid.best_params_}")
    else:
        # Train Logistic Regression directly without tuning
        model = LogisticRegression(max_iter=200, class_weight='balanced')  # Balanced class weights handle imbalance
        model.fit(train_vectors, train_labels)  # Train the model on the provided data
    
    # Save the trained model to the specified file path
    joblib.dump(model, model_path)  
    print(f"Model saved to {model_path}")  # Notify the user of the saved model
    
    # Return the trained Logistic Regression model
    return model
