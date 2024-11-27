# **Stock Market Prediction Using Daily News Headlines**

### **Overview**
This project was developed as part of the **Deep Learning and Natural Language Processing** course. The primary objective is to predict stock market movement (up or down) using daily news headlines and Dow Jones Industrial Average (DJIA) data. The project combines machine learning and NLP techniques to extract meaningful insights from textual data.

---

### **Key Features**
- **Natural Language Processing (NLP)**:
  - Text preprocessing using TF-IDF vectorization.
  - Feature extraction from top 25 daily news headlines.
- **Stock Market Features**:
  - Integration of stock-specific numerical features such as daily return and volatility.
- **Machine Learning**:
  - Logistic Regression for binary classification.
  - Hyperparameter tuning using GridSearchCV.
- **Evaluation**:
  - Performance evaluated using metrics like ROC-AUC, accuracy, and F1-score.
  - Confusion matrix for detailed analysis of predictions.

---

### **Technologies Used**
- **Scikit-Learn**: Machine learning model and evaluation.
- **Pandas & NumPy**: Data manipulation and preprocessing.
- **TF-IDF Vectorizer**: Feature engineering from text data.
- **Joblib**: Model saving and loading.

---

### **Future Enhancements**
- Implement Recurrent Neural Networks (RNNs) or Transformers for advanced sequential analysis.
- Experiment with pre-trained models like BERT for better text embeddings.
- Incorporate additional stock indices or datasets for broader applicability.

---

### **How to Run**
1. Clone the repository:
   ```bash
   git clone <repo_url>
