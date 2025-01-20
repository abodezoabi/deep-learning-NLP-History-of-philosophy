# Text Classification Project: History of Philosophy

This project explores various machine learning and deep learning models to classify sentences into one of four major philosophical schools: Aristotle, German Idealism, Plato, and Continental Philosophy. The dataset contains over 150,000 sentences extracted from philosophical texts.

## Project Stages and Model Performance

### Stage 1: Baseline Model
- **Model**: Basic rule-based classifier.
- **Accuracy**: 30%
- **Discussion**: Demonstrated limited capability in handling complex text classification, lacking nuanced contextual understanding.

### Stage 2: Logistic Regression with Softmax
- **Model**: Logistic regression with a softmax output layer for multi-class classification, using cross-entropy loss.
- **Variants**:
  - **Bag of Words + Ridge + SGD**: 65% accuracy
  - **TF-IDF + Ridge + SGD**: 25% accuracy
  - **Word2Vec + Ridge + LBFGS**: 81% accuracy
- **Discussion**: Showed significant improvement by modeling probabilities across multiple classes. Best performance achieved with Word2Vec feature representation.

### Stage 3: Fully Connected Neural Networks (FCNNs)
- **Model**: FCNNs with multiple dense layers and dropout to prevent overfitting.
- **Accuracy**: 84%
- **Discussion**: Captured more complex patterns in the data, further improving classification accuracy.

### Stage 4: RNNs and Bidirectional LSTMs
- **Model**: Started with RNNs and enhanced with Bidirectional LSTMs, incorporating dropout and cyclic learning rate scheduling.
- **Initial Accuracy**: 85%
- **Transition to BERT**: Shifted to BERT due to its superior contextual processing capabilities.

### Stage 5: BERT Implementation
- **Model**: BERT (Bidirectional Encoder Representations from Transformers) leveraging pre-trained language models and attention mechanisms.
- **Accuracy**: Close to 90%
- **Discussion**: Achieved the highest accuracy, effectively handling complex contextual information in text classification.

## How to Run This Project

### Prerequisites
Ensure you have Python and the necessary libraries installed:
- PyTorch
- Transformers by Hugging Face
- Scikit-Learn
- Pandas

### Installation
Clone the repository and install the required Python packages:
