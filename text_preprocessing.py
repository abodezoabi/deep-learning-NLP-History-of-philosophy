import pandas as pd
from text_processing import TextProcessor
import numpy as np


def load_data(filepath):

    data = pd.read_csv(filepath)
    train_data = data[data['Date'] < '2015-01-01']
    test_data = data[(data['Date'] >= '2016-01-01') & (data['Date'] <= '2016-07-01')]
    valid_data = data[(data['Date'] >= '2015-01-01') & (data['Date'] < '2016-01-01')]
    return train_data, test_data, valid_data, data


def process_data(full_data, split_data):
    """
    Process a CSV file to compute TF-IDF matrix for headlines.

    Parameters:
    filepath (str): Path to the CSV file. The file should have a column 'Label' for labels
    and other columns containing text headlines.

    Returns:
    tfidf_matrix: Sparse matrix of TF-IDF scores.
    labels: List of labels.
    feature_names: List of feature names (terms).
    """

    labels = split_data.iloc[:, 1].values
    labels_column = labels.reshape(-1, 1)  

    combined_headlines = []

    for _, row in full_data.iterrows():
        combined_text = ' '.join(str(row[col]) for col in row.index[2:27] if pd.notnull(row[col]))
        combined_headlines.append(combined_text)
    
    processor = TextProcessor()
    fit_cleaned_headlines = [" ".join(processor.clean_text(headlines)) for headlines in combined_headlines]

    combined_headlines = []

    for _, row in split_data.iterrows():
        combined_text = ' '.join(str(row[col]) for col in row.index[2:27] if pd.notnull(row[col]))
        combined_headlines.append(combined_text)
    
    transform_cleanes_headlines = [" ".join(processor.clean_text(headlines)) for headlines in combined_headlines]

    tfidf_matrix, feature_names = processor.compute_tfidf(fit_cleaned_headlines, transform_cleanes_headlines)

    tfidf_matrix_array = tfidf_matrix.toarray()  

    tfidf_matrix = np.hstack((labels_column, tfidf_matrix_array))  

    return tfidf_matrix

