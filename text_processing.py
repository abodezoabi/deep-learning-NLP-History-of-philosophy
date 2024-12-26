import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TextProcessor:
    def __init__(self, n=2):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english')) 
        self.n = n  # This will control the size of the N-grams (e.g., 2 for bigrams)

    def clean_text(self, text):
        text = re.sub(r"b[\'\"]", '', text)  
        text = re.sub(r"[^\x00-\x7F]+", '', text)  # Remove non-ASCII characters
        text = re.sub(r"[^a-zA-Z\s]", '', text.lower())  # Remove non-alphabetic characters
        tokens = word_tokenize(text)  # Tokenize the text
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]  # Lemmatize and remove stopwords
        return tokens

    def N_Grams(self, documents):
        """
        Convert documents into n-grams after cleaning and lemmatizing the text.

        Parameters:
        documents (list of str): List of text documents to process.

        Returns:
        ngram_documents (list of str): List of documents with n-grams instead of single words.
        """
        ngram_documents = [] 
        for doc in documents:
            tokens = doc.split()  # Already cleaned, just split into tokens
            if len(tokens) >= self.n:  # Ensure there are enough tokens to create n-grams
                ngrams_list = list(ngrams(tokens, self.n))  # Create n-grams
                ngram_text = [' '.join(ngram) for ngram in ngrams_list]  # Join each n-gram into a string
                ngram_documents.append(' '.join(ngram_text))  # Join all n-grams in the document
            else:
                # If not enough tokens to form N-grams, just append the tokens as-is
                print(f"Document skipped (insufficient tokens for {self.n}-grams): {doc}")  # Print the document
                ngram_documents.append(' '.join(tokens))
        return ngram_documents

    def compute_tfidf(self, documents, data):
        """
        Compute TF-IDF scores for a list of documents.

        Parameters:
        documents (list of str): List of text documents.

        Returns:
        tfidf_matrix: Sparse matrix of TF-IDF scores.
        feature_names: List of feature names (terms).
        """
        # Define ngram_range parameter to use N-grams
        vectorizer = TfidfVectorizer(max_features=10000)
        vectorizer.fit(documents)
        tfidf_matrix = vectorizer.transform(data)
        feature_names = vectorizer.get_feature_names_out()
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        print(f"Number of features: {len(feature_names)}")
        print(f"Number of rows in 'data': {len(data)}")
        with open('tf_idf_words.txt', 'w') as f:
            f.write("Feature names (terms) from TF-IDF vectorizer:\n")
            for feature in feature_names:
                f.write(f"{feature}\n")
        return tfidf_matrix, feature_names
