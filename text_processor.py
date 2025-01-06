import pandas as pd
import re
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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
    
    def read_data(self, filepath):
        
        data = pd.read_csv(filepath)
        sentiment_counts = data['school'].value_counts()

        print(sentiment_counts) # List of schools name and how much have for each school

        schools = ['plato']
        for school in schools:
            data[school] = data['school'].apply(lambda x: 1 if x == school else 0)

        data = data.drop(columns=['school'])

        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        
        y_train = train_data.iloc[:, 1:].values  # intelligence_score column 
        y_train = y_train.reshape(-1,1)

        y_test = test_data.iloc[:, 1:].values
        y_test = y_test.reshape(-1,1)

        return train_data, test_data, y_train, y_test
    
    def compute_bow(self, documents, data):
        """
        Compute Bag-Of-Bow scores for a list of documents.

        Parameters:
        documents (list of str): List of text documents.

        Returns:
        bow_matrix: Sparse matrix of Bag-Of-Bow scores.
        feature_names: List of feature names (terms).
        """
        # Define ngram_range parameter to use N-grams
        vectorizer = CountVectorizer()
        vectorizer.fit(documents)
        bow_matrix = vectorizer.transform(data)
        feature_names = vectorizer.get_feature_names_out()
        with open('bow_words.txt', 'w') as f:
            f.write("Feature names (terms) from Bag Of Words vectorizer:\n")
            for feature in feature_names:
                f.write(f"{feature}\n")
        bow_matrix_array = bow_matrix.toarray()
        print(f'Data shape : {bow_matrix_array.shape}')
        return bow_matrix_array
    
    def compute_tfidf(self, documents, data):
        """
        Compute TF-Idata scores for a list of documents.

        Parameters:
        documents (list of str): List of text documents.

        Returns:
        tfidata_matrix: Sparse matrix of TF-Idata scores.
        feature_names: List of feature names (terms).
        """
        # Define ngram_range parameter to use N-grams
        vectorizer = TfidfVectorizer()
        vectorizer.fit(documents)
        tfidf_matrix = vectorizer.transform(data)
        feature_names = vectorizer.get_feature_names_out()        
        with open('tfidf_words.txt', 'w') as f:
            f.write("Feature names (terms) from TF-Idata vectorizer:\n")
            for feature in feature_names:
                f.write(f"{feature}\n")

        tfidf_matrix_array = tfidf_matrix.toarray()
        print(f'Data shape : {tfidf_matrix_array.shape}')
        return tfidf_matrix_array, feature_names
    
    def convert_to_vector(self, fit_data, transfrom_data, labels):
        """
        Process a CSV file to compute TF-Idata matrix for headlines.

        Parameters:
        filepath (str): Path to the CSV file. The file should have a column 'Label' for labels
        and other columns containing text headlines.

        Returns:
        tfidata_matrix: Sparse matrix of TF-Idata scores.
        bow_matrix: Sparse matrix of bow scores.
        labels: List of labels.
        feature_names: List of feature names (terms).
        """ 
        # Extract all texts from the 'sentence_str' column
        combined_texts = fit_data['sentence_str'].astype(str).tolist()

        # Clean the text
        fit_cleaned_texts = [" ".join(self.clean_text(text)) for text in combined_texts]

        # Extract all texts from the 'sentence_str' column
        combined_texts = transfrom_data['sentence_str'].astype(str).tolist()

        # Clean the text
        transform_cleaned_texts = [" ".join(self.clean_text(text)) for text in combined_texts]

        # tfidata_matrix = processor.compute_tfidf(fit_cleaned_headlines, transform_cleanes_headlines)

        bow_matrix = self.compute_bow(fit_cleaned_texts, transform_cleaned_texts)

        # tfidata_matrix = np.hstack((labels, tfidata_matrix_array))  

        bow_matrix = np.hstack((labels, bow_matrix))

        return bow_matrix



