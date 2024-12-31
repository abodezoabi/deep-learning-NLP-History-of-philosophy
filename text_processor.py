import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath, encoding = 'ISO-8859-1')
    df.dropna(inplace=True)
    data = df.copy()
    data.reset_index(inplace=True)
    
    train = data[data['Date'] < '20150101']
    test = data[data['Date'] > '20141231']
    y_train = train['Label']
    train = train.iloc[:, 3:28]

    y_test = test['Label']
    test = test.iloc[:, 3:28]

    return train, y_train, test, y_test


def prepare_text(train, test):
    
    train.replace(to_replace='[^a-zA-Z]', value=' ', regex=True, inplace=True)
    test.replace(to_replace='[^a-zA-Z]', value=' ', regex=True, inplace=True)

    new_columns = [str(i) for i in range(0,25)]
    
    train.columns = new_columns
    test.columns = new_columns

    for i in new_columns:
        train[i] = train[i].str.lower()
        test[i] = test[i].str.lower()

    train_headlines = []
    test_headlines = []

    for row in range(0, train.shape[0]):
        train_headlines.append(' '.join(str(x) for x in train.iloc[row, 0:25]))

    for row in range(0, test.shape[0]):
        test_headlines.append(' '.join(str(x) for x in test.iloc[row, 0:25]))
        
    ps = PorterStemmer()
    train_corpus = []
    test_corpus = []

    for i in range(0, len(train_headlines)):
        words = train_headlines[i].split()
        words = [word for word in words if word not in set(stopwords.words('english'))]
        words = [ps.stem(word) for word in words]
        headline = ' '.join(words)
        train_corpus.append(headline)

    for i in range(0, len(test_headlines)):
        words = test_headlines[i].split()
        words = [word for word in words if word not in set(stopwords.words('english'))]
        words = [ps.stem(word) for word in words]
        headline = ' '.join(words)
        test_corpus.append(headline)

    return train_corpus, test_corpus


def compute_bow(train, test):
        """
        Compute Bag of Words (BoW) representation for a list of documents.

        Parameters:
        documents (list of str): List of text documents.
        data (list of str): Data to transform into BoW representation.

        Returns:
        bow_matrix: Sparse matrix of BoW counts.
        feature_names: List of feature names (terms).
        """
        # Initialize the CountVectorizer (BoW)
        vectorizer = CountVectorizer(max_features=10000)

        # Fit the model on the documents to learn the vocabulary
        train_vec = vectorizer.fit_transform(train)
        # Val true - transform train
        # Transform the data (documents) into a BoW matrix
        test_vec = vectorizer.transform(test)

        # Get the feature names (terms)
        train_feature_names = vectorizer.get_feature_names_out()

        print(f"Train BoW matrix shape: {train_vec.shape}")
        print(f"Number of features: {len(train_feature_names)}")
        print(f"Number of rows in 'data': {len(train)}")

        # Optionally, save the feature names to a file
        with open('bow_words.txt', 'w') as f:
            f.write("Feature names (terms) from CountVectorizer:\n")
            for feature in train_feature_names:
                f.write(f"{feature}\n")
        
        x_train = train_vec.toarray()
        x_test = test_vec.toarray()  

        return x_train, x_test

def compute_tfidf(train, test):
        """
        Compute TF-IDF scores for a list of documents.

        Parameters:
        documents (list of str): List of text documents.

        Returns:
        tfidf_matrix: Sparse matrix of TF-IDF scores.
        feature_names: List of feature names (terms).
        """
        # Define ngram_range parameter to use N-grams
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(2,2))
        # Fit the model on the documents to learn the vocabulary
        train_vec = vectorizer.fit_transform(train)
        # Val true - transform train
        # Transform the data (documents) into a BoW matrix
        test_vec = vectorizer.transform(test)

        # Get the feature names (terms)
        train_feature_names = vectorizer.get_feature_names_out()

        print(f"Train BoW matrix shape: {train_vec.shape}")
        print(f"Number of features: {len(train_feature_names)}")
        print(f"Number of rows in 'data': {len(train)}")

        # Optionally, save the feature names to a file
        with open('tf_idf.txt', 'w') as f:
            f.write("Feature names (terms) from CountVectorizer:\n")
            for feature in train_feature_names:
                f.write(f"{feature}\n")
        
        x_train = train_vec.toarray()
        x_test = test_vec.toarray()  

        return x_train, x_test