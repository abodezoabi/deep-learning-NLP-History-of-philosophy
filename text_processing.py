import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

def load_and_prepare_data(filepath, stock_filepath=None):
    """Load the dataset and optionally merge stock data."""
    data = pd.read_csv(filepath)
    
    # Optional: Merge stock price features if provided
    if stock_filepath:
        stock_data = pd.read_csv(stock_filepath)
        stock_data['Daily_Return'] = (stock_data['Adj Close'] - stock_data['Open']) / stock_data['Open'] * 100
        stock_data['Volatility'] = (stock_data['High'] - stock_data['Low']) / stock_data['Open'] * 100
        data = pd.merge(data, stock_data[['Date', 'Daily_Return', 'Volatility']], on='Date', how='left')
    
    # Split into training and testing sets
    train = data[data['Date'] < '2015-01-01']
    test = data[data['Date'] > '2014-12-31']
    return train, test

def combine_headlines(data):
    """Combine the top 25 headlines into a single string per row."""
    return data.iloc[:, 2:27].apply(lambda row: ' '.join(str(x) for x in row if pd.notnull(x)), axis=1)

def vectorize_text(train_headlines, test_headlines):
    """Convert text into numerical features using TF-IDF."""
    vectorizer = TfidfVectorizer(max_features=5000)
    train_vectors = vectorizer.fit_transform(train_headlines)
    test_vectors = vectorizer.transform(test_headlines)
    return vectorizer, train_vectors, test_vectors

def add_stock_features(train, test, train_vectors, test_vectors):
    """Add numerical stock features to the text vectors."""
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train[['Daily_Return', 'Volatility']].fillna(0))
    test_features = scaler.transform(test[['Daily_Return', 'Volatility']].fillna(0))
    train_combined = hstack([train_vectors, train_features])
    test_combined = hstack([test_vectors, test_features])
    return train_combined, test_combined
