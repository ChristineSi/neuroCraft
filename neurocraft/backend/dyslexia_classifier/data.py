# Example: Dummy model for illustration purposes
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from neurocraft.backend.dyslexia_classifier.text_preprocessor import TextPreprocessor
from neurocraft.backend.dyslexia_classifier.text_preprocessor import EmbeddingCreator
from pathlib import Path

from neurocraft.params import *


text_preprocessor = TextPreprocessor()
embedding_creator = EmbeddingCreator()

def load_data(file_path):
    """
    Load and preprocess the data from a CSV file.
    """
    df = pd.read_csv(file_path)
    return df

def preprocess_train_test_data(data):
    """
    Preprocesses the input text for dyslexia classification and text simplification.
    """
    drop_columns = ['Last Changed', 'URL', 'Anthology', 'MPAA \n#Max', 'Pub Year', 'MPAA\n#Avg', 'License', 'British Words', 'firstPlace_pred', 'secondPlace_pred', 'thirdPlace_pred',
    'fourthPlace_pred', 'fifthPlace_pred', 'sixthPlace_pred', 'ID', 'Author', 'Title', 'Source', 'Category', 'Location', 'MPAA\nMax', 'BT s.e.', 'Kaggle split']
    data = data.drop(columns=drop_columns)

    # Calculating quantiles for bin edges
    quantiles = data['BT Easiness'].quantile([0, 0.3333, 0.6667, 1]).tolist()

    # Correct number of labels for 3 bins
    labels_dict = {'hard':0, 'acceptable':1, 'easy':2}

    # Using 'quantiles' for bins and including 6 labels
    data['BT Easiness'] = pd.cut(
        x=data['BT Easiness'],
        bins=quantiles,
        labels=labels_dict.values(),
        include_lowest=True
    )

    return data

def preprocess_text(text):
    # Preprocess the text using the methods from the TextPreprocessor class
    text = text_preprocessor.preprocess(text)
    return text


def embed_data(text):
    """
    Converts the text data into word embeddings and pads the sequences.
    """
    text = embedding_creator.create_embeddings(text)
    return text

def define_X_y(data):
    """
    Defines the X and y variables for the model.
    """
    X = data['Excerpt']
    y = data['BT Easiness']
    return X, y

def split_data(X, y):
    """
    Splits the data into training and testing sets.
    """
    X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Separate the 'Excerpt' column from the training and testing sets
    X_train_text = X_train_text.values
    X_test_text = X_test_text.values
    return X_train_text, X_test_text, y_train, y_test
