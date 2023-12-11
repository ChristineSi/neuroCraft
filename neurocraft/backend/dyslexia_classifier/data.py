# Example: Dummy model for illustration purposes
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from neurocraft.backend.dyslexia_classifier.text_preprocessor import TextPreprocessor
from neurocraft.backend.dyslexia_classifier.text_preprocessor import EmbeddingCreator
from pathlib import Path

from neurocraft.params import *

class DyslexiaData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()
        self.text_preprocessor = TextPreprocessor()
        self.embedding_creator = EmbeddingCreator()



    def load_data(self):
        """
        Load and preprocess the data from a CSV file.
        """
        df = pd.read_csv(self.file_path)
        return df

    def preprocess_data(self):
        """
        Preprocesses the input text for dyslexia classification and text simplification.
        """
        drop_columns = ['Last Changed', 'URL', 'Anthology', 'MPAA \n#Max', 'Pub Year', 'MPAA\n#Avg', 'License', 'British Words', 'firstPlace_pred', 'secondPlace_pred', 'thirdPlace_pred',
       'fourthPlace_pred', 'fifthPlace_pred', 'sixthPlace_pred', 'ID', 'Author', 'Title', 'Source', 'Category', 'Location', 'MPAA\nMax', 'BT s.e.', 'Kaggle split']
        self.data = self.data.drop(columns=drop_columns)

        # Calculating quantiles for bin edges
        quantiles = self.data['BT Easiness']..quantile([0, 0.3333, 0.6667, 1]).tolist()

        # Correct number of labels for 3 bins
        labels_dict = {'hard':0, 'acceptable':1, 'easy':2}

        # Using 'quantiles' for bins and including 6 labels
        self.data['BT Easiness'] = pd.cut(
            x=self.data['BT Easiness'],
            bins=quantiles,
            labels=labels_dict.values(),
            include_lowest=True
        )

        return self.data

    def preprocess_text(self, text):
        # Preprocess the text using the methods from the TextPreprocessor class
        text = self.text_preprocessor.preprocess(text)
        return text


    def split_data(self):
        """
        Splits the data into training and testing sets.
        """
        X = self.data.drop(columns=['BT Easiness'])
        y = self.data['BT Easiness']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Separate the 'Excerpt' column from the training and testing sets
        X_train_text = X_train['Excerpt'].values
        X_test_text = X_test['Excerpt'].values
        return X_train_text, X_test_text, y_train, y_test

    def embed_data(self, X_train_text, X_test_text):
        """
        Converts the text data into word embeddings and pads the sequences.
        """

        X_train_text = self.embedding_creator.create_embeddings(X_train_text)
        X_test_text = self.embedding_creator.create_embeddings(X_test_text)
        return X_train_text, X_test_text
