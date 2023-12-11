# Example: Dummy model for illustration purposes
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from neurocraft.backend.dyslexia_classifier.feature_engineering import FeatureEngineer
from neurocraft.backend.dyslexia_classifier.text_preprocessor import TextPreprocessor
from pathlib import Path

from neurocraft.params import *

class DyslexiaData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()
        self.feature_engineer = FeatureEngineer()
        self.text_preprocessor = TextPreprocessor()


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

        # Instantiate the FeatureEngineer class
        feature_engineer = FeatureEngineer()

        # Call the methods from the FeatureEngineer class on the text
        avg_word_length = feature_engineer.avg_word_length(text)
        avg_syllaba_word = feature_engineer.avg_syllaba_word(text)
        vowel_count = feature_engineer.count_vowels(text)
        punctuation_count = feature_engineer.count_punctuation(text)
        capital_char_ratio = feature_engineer.ratio_capital_chars(text)
        capital_word_count = feature_engineer.count_capital_words(text)
        word_length_std = feature_engineer.word_length_std(text)
        sentence_len = feature_engineer.sentence_len(text)

        # Combine the results into a dictionary
        features = {
            'avg_word_length': avg_word_length,
            'avg_syllaba_word': avg_syllaba_word,
            'vowel_count': vowel_count,
            'punctuation_count': punctuation_count,
            'capital_char_ratio': capital_char_ratio,
            'capital_word_count': capital_word_count,
            'word_length_std': word_length_std,
            'sentence_len': sentence_len
        }

        # Return the features
        return features

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

    def embed_and_pad_data(self, X_train_text, X_test_text):
        """
        Converts the text data into word embeddings and pads the sequences.
        """
        word2vec = Word2Vec(sentences=X_train_text, min_count=10)

        def embed_sentence(word2vec, sentence):
            embedded_sentence = []
            for word in sentence:
                if word in word2vec.wv:
                    embedded_sentence.append(word2vec.wv[word])

            return np.array(embedded_sentence)

        def embedding(word2vec, sentences):
            embed = []

            for sentence in sentences:
                embedded_sentence = embed_sentence(word2vec, sentence)
                embed.append(embedded_sentence)

            return embed

        X_train_text = embedding(word2vec, X_train_text)
        X_test_text = embedding(word2vec, X_test_text)

        maxlen=105

        X_train_pad = pad_sequences(X_train_text, dtype=float, padding='post', maxlen=maxlen)
        X_test_pad = pad_sequences(X_test_text, dtype=float, padding='post', maxlen=maxlen)

        return X_train_pad, X_test_pad
