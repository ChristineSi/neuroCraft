# Example: Dummy model for illustration purposes
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from feature_engineering import FeatureEngineer
from text_preprocessor import TextPreprocessor

class DyslexiaData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.feature_engineer = FeatureEngineer()
        self.text_preprocessor = TextPreprocessor()
        self.data = self.load_data()
        self.preprocess_data()

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
        self.data['excerpt'] = self.data['excerpt'].apply(self.preprocess_text)
        label_encoder = LabelEncoder()
        self.data['label'] = label_encoder.fit_transform(self.data['label'])

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
