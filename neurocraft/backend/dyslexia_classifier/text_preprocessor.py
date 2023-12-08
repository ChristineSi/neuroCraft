import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import text_to_word_sequence
# for other files, use -> from text_preprocessor import TextPreprocessor
#                         tp = TextPreprocessor()
#                         preprocessed_text = tp.preprocess("This is a sample text.")

class TextPreprocessor:
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def basic_cleaning(self, sentence):
        sentence = sentence.lower()
        sentence = ''.join(char for char in sentence if not char.isdigit())
        # Adding special quotes to the regular expression pattern
        special_quotes = '“”‘’'
        pattern = '[' + re.escape(string.punctuation + special_quotes) + ']'
        sentence = re.sub(pattern, '', sentence)
        sentence = sentence.strip()
        return sentence

    def tokenize(self, sentence):
        return text_to_word_sequence(sentence)

    def lemmatize(self, sentence):
        verbs_lemmatized = [self.lemmatizer.lemmatize(word, pos='v') for word in sentence]
        nouns_lemmatized = [self.lemmatizer.lemmatize(word, pos='n') for word in verbs_lemmatized]
        adverbs_lemmatized = [self.lemmatizer.lemmatize(word, pos='r') for word in nouns_lemmatized]
        adj_lemmatized = [self.lemmatizer.lemmatize(word, pos='a') for word in adverbs_lemmatized]
        sat_lemmatized = [self.lemmatizer.lemmatize(word, pos='a') for word in adj_lemmatized]
        return sat_lemmatized

#    def lemmatizer(sentence):
        wnl = WordNetLemmatizer()
        verbs_lemmatized = []
        for word in sentence:
            verbs_lemmatized.append(wnl.lemmatize(word, pos = 'v'))
        nouns_lemmatized = []
        for word in verbs_lemmatized:
            nouns_lemmatized.append(wnl.lemmatize(word, pos = 'n'))
        adverbs_lemmatized = []
        for word in nouns_lemmatized:
            adverbs_lemmatized.append(wnl.lemmatize(word, pos = 'r'))
        adj_lemmatized = []
        for word in adverbs_lemmatized:
            adj_lemmatized.append(wnl.lemmatize(word, pos = 'a'))
        sat_lemmatized = []
        for word in adj_lemmatized:
            sat_lemmatized.append(wnl.lemmatize(word, pos = 'a'))
        return sat_lemmatized

    def remove_stopwords(self, sentence):
        return [word for word in sentence if word not in self.stopwords]

    def preprocess(self, text):
        text = self.basic_cleaning(text)
        text = self.tokenize(text)
        text = self.lemmatize(text)
        text = self.remove_stopwords(text)
        return text
