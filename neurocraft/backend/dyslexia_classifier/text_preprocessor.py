import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import text_to_word_sequence
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
# for other files, use -> from text_preprocessor import TextPreprocessor
#                         tp = TextPreprocessor()
#                         preprocessed_text = tp.preprocess("This is a sample text.")

class TextPreprocessor:
    def basic_cleaning(self, sentence):
        sentence = sentence.lower()
        sentence = ''.join(char for char in sentence if not char.isdigit())
        # Adding special quotes to the regular expression pattern
        special_quotes = '“”‘’'
        pattern = '[' + re.escape(string.punctuation + special_quotes) + ']'
        sentence = re.sub(pattern, '', sentence)
        sentence = sentence.strip()
        return sentence

    def preprocess(self, text):
        text = self.basic_cleaning(text)
        return text


class EmbeddingCreator:
    def __init__(self, model_name="roberta-base", max_length=190):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
        self.model = TFAutoModel.from_pretrained(model_name, from_pt=True)
        self.max_length = max_length

    def create_embeddings(self, excerpts):
        tokenized_excerpts = self.tokenizer(excerpts, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="tf")
        embeddings = self.model.predict(tokenized_excerpts["input_ids"])
        return embeddings.last_hidden_state
