import numpy as np
import pandas as pd
import string
from textstat.textstat import textstat
from nltk.tokenize import sent_tokenize
from lexicalrichness import LexicalRichness
#for other files, use -> from feature_engineering import FeatureEngineer

class FeatureEngineer:
    def __init__(self):
        self.df = pd.DataFrame()

    def avg_word_count(self, row):
        return row[['Joon\nWC v1', 'Google\nWC']].mean()

    def count_unique_words(self, row):
        words = row['Excerpt'].split()
        return len(set(words))

    def avg_word_length(self, x):
        punctuation = string.punctuation
        for punc in punctuation:
            x = x.replace(punc,' ')
        words = x.split(' ')
        words = [word for word in words if len(word) > 0]
        word_len = [len(word) for word in words]
        return np.mean(word_len)

    def calculate_average_sentence_count(self, row):
        sentence_count_columns = ['Sentence\nCount v1', 'Sentence\nCount v2']
        return row[sentence_count_columns].mean()

    def avg_syllaba_word(self, x):
        syll_count = [textstat.syllable_count(word) for word in x.split()]
        return np.mean(syll_count)

    def count_vowels(self, word):
        vowels = set("AEIOUaeiou")
        return sum(1 for char in word if char in vowels)

    def count_punctuation(self, x):
        punc = string.punctuation
        count=0
        for char in x:
            if char in punc:
                count+=1
        return count/ len(x)

    def count_characters_per_sentence(self, excerpt):
        sentences = sent_tokenize(excerpt)
        char_counts_per_sentence = [len(sentence) for sentence in sentences]
        return char_counts_per_sentence

    def avg_chars_per_sentence(self, row):
        total_chars = sum(row['Characters Per Sentence'])
        total_sentences = row['Average Sentence Count']
        return total_chars / total_sentences if total_sentences > 0 else 0

    def ratio_capital_chars(self, excerpt):
        count = 0
        for char in excerpt:
            if char.isupper():
                count += 1
        return count/len(excerpt)

    def count_capital_words(self, excerpt):
        return sum(1 for word in excerpt.split() if word.isupper() and word.isalpha())

    def word_length_std(self, excerpt):
        word_lengths = [len(word) for word in excerpt.split()]
        return np.std(word_lengths) if len(word_lengths) > 0 else 0

    def sentence_length_variation(self, row):
        sentence_lengths = self.count_characters_per_sentence(row['Excerpt'])
        return np.std(sentence_lengths)

    def sentence_len(self, x):
        x = x.replace("?",'.').replace("!",'.')
        sentences = x.split('.')
        sentences = [sentence for sentence in sentences if len(sentence) > 0]
        sent_len = [len(sentence.split()) for sentence in sentences]
        return np.mean(sent_len)

    def add_lexical_richness_features(self, df):
        cttr_list, msttr_list, mtld_list, yule_k_list = [], [], [], []
        for text in df['Excerpt']:
            lex = LexicalRichness(text)
            cttr_list.append(lex.cttr)
            msttr_list.append(lex.msttr(segment_window=25))  # Adjust the segment_window as needed
            mtld_list.append(lex.mtld(threshold=0.72))  # Adjust the threshold as needed
            yule_k_list.append(lex.yulek)
        df['cttr'] = cttr_list
        df['msttr'] = msttr_list
        df['mtld'] = mtld_list
        df['yule_k'] = yule_k_list
        return df
