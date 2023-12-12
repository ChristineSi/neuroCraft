from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import nltk
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
from neurocraft.params import *

def plot_confusion_matrix(y_true, y_pred, labels_dict):
    """
    Plot the confusion matrix.

    Args:
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels.
    - labels_dict (dict): Dictionary mapping label indices to label names.

    Returns:
    - None
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(15,15))
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt='g')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labels_dict.keys())
    ax.yaxis.set_ticklabels(labels_dict.keys())
    plt.show()

def split_data(data, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_df, test_df

def chunk_text(text, max_chunk_size=200):
    """
    Split a long excerpt into chunks of a specified size while preserving sentence boundaries.

    Args:
    - text (str): Input text.
    - max_chunk_size (int): Maximum number of words in each chunk.

    Returns:
    - list: List of text chunks.
    """
    words = text.split()
    chunks = []
    current_chunk = ""

    for word in words:
        # Check if adding the current word exceeds the max_chunk_size
        if len(current_chunk.split()) + len(word.split()) <= max_chunk_size:
            current_chunk += ' ' + word
        else:
            # Check if the last character in the current_chunk is a sentence-ending punctuation
            if current_chunk[-1] in ['.', '!', '?']:
                chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                # If not, find the last sentence-ending punctuation and create a new chunk
                last_sentence_end = max(current_chunk.rfind('.'), current_chunk.rfind('!'), current_chunk.rfind('?'))
                if last_sentence_end != -1:
                    chunks.append(current_chunk[:last_sentence_end + 1].strip())
                    current_chunk = current_chunk[last_sentence_end + 1:] + ' ' + word
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = word

    # Add the last chunk
    chunks.append(current_chunk.strip())

    return chunks

def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file.

    Args:
    - file_path (str): Path to the PDF file.

    Returns:
    - str: Text extracted from the PDF.
    """
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            text = ''
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def extract_text_from_website(url):
    """
    Extract text from a website.

    Args:
    - url (str): URL of the website.

    Returns:
    - str: Text extracted from the website.
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
        return text
    except Exception as e:
        print(f"Error extracting text from website: {e}")
        return None
