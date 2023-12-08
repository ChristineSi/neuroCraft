from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import nltk
import PyPDF2
import requests
from bs4 import BeautifulSoup

def plot_confusion_matrix(y_true, y_pred, labels_dict):
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

def chunk_text(excerpt, max_chunk_size=200):
    """
    Split a long excerpt into chunks of a specified size.

    Args:
    - excerpt (str): Input excerpt.
    - max_chunk_size (int): Maximum number of words in each chunk.

    Returns:
    - list: List of excerpt chunks.
    """
    words = nltk.word_tokenize(excerpt)
    chunks = [' '.join(words[i:i+max_chunk_size]) for i in range(0, len(words), max_chunk_size)]
    return chunks

def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file.

    Args:
    - file_path (str): Path to the PDF file.

    Returns:
    - str: Text extracted from the PDF.
    """
    pdf_file_obj = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfFileReader(pdf_file_obj)
    text = ''
    for page_num in range(pdf_reader.numPages):
        page_obj = pdf_reader.getPage(page_num)
        text += page_obj.extractText()
    pdf_file_obj.close()
    return text

def extract_text_from_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
    return text

#def compute_precision(y_true, y_pred):
    #return precision_score(y_true, y_pred)

#def compute_recall(y_true, y_pred):
    #return recall_score(y_true, y_pred)

#def load_data_from_csv(file_path):
    #return pd.read_csv(file_path)

#def save_data_to_csv(df, file_path):
    #df.to_csv(file_path, index=False)
