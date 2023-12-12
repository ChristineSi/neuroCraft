import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

# gpu is slower for this challenge
# Disable GPUs:
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from neurocraft.backend.dyslexia_classifier.data import load_data, preprocess_train_test_data, preprocess_text, embed_data, define_X_y, split_data
from neurocraft.backend.dyslexia_classifier.nlp_model import NLPModel
from neurocraft.backend.dyslexia_classifier.registry import save_model, load_model
from neurocraft.params import *

def training_flow():
    print(Fore.MAGENTA + "\n ⭐️ Use case: training_flow" + Style.RESET_ALL)

    # file_path = Path(LOCAL_DATA_PATH).joinpath("raw", "CLEAR Corpus 6.01 - CLEAR Corpus 6.01.csv")
    # file_path = os.path.join(os.path.expanduser('~'), 'code', 'AndreaCalcagni', 'neuroCraft', 'raw_data', 'CLEAR Corpus 6.01 - CLEAR Corpus 6.01.csv')
    file_path= os.path.join(os.path.dirname(os.getcwd()), 'neuroCraft', 'raw_data', 'CLEAR Corpus 6.01 - CLEAR Corpus 6.01.csv')

    df = load_data(file_path)

    df = preprocess_train_test_data(df)

    X, y = define_X_y(df)
    X_train_text, X_test_text, y_train, y_test = split_data(X, y)

    X_train_text = [preprocess_text(sentence) for sentence in X_train_text]
    X_test_text = [preprocess_text(sentence) for sentence in X_test_text]

    X_train_text = embed_data(X_train_text)
    X_test_text = embed_data(X_test_text)

    model = NLPModel(X_train_text, y_train, X_train_text.shape[1:])
    model.build_model(X_train_text.shape[1:])
    model, history = model.train()


    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)

    print("✅ training() done \n")

def pred(X_pred = None):
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    if X_pred is None:
        X_pred = text = """
        Mr. Grimes was to come up next morning to Sir John Harthover's, at the Place,
        for his old chimney-sweep was gone to prison, and the chimneys wanted sweeping.
        And so he rode away, not giving Tom time to ask what the sweep had gone to prison for,
        which was a matter of interest to Tom, as he had been in prison once or twice himself.
        Moreover, the groom looked so very neat and clean, with his drab gaiters, drab breeches,
        drab jacket, snow-white tie with a smart pin in it, and clean round ruddy face,
        that Tom was offended and disgusted at his appearance, and considered him a stuck-up fellow,
        who gave himself airs because he wore smart clothes, and other people paid for them;
        and went behind the wall to fetch the half-brick after all; but did not,
        remembering that he had come in the way of business, and was, as it were, under a flag of truce.
        """

    model = load_model()
    assert model is not None
    X_processed = preprocess_text(X_pred)
    X_embedded = embed_data(X_processed)
    y_pred = model.predict(X_embedded)
    class_labels_prediction = np.argmax(y_pred, axis=1)

    print("\n✅ prediction done: ", class_labels_prediction, "\n")
    return class_labels_prediction
