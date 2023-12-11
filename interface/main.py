import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from neurocraft.backend.dyslexia_classifier.data import load_data, preprocess_train_test_data, preprocess_text, embed_data, define_X_y, split_data
from neurocraft.backend.dyslexia_classifier.NLP_models.nlp_model import NLPModel

def training_flow():
    print(Fore.MAGENTA + "\n ⭐️ Use case: training_flow" + Style.RESET_ALL)

    file_path = Path(LOCAL_DATA_PATH).joinpath("raw", "CLEAR Corpus 6.01 - CLEAR Corpus 6.01.csv")

    df = load_data(file_path)

    df = preprocess_train_test_data(df)

    X, y = define_X_y(df)
    X_train_text, X_test_text, y_train, y_test = split_data(X, y)

    X_train_text = preprocess_text(X_train_text)
    X_test_text = preprocess_text(X_test_text)

    X_train_text = embed_data(X_train_text)
    X_test_text = embed_data(X_test_text)

    model = NLPModel(X_train_text, y_train)
    model = model.build_model()
    model, history = model.train()

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)

    print("✅ preprocess() done \n")
