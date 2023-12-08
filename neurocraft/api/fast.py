from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware #do we include middleware? ask TA
from neurocraft.utils import chunk_text, extract_text_from_pdf, extract_text_from_website
from neurocraft.backend.dyslexia_classifier.NLP_models.combined_model import CombinedModel
from neurocraft.backend.dyslexia_classifier.data import DyslexiaData

app = FastAPI()

# Middleware in FastAPI is code that runs befre processing the request and after processing the response
# CORS: Cross-Origin Resource Sharing
# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

dyslexia_data = DyslexiaData("raw_data/CLEAR Corpus 6.01 - CLEAR Corpus 6.01.csv")
dyslexia_data.load_data()
dyslexia_data.preprocess_data()
X_train_num, X_test_num, X_train_text, X_test_text, y_train, y_test = dyslexia_data.split_data()
X_train_pad, X_test_pad = dyslexia_data.embed_and_pad_data(X_train_text, X_test_text)

# Create the CombinedModel instance with the loaded or preprocessed data
input_dim = X_train_pad.shape[1]
labels_dict = {'very hard': 0, 'hard': 1, 'moderately hard': 2, 'acceptable': 3, 'easy': 4, 'very easy': 5}
model = CombinedModel(X_train_pad, input_dim, labels_dict)

# Endpoint for dyslexia classification
@app.post("/classify-dyslexia")
def classify_dyslexia(text: str):
    """
    Endpoint to classify text for dyslexia.

    Parameters:
    - text: The text to be classified.

    Returns:
    - result: The classification result.
    """
    try:
        if text.endswith('.pdf'):
            text = extract_text_from_pdf(text)
        elif text.startswith('http'):
            text = extract_text_from_website(text)
        chunks = chunk_text(text)
        results = []
        for chunk in chunks:
            # Preprocess your text and extract features
            dyslexia_data = DyslexiaData(chunk)
            X_test_text = dyslexia_data.preprocess_text(chunk)
            prediction = model.predict(X_test_text)
            results.append(prediction)
    except Exception as e:
        # Handle exceptions, e.g., model not loaded or input validation error
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for text simplification
@app.post("/simplify-text")
def simplify_text_endpoint(text: str):
    """
    This endpoint takes a string of text as input and returns a simplified version of the text.
    The simplification process is designed to make the text easier to read for dyslexic readers.

    Parameters:
    text (str): The text to be simplified. This should be a string of text.

    Returns:
    dict: A dictionary with a single key-value pair. The key is 'result' and the value is the simplified text (str).
    """
    try:
        simplified_text = simplify_text(text)
        return {"result": simplified_text}
    except Exception as e:
        # Handle exceptions, e.g., model not loaded or input validation error
        raise HTTPException(status_code=500, detail=str(e))

# Default endpoint
@app.get("/")
def root():
    """
    Default endpoint. Welcome message for the Dyslexia Classification API.

    Returns:
    - message: A welcome message.
    """
    return {"message": "Welcome to the Dyslexia Classification API by neuroCraft!"}
