from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware #do we include middleware? ask TA
from backend.dyslexia_classifier.utils import chunk_text, extract_text_from_pdf, extract_text_from_website
from backend.dyslexia_classifier.models.combined_model import CombinedModel
from backend.dyslexia_classifier.data import DyslexiaData

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

model = CombinedModel(input_shape_text, input_dim_num, labels_dict)

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
