import os
import openai
import traceback
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from neurocraft.backend.text_simplification.simplification_model import TextSimplificationModel
from neurocraft.utils import extract_text_from_pdf, extract_text_from_website, chunk_text
#from neurocraft.backend.dyslexia_classifier.classification_model import NLPModel
from neurocraft.backend.dyslexia_classifier.mock_model import MockModel

app = FastAPI()
#app.state.model

# Create instances of the classification and simplification models
mock_model = MockModel()
openai.api_key = os.getenv("API_KEY")
simplification_model = TextSimplificationModel()


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

'''
# Endpoint for dyslexia classification
@app.get("/classify-dyslexia")
async def classify_dyslexia(
    text: str = None,
    file: UploadFile = None,
    url: str = None
):
    try:
        # Check if the input is text, file, or URL
        if text:
            content = text
        elif file:
            content = await file.read()
        elif url:
            content = extract_text_from_website(url)
        else:
            raise HTTPException(status_code=400, detail="Invalid input. Provide text, file, or URL.")

        # If the content is a PDF file, extract text using your function
        if file and file.filename.endswith('.pdf'):
            content = extract_text_from_pdf(content)

        # Split the content into chunks
        chunks = chunk_text(content)

        # Initialize an empty list to store classification results for each chunk
        results = []

        # Iterate through each chunk and classify
        for chunk in chunks:
            prediction = mock_model.predict(chunk)
            results.append(prediction)

        # Return the aggregated results
        return {"classification_results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint for text simplification
@app.get("/simplify-text")
async def simplify_text(
    text: str = None,
    file: UploadFile = None,
    url: str = None
):
    """
    This endpoint takes text, a file, or a URL as input and returns a simplified version of the text.
    The simplification process is designed to make the text easier to read for dyslexic readers.

    Parameters:
    text (str): The text to be simplified.
    file (UploadFile): The file to be simplified (if provided).
    url (str): The URL to extract text from and simplify (if provided).

    Returns:
    dict: A dictionary with a single key-value pair. The key is 'simplified_text' and the value is the simplified text (str).
    """
    try:
        # Check if the input is text, file, or URL
        if text:
            content = text
        elif file:
            content = await file.read()
        elif url:
            content = extract_text_from_website(url)
        else:
            raise HTTPException(status_code=400, detail="Invalid input. Provide text, file, or URL.")

        # If the content is a PDF file, extract text using your function
        if isinstance(content, bytes) and file.filename.endswith('.pdf'):
            content = extract_text_from_pdf(content)

        # Simplify the text using the simplification model
        simplified_text = simplification_model.simplify_text(content)

        return {"simplified_text": simplified_text}
    except Exception as e:
        # Handle exceptions, e.g., model not loaded or input validation error
        raise HTTPException(status_code=500, detail=str(e))

'''
# Endpoint for dyslexia classification with text simplification
@app.get("/classify-simplify-dyslexia")
async def classify_simplify_dyslexia(
    text: str = None,
    file: UploadFile = None,
    url: str = None
):
    try:
        # Check if the input is text, file, or URL
        if text:
            content = text
        elif file:
            content = await file.read()
        elif url:
            content = extract_text_from_website(url)
        else:
            raise HTTPException(status_code=400, detail="Invalid input. Provide text, file, or URL.")

        # If the content is a PDF file, extract text using your function
        if file and file.filename.endswith('.pdf'):
            content = extract_text_from_pdf(content)

        # Split the content into chunks
        chunks = chunk_text(content)

        # Initialize an empty list to store results for each chunk
        simplified_texts = []

        for chunk in chunks:
            try:
                simplified_text = simplification_model.simplify_text(chunk)
                simplified_texts.append(simplified_text)
            except Exception as e:
                # Handle exceptions for individual chunks
                error_message = f"Error simplifying chunk: {chunk}. Exception details: {str(e)}"
                print(error_message)
                simplified_texts.append({"error": error_message})

        return {"simplified_texts": simplified_texts}

    except Exception as e:
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
