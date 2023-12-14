import os
import openai
import traceback
from fastapi import FastAPI, UploadFile, HTTPException
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from neurocraft.backend.text_simplification.simplification_model import TextSimplificationModel
from neurocraft.utils import extract_text_from_pdf, extract_text_from_website, chunk_text
from neurocraft.interface.main import pred
from dotenv import load_dotenv

app = FastAPI()
#app.state.model



load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# openai.api_key = os.getenv("API_KEY")
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
@app.get("/prediction-only")
def prediction_only(text: str):
    try:
        # Chunk the text before making predictions
        text_chunks = chunk_text(text)

        # Classify each chunk based on the predictions
        predictions = []
        for i, chunk in enumerate(text_chunks):
            # Classify each chunk using your pred function
            prediction = pred(X_pred=chunk)
            predictions.append(int(prediction))

        # Calculate the average prediction
        average_prediction = int(sum(predictions) / len(predictions))

        # Return a dictionary with average prediction and predictions for each chunk
        return {
            "average_prediction": average_prediction,
            "predictions": predictions,
        }

    except Exception as e:
        # Handle exceptions, e.g., model not loaded or input validation error
        raise HTTPException(status_code=500, detail=str(e))
'''
#http://localhost:8000/average-prediction
@app.post("/average-prediction")
def average_prediction(text: str):
    try:
        # Chunk the text before making predictions
        text_chunks = chunk_text(text)
        # Classify each chunk based on the predictions
        predictions = []
        for i, chunk in enumerate(text_chunks):
            # Classify each chunk using your pred function
            prediction = pred(X_pred=chunk)
            predictions.append(int(prediction))
        # Calculate the average prediction
        average_prediction = int(sum(predictions) / len(predictions))
        # Return a dictionary with only the average prediction
        return average_prediction
    except Exception as e:
        # Handle exceptions, e.g., model not loaded or input validation error
        raise HTTPException(status_code=500, detail=str(e))

#@app.post("/average-prediction")
#def average_prediction(text: str):
#    try:
#        # Chunk the text before making predictions
#        text_chunks = chunk_text(text)
#
#        # Classify each chunk based on the predictions
#        #predictions = []
#        #for i, chunk in enumerate(text_chunks):
#        #    # Classify each chunk using your pred function
#        #    prediction = pred(X_pred=chunk)
#        #    predictions.append(int(prediction))
##
#        # Calculate the average prediction
#        # average_prediction = int(sum(predictions) / len(predictions))
#
#        # Return a dictionary with only the average prediction
#        return {"classification": 1}

    except Exception as e:
        # Handle exceptions, e.g., model not loaded or input validation error
        raise HTTPException(status_code=500, detail=str(e))

# http://localhost:8000/simplified-text
@app.post("/simplified-text")
def simplified_text(text: str):
    try:
        # Chunk the text before simplifying
        text_chunks = chunk_text(text)

        # Simplify each chunk using your simplification_model
        simplified_texts = []
        for chunk in text_chunks:
            simplified_chunk = simplification_model.simplify_text(chunk)
            simplified_texts.append(simplified_chunk)

        # Return only the simplified texts
        return simplified_texts

    except Exception as e:
        # Handle exceptions, e.g., model not loaded or input validation error
        raise HTTPException(status_code=500, detail=str(e))

# http://localhost:8000/classify-simplify
@app.post("/classify-simplify")
def classify_simplify(text: str):
    try:
        # Chunk the text before making predictions
        text_chunks = chunk_text(text)

        # Classify and simplify each chunk based on the predictions
        predictions = []
        simplified_texts = []
        for i, chunk in enumerate(text_chunks):
            # Classify each chunk using your pred function
            prediction = pred(X_pred=chunk)
            predictions.append(int(prediction))

        simplified_texts = simplification_model.simplify_text(text)

        # Calculate the average prediction
        average_prediction = int(sum(predictions) / len(predictions))

        # Return a dictionary with only the average prediction and simplified texts
        return average_prediction, simplified_texts

    except Exception as e:
        # Handle exceptions, e.g., model not loaded or input validation error
        raise HTTPException(status_code=500, detail=str(e))
'''
@app.get("/all-text")
def all_text(text: str):
    try:
        # Chunk the text before making predictions
        text_chunks = chunk_text(text)

        # Classify and simplify each chunk based on the predictions
        predictions = []
        simplified_texts = []
        simplified_text_predictions = []  # Store predictions for simplified texts
        for i, chunk in enumerate(text_chunks):
            # Classify each chunk using your pred function
            prediction = pred(X_pred=chunk)
            predictions.append(int(prediction))

            # Simplify each chunk using your simplification_model
            simplified_chunk = simplification_model.simplify_text(chunk)
            simplified_texts.append(simplified_chunk)

            # Predict the difficulty of the simplified text
            simplified_text_prediction = pred(X_pred=simplified_chunk)
            simplified_text_predictions.append(int(simplified_text_prediction))

        # Calculate the average prediction
        average_prediction = int(sum(predictions) / len(predictions))

        # Calculate the average prediction for the simplified texts
        average_simplified_text_prediction = int(sum(simplified_text_predictions) / len(simplified_text_predictions))

        # Return a dictionary with average predictions and simplified texts for each chunk
        return {
            "average_prediction": average_prediction,
            "average_simplified_text_prediction": average_simplified_text_prediction,
            "predictions": predictions,
            "simplified_texts": simplified_texts,
            "simplified_text_predictions": simplified_text_predictions
        }

    except Exception as e:
        # Handle exceptions, e.g., model not loaded or input validation error
        raise HTTPException(status_code=500, detail=str(e))

# http://localhost:8000/simplify-chunk
@app.get("/simplify-chunk")
def simplify_chunk(text: str):
    try:
        # Chunk the text before making predictions
        text_chunks = chunk_text(text)

        # Classify each chunk based on the predictions
        predictions = []
        for i, chunk in enumerate(text_chunks):
            # Classify each chunk using your pred function
            prediction = pred(X_pred=chunk)
            predictions.append(int(prediction))

        # Calculate the average prediction
        average_prediction = int(sum(predictions) / len(predictions))

        # Simplify the entire text
        simplified_text = simplification_model.simplify_text(text)

        # Chunk the simplified text before making predictions
        simplified_text_chunks = chunk_text(simplified_text)

        # Classify each chunk of the simplified text
        simplified_predictions = []
        for i, chunk in enumerate(simplified_text_chunks):
            # Classify each chunk using your pred function
            simplified_prediction = pred(X_pred=chunk)
            simplified_predictions.append(int(simplified_prediction))

        # Calculate the average prediction for the simplified text
        average_simplified_prediction = int(sum(simplified_predictions) / len(simplified_predictions))

        # Return a dictionary with only the average prediction and simplified text
        return predictions, average_prediction, simplified_text, simplified_predictions, average_simplified_prediction

    except Exception as e:
        # Handle exceptions, e.g., model not loaded or input validation error
        raise HTTPException(status_code=500, detail=str(e))
'''
#The following endpoint is for the neuroCraft demo
#1. chunk the text -> do predictions and average of predictions -> simplify the chunked text -> do predictions on the chunked_simplified_texts and average of these predictions
@app.get("/neurocraft")
def neurocraft(text: str):
    try:
        # Chunk the text before making predictions
        text_chunks = chunk_text(text)
        # Classify each chunk based on the predictions
        predictions = []
        for i, chunk in enumerate(text_chunks):
            # Classify each chunk using your pred function
            prediction = pred(X_pred=chunk)
            predictions.append(int(prediction))
        # Calculate the average prediction
        average_prediction = int(sum(predictions) / len(predictions))

        # Simplify the entire text
        simplified_text = simplification_model.simplify_text(text)

        # Chunk the simplified text before making predictions
        simplified_text_chunks = chunk_text(simplified_text)

        # Classify each chunk of the simplified text
        simplified_predictions = []
        for i, chunk in enumerate(simplified_text_chunks):
            # Classify each chunk using your pred function
            simplified_prediction = pred(X_pred=chunk)
            simplified_predictions.append(int(simplified_prediction))

        # Calculate the average prediction for the simplified text
        average_simplified_prediction = int(sum(simplified_predictions) / len(simplified_predictions))

        # Return a dictionary with the required information
        return {
            "predictions": predictions,
            "average_prediction": average_prediction,
            "simplified_text": simplified_text,
            "simplified_text_predictions": simplified_predictions,
            "average_simplified_prediction": average_simplified_prediction,
        }

    except Exception as e:
        # Handle exceptions, e.g., model not loaded or input validation error
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/neurocraft")
def neurocraft(text: str):
    try:
        # Chunk the text before making predictions
        text_chunks = chunk_text(text)
        # Classify each chunk based on the predictions
        predictions = []
        for i, chunk in enumerate(text_chunks):
            # Classify each chunk using your pred function
            prediction = pred(X_pred=chunk)
            predictions.append(int(prediction))
        # Calculate the average prediction
        average_prediction = int(sum(predictions) / len(predictions))

        # Simplify the entire text
        simplified_text = simplification_model.simplify_text(text)

        # Chunk the simplified text before making predictions
        simplified_text_chunks = chunk_text(simplified_text)

        # Classify each chunk of the simplified text
        simplified_predictions = []
        for i, chunk in enumerate(simplified_text_chunks):
            # Classify each chunk using your pred function
            simplified_prediction = pred(X_pred=chunk)
            simplified_predictions.append(int(simplified_prediction))

        # Calculate the average prediction for the simplified text
        average_simplified_prediction = int(sum(simplified_predictions) / len(simplified_predictions))

        # Return a dictionary with the required information
        return {
            "predictions": predictions,
            "average_prediction": average_prediction,
            "simplified_text": simplified_text,
            "simplified_text_predictions": simplified_predictions,
            "average_simplified_prediction": average_simplified_prediction,
        }

    except Exception as e:
        # Handle exceptions, e.g., model not loaded or input validation error
        raise HTTPException(status_code=500, detail=str(e))

'''
#2. chunk the text -> do predictions and average of predictions -> simplify the original text -> chunk the simplified text -> do predictions of chunked_simplified_texts and average of these predictions
@app.get("/scenario-2")
def scenario_2(text: str):
    try:
        # Chunk the text before making predictions
        text_chunks = chunk_text(text)

        # Classify and simplify each chunk based on the predictions
        predictions = []
        simplified_texts = []
        simplified_predictions = []
        for i, chunk in enumerate(text_chunks):
            # Classify each chunk using your pred function
            prediction = pred(X_pred=chunk)
            predictions.append(int(prediction))

            # Simplify each chunk using your simplification_model
            simplified_chunk = simplification_model.simplify_text(chunk)
            simplified_texts.append(simplified_chunk)

            # Predict the difficulty of the simplified text
            simplified_prediction = pred(X_pred=simplified_chunk)
            simplified_predictions.append(int(simplified_prediction))

        # Calculate the average prediction
        average_prediction = int(sum(predictions) / len(predictions))

        # Calculate the average prediction for the simplified texts
        average_simplified_prediction = int(sum(simplified_predictions) / len(simplified_predictions))

        # Return a dictionary with the required information
        return {
            "predictions": predictions,
            "average_prediction": average_prediction,
            "simplified_texts": simplified_texts,
            "simplified_text_predictions": simplified_predictions,
            "average_simplified_prediction": average_simplified_prediction
        }

    except Exception as e:
        # Handle exceptions, e.g., model not loaded or input validation error
        raise HTTPException(status_code=500, detail=str(e))

'''
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
            pdf_text = extract_text_from_pdf(content)
            if pdf_text:
                content = pdf_text
            else:
                raise HTTPException(status_code=500, detail="Error extracting text from PDF")

        print(f"Original Text: {content}")

        # Split the content into chunks using chunk_text
        chunks = chunk_text(content)
        print(f"Number of Chunks: {len(chunks)}")

        # Initialize an empty list to store classification results for each chunk
        results = []

        # Iterate through each chunk and classify
        for i, chunk in enumerate(chunks):
            try:
                # Perform classification or other processing on each chunk
                # For now, let's just append the chunk to results
                results.append({"chunk_index": i, "chunk": chunk})
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error processing chunk {i}: {chunk}. Error details: {str(e)}")

        # Return the aggregated results
        return {"classification_results": results}
    except HTTPException as e:
        raise e  # Let FastAPI handle HTTP exceptions with proper responses
    except Exception as e:
        # Print the full traceback of the exception
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simplify-text")
async def simplify_text(
    text: Optional[str] = None,
    file: UploadFile = None,
    url: Optional[str] = None
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

        # Pass the extracted text through the chunk_text function
        text_chunks = chunk_text(content)

        # Check if text_chunks is None or empty
        if text_chunks is None or not text_chunks:
            raise HTTPException(status_code=400, detail="Unable to extract text chunks from the provided content.")

        # Classify each text chunk using the classification model
        classification_results = [pred(X_pred=chunk) for chunk in text_chunks if chunk]

        # Check if classification_results is None or empty
        if classification_results is None or not classification_results:
            raise HTTPException(status_code=400, detail="Unable to classify text chunks.")

        # Combine the classified chunks into a single string
        simplified_text = " ".join(classification_results)

        return {"simplified_text": simplified_text}
    except Exception as e:
        # Handle exceptions, e.g., model not loaded or input validation error
        raise HTTPException(status_code=500, detail=str(e))


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
        results = []

        for chunk in chunks:
            try:
                dyslexia_prediction = pred(model, chunk)  # Use the dyslexia classification function with the loaded model
                simplified_text = simplification_model.simplify_text(chunk)

                print(f"Chunk: {chunk}")
                print(f"Dyslexia Prediction: {dyslexia_prediction}")
                print(f"Simplified Text: {simplified_text}")

                results.append({
                    "chunk": chunk,
                    "dyslexia_classification": dyslexia_prediction,
                    "simplified_text": simplified_text
                })
            except Exception as e:
                print(f"Error processing chunk: {chunk}")
                print(f"Exception details: {traceback.format_exc()}")

        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

'''





# Default endpoint
@app.get("/")
def root():
    """
    Default endpoint. Welcome message for the Dyslexia Classification API.

    Returns:
    - message: A welcome message.
    """
    return {"message": "Welcome to the Dyslexia Classification API by neuroCraft!"}
