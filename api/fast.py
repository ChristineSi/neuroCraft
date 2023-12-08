from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware #do we include middleware? ask TA
#from your_module import classify_text, simplify_text  # Import your functions

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
        result = classify_text(text)
        return {"result": result}
    except Exception as e:
        # Handle exceptions, e.g., model not loaded or input validation error
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for text simplification
@app.post("/simplify-text")
def simplify_text_endpoint(text: str):
    """
    Endpoint to simplify text for dyslexia readers.

    Parameters:
    - text: The text to be simplified.

    Returns:
    - result: The simplified text.
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
