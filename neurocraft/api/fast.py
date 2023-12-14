import os
import openai
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from neurocraft.backend.text_simplification.simplification_model import TextSimplificationModel
from neurocraft.utils import chunk_text
from pdfminer.high_level import extract_text
from neurocraft.interface.main import pred
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from fastapi.responses import StreamingResponse
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

app = FastAPI()
#app.state.model

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
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

def extract_pdf_text(file_path: str) -> str:
    try:
        with open(file_path, 'rb') as pdf_file:
            pdf_text = extract_text(pdf_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return pdf_text

def wrap_text(text, max_width, canvas, font_size):
    """
    Wrap text to fit within a specified width.

    Parameters:
    - text: The text to be wrapped.
    - max_width: The maximum width for the wrapped text.
    - canvas: The ReportLab canvas.
    - font_size: The font size.

    Returns:
    A list of lines representing the wrapped text.
    """
    lines = []
    width, height = canvas._pagesize
    text_lines = text.split('\n')

    for line in text_lines:
        split_line = simpleSplit(line, canvas._fontname, font_size, max_width)
        lines.extend(split_line)

    return lines

def set_background(canvas):
    canvas.setFillColor('#F4F4EC')  # Set background color to cream (cream-yellow #fffdd0)
    canvas.rect(0, 0, letter[0], letter[1], fill=True, stroke=False)

def create_simplified_pdf(original_text: str, simplified_text: str) -> bytes:
    # Create a BytesIO buffer to store the PDF content
    pdf_buffer = BytesIO()

    # Register dyslexia-friendly font
    dyslexic_font_path = "raw_data/OpenDyslexic-Regular.ttf"  # Update with the correct path
    pdfmetrics.registerFont(TTFont("DyslexicFont", dyslexic_font_path))

    # Create a PDF document
    pdf_canvas = canvas.Canvas(pdf_buffer, pagesize=letter)

    # Set dyslexia-friendly font and size
    dyslexia_font_size = 14
    pdf_canvas.setFont("DyslexicFont", dyslexia_font_size)

    # Set background color
    set_background(pdf_canvas)

    # Set dyslexia-friendly colors
    pdf_canvas.setFillColor(colors.black)  # Dark colored text
    pdf_canvas.setStrokeColor(colors.black)

    # Add simplified text
    current_height = 700  # Adjust as needed
    simplified_text = simplified_text.replace('\n', ' ')

    # Calculate line height based on font size
    line_height = dyslexia_font_size * 1.5  # Line spacing of 1.5

    # Add extra space before the first line
    current_height -= line_height

     # Add simplified text with improved formatting
    wrapped_simplified_text = wrap_text(simplified_text, 400, pdf_canvas, dyslexia_font_size - 2)
    for line in wrapped_simplified_text:
        pdf_canvas.drawString(100, current_height, line)
        current_height -= line_height

    # Save the PDF content
    pdf_canvas.save()

    # Reset the buffer position for reading
    pdf_buffer.seek(0)

    return pdf_buffer.read()

#FOR DEMO DAY
@app.post("/pdf-prediction")
def pdf_prediction(file: UploadFile = File(...)):
    try:
        with open(file.filename, 'wb') as f:
            while contents := file.file.read(1024 * 1024):
                f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    # Extract text from the uploaded PDF
    extracted_text = extract_pdf_text(file.filename)

    # Chunk the text before making predictions
    text_chunks = chunk_text(extracted_text)

    # Classify each chunk based on the predictions
    predictions = [pred(chunk) for chunk in text_chunks]

    # Calculate the average prediction
    average_prediction = int(sum(predictions) / len(predictions))

    return average_prediction

#FOR DEMO DAY
@app.post("/pdf-simplification")
def pdf_simplification(file: UploadFile = File(...)):
    try:
        with open(file.filename, 'wb') as f:
            while contents := file.file.read(1024 * 1024):
                f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    try:
        # Extract text from the uploaded PDF
        extracted_text = extract_pdf_text(file.filename)

        # Simplify the extracted text
        simplified_text = simplification_model.simplify_text(extracted_text)

        # Create the simplified PDF
        simplified_pdf_content = create_simplified_pdf(extracted_text, simplified_text)

        # Return the simplified PDF as a downloadable file
        return StreamingResponse(BytesIO(simplified_pdf_content), media_type="application/pdf", headers={"Content-Disposition": "attachment;filename=simplified_text.pdf"})

    except Exception as e:
        # Handle exceptions, e.g., PDF processing error or input validation error
        raise HTTPException(status_code=500, detail=str(e))


#TEXT (PDF EXTRACTION IN FRONTEND)
@app.post("/text-simplification")
def text_simplification(text: str):
    try:
        # Simplify text
        simplified_text = simplification_model.simplify_text(text)

        # Return only the simplified texts
        return simplified_text

    except Exception as e:
        # Handle exceptions, e.g., model not loaded or input validation error
        raise HTTPException(status_code=500, detail=str(e))

#PREVIOUS FOR TEXT (PDF EXTRACTION IN FRONTEND)
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

'''
#PREVIOUS FOR TEXT (PDF EXTRACTION IN FRONTEND)
#SIMPLIFIED CHUNK PER CHUNK
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

#1. chunk the text -> do predictions and average of predictions -> simplify the chunked text -> do predictions on the chunked_simplified_texts and average of these predictions
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

# Default endpoint
@app.get("/")
def root():
    """
    Default endpoint. Welcome message for the Dyslexia Classification API.

    Returns:
    - message: A welcome message.
    """
    return {"message": "Welcome to the Dyslexia Classification API by neuroCraft!"}
