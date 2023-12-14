# neuroCraft
**LeWagon** - Data Science Project.
<br>
<br>
Simplifying the lifes of neurodivergent learners.

## Overview
NeuroCraft aims to create a model to classify texts regarding their easiness for dyslexia readers. After testing different models, we decied to use a [RoBEETa model](https://huggingface.co/docs/transformers/model_doc/roberta). The model is trained on the [CLEAR (CommonLit Ease of Readability) Corpus](https://docs.google.com/spreadsheets/d/1sfsZhhP2umXXtmEP_NRErxLuwgN98TyH7LWOq3j07O0/edit?ref=commonlit.org), an open dataset of 5,000 excerpts for students from grades 3 to 12 in English Language Arts classrooms. The texts will be simplified through a Language Model (LLM).

Would you like to know more? Check our platform [here](https://neurocraft.streamlit.app/)

## Usage

### API
#### PDF Classification
- **Endpoint:** /pdf-prediction
- **Method:** POST
- **Description:** Upload a PDF file for text readibility classification.
- **Usage:**
  ```console
  curl -X POST -F "file=@your_pdf.pdf" http://localhost:8000/pdf-prediction
  ```
- **Response:** Returns the average dyslexia prediction for the provided PDF.

#### PDF Simplification
- **Endpoint:** /pdf-simplification
- **Method:** POST
- **Description:** Upload a PDF file for text simplification.
- **Usage:**
  ```console
  curl -X POST -F "file=@your_pdf.pdf" http://localhost:8000/pdf-simplification
  ```
- **Response:** Returns the simplified text for the provided PDF.

#### Text Classification
- **Endpoint:** /average-prediction
- **Method:** POST
- **Description:**
- **Usage:**
  ```console
  curl -X POST -F "file=@your_pdf.pdf" http://localhost:8000/average-prediction
  ```
- **Response:** Returns the average dyslexia prediction for the provided text.

#### Text Simplification
- **Endpoint:** /pdf-text-simplification
- **Method:** POST
- **Description:** Simplify text input.
- **Usage:**
  ```console
  curl -X POST -F "file=@your_pdf.pdf" http://localhost:8000/text-simplification
  ```
- **Response:** Returns the simplified text for the provided text.

### Setup instructions 🚀
Follow these steps to set up the NeuroCraft project locally:

1. Clone the repository:
  ```console
  git clone https://github.com/ChristineSi/neuroCraft.git
  cd neurocraft
  ```

2. Install dependencies:
  ```console
  pip install -r requirements.txtç
  ```

3. Set up your environment variables:
- Create a `.env` file in the project root.
`MODEL_TARGET=local
GCP_PROJECT=le-wagon-bootcamp-402214
GCP_REGION=europe-west1
BUCKET_NAME=neurocraft_classification
RAW_DATA_LOCATION=raw_data
MODELS_LOCATION=models
GAR_IMAGE=neurocraft
GAR_MEMORY=4Gi`

- Add your OpenAI API key:
`OPENAI_API_KEY=your_api_key_here`

4. Run the FastAPI server:
`uvicorn neurocraft.api.fast:app --reload`


### Next Steps


### Limitations


### About
NeuroCraft was co-created by [Andrea Calcagni](https://github.com/AndreaCalcagni), [Pei-Yu Chen](https://github.com/renee1j), [Patricia Moreno Gaona](https://github.com/patmg-coder) and [Christine Sigrist](https://github.com/ChristineSi) as the final project for the Data Science bootcamp at [Le Wagon](https://www.lewagon.com/) (batch 1451) on December 2023. The code is written in Python, hosted and published with [Streamlit](https://streamlit.io/).
