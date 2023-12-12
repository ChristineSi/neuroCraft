import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as pdf_document:
        num_pages = pdf_document.page_count

        text = ""
        for page_number in range(num_pages):
            page = pdf_document[page_number]
            text += page.get_text()

    return text

def chunk_text(text, chunk_size=200):
    # Split the text into chunks of the specified size
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Example usage
pdf_path = 'raw_data/The future of philosophy _ John R. Searle-1.pdf'
pdf_text = extract_text_from_pdf(pdf_path)
text_chunks = chunk_text(pdf_text)

print(text_chunks)
