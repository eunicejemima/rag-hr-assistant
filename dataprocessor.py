from pdfreader import read_pdf
from chunker import chunk_pages
from embedder import embed_chunks
from openai import OpenAI
from dotenv import load_dotenv
from typing import List

pdf_path = "./resources/HRPolicy.pdf"
def run():
    # Read HR Policy PDF and extract text
    pages = read_pdf(pdf_path)

    # Chunk the extracted text into manageable pieces
    chunks = chunk_pages(pages, chunk_size=900, chunk_overlap=150)

    embedded_chunks = embed_chunks(chunks)
    print(f"Embedded {len(embedded_chunks)} chunks from the PDF.")
    print(f"First chunk: {embedded_chunks[0][:10]}")  # Print first 10 dimensions of the first embedding
   
    
if __name__ == "__main__":
    run()