import os
from pypdf import PdfReader


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "data")


def load_pdfs():
    documents = []

    for filename in os.listdir(DATA_FOLDER):

        if filename.lower().endswith(".pdf"):

            filepath = os.path.join(DATA_FOLDER, filename)

            print(f"Reading: {filename}")

            reader = PdfReader(filepath)

            text = ""

            for page in reader.pages:
                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

            documents.append({
                "source": filename,
                "text": text
            })

    return documents


def create_chunks(text, chunk_size=500, overlap=50):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def process_documents():

    documents = load_pdfs()

    all_chunks = []

    for document in documents:

        chunks = create_chunks(document["text"])

        for chunk in chunks:

            all_chunks.append({
                "source": document["source"],
                "text": chunk
            })

    print(f"\nTotal chunks created: {len(all_chunks)}")

    return all_chunks


if __name__ == "__main__":

    chunks = process_documents()

    for chunk in chunks[:3]:
        print("\nSOURCE:", chunk["source"])
        print(chunk["text"][:300])