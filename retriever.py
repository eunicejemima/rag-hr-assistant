import os
import pickle
import faiss
import numpy as np

from embedder import Embedder


VECTOR_FOLDER = "vector_store"

INDEX_FILE = os.path.join(
    VECTOR_FOLDER,
    "index.faiss"
)

CHUNKS_FILE = os.path.join(
    VECTOR_FOLDER,
    "chunks.pkl"
)


class Retriever:

    def __init__(self):

        self.embedder = Embedder()

        self.index = None

        self.chunks = []


    def create_vector_store(self, chunks):

        os.makedirs(VECTOR_FOLDER, exist_ok=True)

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        print("Creating embeddings...")

        embeddings = self.embedder.create_embeddings(texts)

        embeddings = np.array(
            embeddings
        ).astype("float32")

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(
            dimension
        )

        self.index.add(embeddings)

        self.chunks = chunks

        faiss.write_index(
            self.index,
            INDEX_FILE
        )

        with open(
            CHUNKS_FILE,
            "wb"
        ) as file:

            pickle.dump(
                self.chunks,
                file
            )

        print("Vector store created successfully!")


    def load_vector_store(self):

        if not os.path.exists(INDEX_FILE):

            raise FileNotFoundError(
                "Vector store does not exist. "
                "Run the document processing first."
            )

        self.index = faiss.read_index(
            INDEX_FILE
        )

        with open(
            CHUNKS_FILE,
            "rb"
        ) as file:

            self.chunks = pickle.load(file)

        print("Vector store loaded.")


    def search(self, query, top_k=3):

        query_embedding = self.embedder.create_embeddings(
            [query]
        )

        query_embedding = np.array(
            query_embedding
        ).astype("float32")

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for index in indices[0]:

            if index < len(self.chunks):

                results.append(
                    self.chunks[index]
                )

        return results