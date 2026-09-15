from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:

    def __init__(self):

        print("Loading embedding model...")

        self.model = SentenceTransformer(MODEL_NAME)

    def create_embeddings(self, texts):

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True
        )

        return embeddings