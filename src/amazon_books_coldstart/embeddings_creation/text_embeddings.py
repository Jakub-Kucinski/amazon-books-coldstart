import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_SIZE = 384
MAX_SEQ_LENGTH = 256

model = SentenceTransformer(MODEL_NAME)
model.max_seq_length = MAX_SEQ_LENGTH


def encode(text: str) -> np.ndarray:
    return model.encode(text)


def encode_list(texts: list[str]) -> np.ndarray:
    return model.encode(texts, show_progress_bar=True)


def load_index(index_path):
    index = faiss.read_index(index_path)
    return index


def build_index(
    embeddings: np.ndarray, file_path: str = "", save=True
) -> faiss.IndexFlatL2:
    index = faiss.IndexFlatL2(OUTPUT_SIZE)
    index.add(embeddings)
    if save:
        faiss.write_index(index, file_path)
    return index


def find_neighbors(index: faiss.IndexFlatL2, embedding: np.ndarray, k: int):
    index.search(embedding, k)


# policzenie zanurzeń
# zbudowanie indeksu


# na podstawie id książki dostać zanurzenie
# na podstawie zanurzenia chcemy dostać ileś najbliższych idków książek
