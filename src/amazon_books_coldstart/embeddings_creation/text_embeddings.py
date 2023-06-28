import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.amazon_books_coldstart import config

model = SentenceTransformer(config.MODEL_NAME)
model.max_seq_length = config.MAX_SEQ_LENGTH


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
    index = faiss.IndexFlatL2(config.OUTPUT_SIZE)
    index.add(embeddings)
    if save:
        faiss.write_index(index, file_path)
    return index


def find_neighbors(index: faiss.IndexFlatL2, embedding: np.ndarray, k: int):
    index.search(embedding, k)
