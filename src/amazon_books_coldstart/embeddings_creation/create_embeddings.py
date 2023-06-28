import json

import faiss
import numpy as np
import pandas as pd

from src.amazon_books_coldstart.embeddings_creation.text_embeddings import (
    build_index,
    encode_list,
)

FILE_PREFIXES = ["train", "validation", "test"]


def create_embeddings(file_prefix, save=True):
    df = pd.read_csv(f"data/02_intermediate/{file_prefix}_books.csv")
    embeddings = encode_list(df["description"].tolist())
    if save:
        np.save(f"data/03_primary/{file_prefix}_embeddings.npy", embeddings)
        save_json(
            {book_id: row for row, book_id in enumerate(df["book_id"].values)},
            f"data/03_primary/{file_prefix}_id_2_row.json",
        )
    return embeddings


def load_index(file_prefix):
    index = faiss.read_index(f"data/03_primary/{file_prefix}.index")
    return index


def save_json(obj, file_path):
    with open(file_path, "w") as f:
        json.dump(obj, f)


for file_prefix in FILE_PREFIXES:
    embeddings = create_embeddings(file_prefix)
    index = build_index(embeddings, f"data/03_primary/{file_prefix}.index")
