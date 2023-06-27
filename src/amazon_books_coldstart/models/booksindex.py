import json

import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_SIZE = 384
MAX_SEQ_LENGTH = 256

model = SentenceTransformer(MODEL_NAME)
model.max_seq_length = MAX_SEQ_LENGTH


class BooksIndex:
    def __init__(self, index_path, book_ids_path):
        self.index = faiss.read_index(index_path)
        self.book_2_row = json.load(open(book_ids_path, mode="r"))
        self.row_2_book = {row: book for book, row in self.book_2_row.items()}

    def find_neighbours(self, description, k):
        embedding = model.encode(description)
        distances, rows = self.index.search(embedding.reshape((1, -1)), k)
        return [distance for distance in distances.reshape((-1))], [
            self.row_2_book[row] for row in rows.reshape((-1))
        ]
