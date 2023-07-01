import faiss
import numpy as np
import ast
import pandas as pd
import json

from sklearn.cluster import KMeans
from src.amazon_books_coldstart.models.booksindex import BooksIndex
from src.amazon_books_coldstart.config import OUTPUT_SIZE

class User_embeddings:
    def __init__(self):
        self.embeddings = np.load("data/03_primary/train_embeddings.npy")
        self.book_id_to_row = json.load(open("data/03_primary/train_id_2_row.json", mode="r"))
        self.user_embeddings = {}
        self.user_sum = {}
        for _, review in pd.read_csv("data/02_intermediate/train_ratings.csv").iterrows():
            book_id = review["book_id"]
            user_id = review["user_id"]
            score = review["score"]

            if not (user_id in self.user_embeddings):
                self.user_embeddings[user_id] = np.zeros(OUTPUT_SIZE)
                self.user_sum[user_id] = 0
            self.user_embeddings[user_id] += self.get_embedding(book_id) * score
            self.user_sum[user_id] += score

        embeddings = []
        self.row_id_to_user_id = []
        for key in self.user_embeddings.keys():
            embeddings.append(self.user_embeddings[key] / self.user_sum[key])
            self.row_id_to_user_id.append(key)
        self.index = faiss.IndexFlatL2(OUTPUT_SIZE)
        embeddings = np.array(embeddings)
        self.index.add(embeddings)

    def get_embedding(self, book_id):
        return self.embeddings[self.book_id_to_row[book_id]]
    
    def recommend_users(self, description_embedding, number_of_users=20):
        res = []
        for row_id in self.index.search(description_embedding.reshape((1, -1)), number_of_users)[1].reshape(-1):
            res.append(self.row_id_to_user_id[row_id])
        return res
