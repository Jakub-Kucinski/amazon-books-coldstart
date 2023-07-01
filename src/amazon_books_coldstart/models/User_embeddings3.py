import faiss
import numpy as np
import ast
import pandas as pd
import json
from tqdm import tqdm
import warnings

from sklearn.cluster import KMeans
from src.amazon_books_coldstart.config import OUTPUT_SIZE

class User_embeddings3:
    def __init__(self, max_clusters=1):
        self.embeddings = np.load("data/04_feature/train_model_output.npy")
        self.book_id_to_row = json.load(open("data/04_feature/train_id_2_row.json", mode="r"))
        self.user_books = {}
        self.user_weights = {}
        self.kmeans = []
        for i in range(max_clusters + 1):
            self.kmeans.append(KMeans(n_clusters=max(1, i), random_state=0, n_init="auto"))
            
        for _, review in pd.read_csv("data/02_intermediate/train_ratings.csv").iterrows():
            book_id = review["book_id"]
            user_id = review["user_id"]
            score = review["score"]

            if not (user_id in self.user_books):
                self.user_books[user_id] = []
                self.user_weights[user_id] = []
            self.user_books[user_id].append(self.get_embedding(book_id))
            self.user_weights[user_id].append(score)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embeddings = []
            self.row_id_to_user_id = []
            for key in tqdm(self.user_books.keys()):
                for number_of_clusters in range(1, min(max_clusters, len(self.user_books[key])) + 1):
                    for cluster_center in self.get_clusters(number_of_clusters, self.user_books[key], self.user_weights[key]):
                        embeddings.append(cluster_center)
                        self.row_id_to_user_id.append(key)
            self.index = faiss.IndexFlatL2(OUTPUT_SIZE)
            embeddings = np.array(embeddings)
            self.index.add(embeddings)

    def get_embedding(self, book_id):
        return self.embeddings[self.book_id_to_row[book_id]]
    
    def get_clusters(self, n_clusters, books, scores):
            self.kmeans[n_clusters].fit(books, sample_weight=scores)
            return self.kmeans[n_clusters].cluster_centers_

    def recommend_users(self, description_embedding, number_of_users=20):
        res = set()
        mult = 0
        ok = False
        while (not ok):
            mult += 1
            res.clear()
            distances, rows = self.index.search(description_embedding.reshape((1, -1)), mult * number_of_users)
            distances = distances.reshape(-1)
            rows = rows.reshape(-1)
            pairs = []
            for i in range(len(distances)):
                pairs.append([distances[i], rows[i]])
            pairs.sort()
            for [_, row_id] in pairs:
                res.add(self.row_id_to_user_id[row_id])
                if len(res) == number_of_users:
                    ok = True
                    break

        return list(res)
