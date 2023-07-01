import faiss
import numpy as np
import ast
import pandas as pd
import json
from tqdm import tqdm
import warnings

from sklearn.cluster import KMeans
from src.amazon_books_coldstart.models.booksindex import Index_from_list
from src.amazon_books_coldstart.models.booksindex import Index_cosine_similarity_from_list

class User_embeddings3:
    def __init__(self, max_clusters, use_cosine_similarity):
        self.embeddings = np.load("data/04_feature/train_model_output.npy")
        self.book_id_to_row = json.load(open("data/04_feature/train_id_2_row.json", mode="r"))
        self.user_books = {}
        self.user_weights = {}
        self.kmeans = []
        self.max_clusters = max_clusters
        self.use_cosine_similarity = use_cosine_similarity
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
            embeddings = np.array(embeddings)
            if use_cosine_similarity:
                self.index = Index_cosine_similarity_from_list(embeddings, self.row_id_to_user_id)
            else:
                self.index = Index_from_list(embeddings, self.row_id_to_user_id)

    def get_embedding(self, book_id):
        return self.embeddings[self.book_id_to_row[book_id]]
    
    def get_clusters(self, n_clusters, books, scores):
            self.kmeans[n_clusters].fit(books, sample_weight=scores)
            return self.kmeans[n_clusters].cluster_centers_

    def recommend_users(self, description_embedding, number_of_users=20):
        res = set()
        distances, users_ids = self.index.search(description_embedding.reshape((1, -1)), self.max_clusters * number_of_users)
        tmp = []
        for i in range(len(distances)):
            tmp.append([distances[i], users_ids[i]])
        tmp.sort()
        if self.use_cosine_similarity:
            tmp.reverse()
        for i in range(len(tmp)):
            res.add(tmp[i][1])
            if len(res) == number_of_users:
                break

        return list(res)
