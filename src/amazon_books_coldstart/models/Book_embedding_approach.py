import ast

import numpy as np
import pandas as pd

from src.amazon_books_coldstart.models.booksindex import Index
from src.amazon_books_coldstart.models.booksindex import Index_cosine_similarity

class Book_embedding_approach:
    def __init__(self, distance_mapping, score_mapping, use_cosine_similarity):
        if use_cosine_similarity:
            self.index = Index_cosine_similarity("data/04_feature/train_model_output.npy", "data/04_feature/train_id_2_row.json")        
        else:
            self.index = Index("data/04_feature/train_model_output.npy", "data/04_feature/train_id_2_row.json")
        self.ratings = pd.read_csv("data/02_intermediate/train_ratings.csv")

        self.distance_mapping = distance_mapping
        self.score_mapping = score_mapping
        self.reviews = {}

        for _, review in self.ratings.iterrows():
            book_id = review["book_id"]
            user_id = review["user_id"]
            score = review["score"]
            if not (book_id in self.reviews):
                self.reviews[book_id] = []
            self.reviews[book_id].append([user_id, score])

    def get_similar_books(self, embedding, number_of_books):
        scores, books = self.index.search(embedding, number_of_books)
        for i in range(number_of_books):
            scores[i] = self.distance_mapping(scores[i])
        return scores, books

    def get_best_users(self, scores, books, number_of_users):
        assert len(scores) == len(books), "Scores and books should have the same length"
        if len(books) == 0:
            return []
        score_dict = {}
        for i in range(len(scores)):
            score_dict[books[i]] = float(scores[i])
        user_score_dict = {}
        for book_id in books:
            for [user_id, score] in self.reviews[book_id]:
                if not (user_id in user_score_dict):
                    user_score_dict[user_id] = 0
                user_score_dict[user_id] += self.score_mapping(float(score)) * score_dict[book_id]

        result = []
        for user_id, score in list(user_score_dict.items()):
            result.append([score, user_id])
        result.sort()
        result.reverse()
        return list(np.array(result)[: min(len(result), number_of_users), 1])

    def recommend_users(self, book_embedding, number_of_users=20):
        scores, books = self.get_similar_books(book_embedding, number_of_users * 20)
        return self.get_best_users(scores, books, number_of_users)
