import pandas as pd
import numpy as np
import ast
from tqdm import tqdm

class Simple_approach():
    def __init__(self):
        self.books = pd.read_csv("data/02_intermediate/train_books.csv")[['book_id', 'authors', 'publisher', 'categories']]
        self.ratings = pd.read_csv("data/02_intermediate/train_ratings.csv")
        self.author = {}
        self.publisher = {}
        self.categories = {}
        self.reviews = {}

        for _, book in self.books.iterrows():
            book_id = book['book_id']
            publisher = book['publisher']
            categories = book['categories']
            for author in set(ast.literal_eval(book['authors'])):
                if not (author in self.author):
                    self.author[author] = []
                self.author[author].append(book_id)
            if not (publisher in self.publisher):
                self.publisher[publisher] = []
            self.publisher[publisher].append(book_id)
            if not (categories in self.categories):
                self.categories[categories] = []
            self.categories[categories].append(book_id)
        
        for _, review in self.ratings.iterrows():
            book_id = review['book_id']
            user_id = review['user_id']
            score = review['score']
            if not (book_id in self.reviews):
                self.reviews[book_id] = []
            self.reviews[book_id].append([user_id, score])
            
    def get_similar_books(self, authors, publisher, categories, number_of_books):        
        candidates = {}
        def add(book_id, score):
            if not (book_id in candidates):
                candidates[book_id] = 0
            candidates[book_id] += score

        for author in set(ast.literal_eval(authors)):
            if author in self.author:
                for book_id in self.author[author]:
                    add(book_id, 10)
        if publisher in self.publisher:
            for book_id in self.publisher[publisher]:
                add(book_id, 1)
        if categories in self.categories:
            for book_id in self.categories[categories]:
                add(book_id, 5)
        
        cand = []
        for item in candidates.items():
            cand.append([item[1], item[0]])
        cand.sort()
        cand.reverse()
        scores = []
        books = []
        for i in range(len(cand)):
            scores.append(cand[i][0])
            books.append(cand[i][1])
        number_of_books = min(len(scores), number_of_books)
        return scores[:number_of_books], books[:number_of_books]
    
    def get_best_users(self, scores, books, number_of_users):
        assert len(scores) == len(books), "Scores and books should have the same length"
        if len(books) == 0:
            return []
        score_dict = {}
        for i in range(len(scores)):
            score_dict[books[i]] = int(scores[i])
        user_score_dict = {}
        for book_id in books:
            for [user_id, score] in self.reviews[book_id]:
                if not (user_id in user_score_dict):
                    user_score_dict[user_id] = 0
                user_score_dict[user_id] += float(score) * score_dict[book_id]
        
        result = []
        for (user_id, score) in list(user_score_dict.items()):
            result.append([score, user_id])
        result.sort()
        result.reverse()
        return list(np.array(result)[:min(len(result), number_of_users), 1])

    def recommend_users(self, authors, publisher, categories, number_of_users=20):
        scores, books = self.get_similar_books(authors, publisher, categories, number_of_users * 20)
        return self.get_best_users(scores, books, number_of_users)