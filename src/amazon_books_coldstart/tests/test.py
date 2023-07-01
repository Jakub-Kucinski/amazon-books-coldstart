import json

import numpy as np
import pandas as pd
from tqdm import tqdm


class Test:
    # mode:
    # "description embedding"
    # "book embedding"
    # "params"
    def __init__(self, dataset_type, mode):
        self.mode = mode
        self.description_embeddings = np.load(
            "data/03_primary/" + dataset_type + "_embeddings.npy"
        )
        self.book_id_to_row_description = json.load(
            open("data/03_primary/" + dataset_type + "_id_2_row.json", mode="r")
        )
        self.book_embeddings = np.load(
            "data/07_model_output/" + dataset_type + "_model_output.npy"
        )
        self.book_id_to_row = json.load(
            open("data/07_model_output/" + dataset_type + "_id_2_row.json", mode="r")
        )

        self.books = pd.read_csv("data/02_intermediate/" + dataset_type + "_books.csv")
        self.ratings = pd.read_csv(
            "data/02_intermediate/" + dataset_type + "_ratings.csv"
        )
        self.number_of_reviews = {}
        self.number_of_users_that_reviewed = {}
        self.reviews = {}
        for _, review in self.ratings.iterrows():
            book_id = review["book_id"]
            user_id = review["user_id"]
            if not (book_id in self.number_of_reviews):
                self.number_of_reviews[book_id] = 0
            self.number_of_reviews[book_id] += 1
            tuple = (book_id, user_id)
            if not (tuple in self.reviews):
                self.reviews[tuple] = 0
                if not (book_id in self.number_of_users_that_reviewed):
                    self.number_of_users_that_reviewed[book_id] = 0
                self.number_of_users_that_reviewed[book_id] += 1
            self.reviews[tuple] += 1

    def get_description_embedding(self, book_id):
        return self.description_embeddings[self.book_id_to_row_description[book_id]]

    def get_book_embedding(self, book_id):
        return self.book_embeddings[self.book_id_to_row[book_id]]

    def test_books_helper(self, get_answer, get_number_of_users):
        precision = 0
        recall = 0
        books_below_threshold = 0
        books_above_threshold = 0
        precision_below_threshold = 0
        precision_above_threshold = 0
        recall_below_threshold = 0
        recall_above_threshold = 0
        threshold = 20
        for index, book in tqdm(self.books.iterrows(), total=self.books.shape[0]):
            book_id = book["book_id"]
            recommended_users = []
            if self.mode == "description embedding":
                recommended_users = get_answer(
                    self.get_description_embedding(book_id),
                    get_number_of_users(book_id),
                )
            if self.mode == "book embedding":
                recommended_users = get_answer(
                    self.get_book_embedding(book_id), get_number_of_users(book_id)
                )
            if self.mode == "params":
                recommended_users = get_answer(
                    book["authors"],
                    book["publisher"],
                    book["categories"],
                    book["description"],
                    get_number_of_users(book_id),
                )
            number_of_reviews = self.number_of_reviews[book_id]
            correct_guesses = 0
            for user_id in recommended_users:
                if (book_id, user_id) in self.reviews:
                    correct_guesses += 1
            prec = correct_guesses / max(1, len(recommended_users))
            rec = correct_guesses / number_of_reviews
            precision += prec
            recall += rec
            if number_of_reviews < threshold:
                precision_below_threshold += prec
                recall_below_threshold += rec
                books_below_threshold += 1
            else:
                precision_above_threshold += prec
                recall_above_threshold += rec
                books_above_threshold += 1
            # print(index, correct_guesses, number_of_reviews, "recall =", correct_guesses / number_of_reviews, "precision =", correct_guesses / max(1, len(recommended_users)))

        def print_stats(text, prec, rec, number_of_books):
            print(text)
            print("number of books =", number_of_books)
            print("average precision =", prec / number_of_books)
            print("average recall =", rec / number_of_books)

        print_stats("all books", precision, recall, len(self.books))
        print_stats(
            "books below threshold",
            precision_below_threshold,
            recall_below_threshold,
            books_below_threshold,
        )
        print_stats(
            "book on and above threshold",
            precision_above_threshold,
            recall_above_threshold,
            books_above_threshold,
        )

    def test_books(self, get_answer):
        def get_number_of_users(_):
            return 20

        return self.test_books_helper(get_answer, get_number_of_users)

    def test_books_2(self, get_answer):
        def get_number_of_users(book_id):
            if book_id in self.number_of_users_that_reviewed:
                return self.number_of_users_that_reviewed[book_id]
            else:
                return 20

        return self.test_books_helper(get_answer, get_number_of_users)


def test_model(dataset_type, model, mode="params"):
    print("Creating testing class")
    test = Test(dataset_type, mode)
    print("Testing model")
    test.test_books(model.recommend_users)
    test.test_books_2(model.recommend_users)
