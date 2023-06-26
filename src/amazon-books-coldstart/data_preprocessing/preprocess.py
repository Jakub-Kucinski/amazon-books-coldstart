import csv

import numpy as np


class Book:
    def __init__(self, row, id):
        self.id = id
        self.title = str(row[0])
        self.desciption = str(row[1])
        self.authors = row[2]
        self.image = str(row[3])
        self.previewLink = str(row[4])
        self.publisher = str(row[5])
        self.published_date = str(row[6])
        self.infoLink = str(row[7])
        self.categories = row[8]
        self.ratingsCount = row[9]


class Review:
    def __init__(self, row):
        def parse_helpfulness(helpfulness):
            helpfulness = helpfulness.split("/")
            return [int(helpfulness[0]), int(helpfulness[1])]

        self.book_id = str(row[0])
        self.book_title = str(row[1])
        if row[2] == "":
            self.price = 0
        else:
            self.price = float(row[2])
        self.user_id = str(row[3])
        self.profile_name = str(row[4])
        self.helpfulness = parse_helpfulness(row[5])
        self.score = float(row[6])
        self.time = int(row[7])
        self.summary = str(row[8])
        self.text = str(row[9])


class Data_reader:
    def __init__(self):
        self.title_to_id = {}
        self.ratings = {}
        self.books = {}
        self.read_ratings()
        self.read_books()
        # 0.7 - train, 0.2 - validation, 0.1 - test
        self.split_dataset(0.7, 0.2)

    def read_ratings(self):
        i = 0
        success = 0
        with open(
            "../../../data/01_raw/Books_rating.csv",
            newline="",
        ) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=",", quotechar="|")
            for row in spamreader:
                if i > 0:
                    try:
                        review = Review(row)
                        self.ratings[str(row[0])] = review
                        self.title_to_id[review.book_title] = review.book_id
                        success += 1
                    except:  # noqa
                        pass
                i += 1
        print("Read " + str(success) + " out of " + str(i) + " ratings.")

    def read_books(self):
        i = 0
        success = 0
        with open(
            "../../../data/01_raw/books_data.csv",
            newline="",
        ) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=",", quotechar="|")
            for row in spamreader:
                if i > 0:
                    try:
                        id = self.title_to_id[row[0]]
                        self.books[id] = Book(row, id)
                        success += 1
                    except:  # noqa
                        pass
                i += 1
        print("Read " + str(success) + " out of " + str(i) + " books.")

    def split_dataset(self, train_size, valid_size):
        train_size = int(len(self.title_to_id) * train_size)
        valid_size = int(len(self.title_to_id) * valid_size)
        np.random.seed(0)
        all_indices = np.random.choice(
            list(self.title_to_id.values()),
            size=len(self.title_to_id),
            replace=False,
        )
        train_indices = set(all_indices[:train_size])
        valid_indices = set(all_indices[train_size : train_size + valid_size])
        test_indices = set(all_indices[train_size + valid_size :])
        ratings = {"test": {}, "train": {}, "validation": {}}
        books = {"test": {}, "train": {}, "validation": {}}
        for id in self.ratings.keys():
            review = self.ratings[id]
            if review.book_id in train_indices:
                ratings["train"][id] = review
            if review.book_id in valid_indices:
                ratings["validation"][id] = review
            if review.book_id in test_indices:
                ratings["test"][id] = review
        for id in self.books.keys():
            book = self.books[id]
            if book.id in train_indices:
                books["train"][id] = review
            if book.id in valid_indices:
                books["validation"][id] = review
            if book.id in test_indices:
                books["test"][id] = review
        self.ratings = ratings
        self.books = books
