from src.amazon_books_coldstart.models.Simple_approach import Simple_approach
from src.amazon_books_coldstart.models.Book_embedding_approach import Book_embedding_approach
from src.amazon_books_coldstart.models.User_embeddings import User_embeddings
from src.amazon_books_coldstart.models.User_embeddings2 import User_embeddings2
from src.amazon_books_coldstart.models.User_embeddings3 import User_embeddings3
from src.amazon_books_coldstart.tests.test import test_model
import numpy as np


class Distance_mapping:

    def f1(self, x):
        return 0.

    def f2(self, x):
        return 20. * (2. - x)

    def f3(self, x):
        return 20. / (x + 1e-5)
    
    def f4(self, x):
        return 20.

    def f5(self, x):
        return x

class Score_mapping:
    
    def f1(self, x):
        return x

    def f2(self, x):
        return x - 3.5
    
    def f3(self, x):
        return 1. / (1. + np.e ** (x - 3.5))

distance_mapping = Distance_mapping()
score_mapping = Score_mapping()

print("Simple approach")
#test_model("test", Simple_approach(True, distance_mapping.f3, score_mapping.f1))

print("User embeddings")
#test_model("test", User_embeddings(), "description embedding")

print("User_embedding2")
#test_model("test", User_embeddings2(4), "description embedding")

print("User_embedding3")
test_model("test", User_embeddings3(4, False), "book embedding")

print("Book_embedding_approach")
#test_model("test", Book_embedding_approach(distance_mapping.f3, score_mapping.f1, False), "book embedding")

