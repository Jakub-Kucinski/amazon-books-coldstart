from src.amazon_books_coldstart.models.User_embeddings3 import User_embeddings3
from src.amazon_books_coldstart.tests.test import test_model

for i in range(1, 9):
    print("max_number_of_clusters =", str(i))
    test_model("validation", User_embeddings3(i), "book embedding")