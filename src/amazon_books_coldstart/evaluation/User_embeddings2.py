from src.amazon_books_coldstart.models.User_embeddings2 import User_embeddings2 
from src.amazon_books_coldstart.tests.test import test_model

test_model("validation", User_embeddings2(5), True)