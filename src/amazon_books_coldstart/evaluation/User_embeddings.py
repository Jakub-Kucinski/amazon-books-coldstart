from src.amazon_books_coldstart.models.User_embeddings import User_embeddings 
from src.amazon_books_coldstart.tests.test import test_model

test_model("validation", User_embeddings(), "description embedding")