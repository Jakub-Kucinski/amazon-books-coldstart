import importlib
from Simple_approach import Simple_approach
# from src.amazon_books_coldstart.evaluation.Simple_approach import Simple_approach

simple_approach = Simple_approach()
test = importlib.import_module("src.amazon-books-coldstart.tests.test").Test('validation')
test.test_books(simple_approach.recommend_users)
test.test_books_2(simple_approach.recommend_users)
