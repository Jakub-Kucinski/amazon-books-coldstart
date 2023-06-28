from src.amazon_books_coldstart.models.Simple_approach import Simple_approach
from src.amazon_books_coldstart.tests.test import Test

simple_approach = Simple_approach()
test = Test("validation")
test.test_books(simple_approach.recommend_users)
test.test_books_2(simple_approach.recommend_users)
