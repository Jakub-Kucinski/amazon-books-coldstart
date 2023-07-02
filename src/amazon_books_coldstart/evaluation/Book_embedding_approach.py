from src.amazon_books_coldstart.models.Book_embedding_approach import Book_embedding_approach
from src.amazon_books_coldstart.tests.test import test_model
import numpy as np


class Distance_mapping:

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

print("======================================\nf2 f1 False\n")
test_model("validation", Book_embedding_approach(distance_mapping.f2, score_mapping.f1, False), "book embedding")
print("======================================\nf2 f2 False\n")
test_model("validation", Book_embedding_approach(distance_mapping.f2, score_mapping.f2, False), "book embedding")
print("======================================\nf2 f3 False\n")
test_model("validation", Book_embedding_approach(distance_mapping.f2, score_mapping.f3, False), "book embedding")
print("======================================\nf3 f1 False\n")
test_model("validation", Book_embedding_approach(distance_mapping.f3, score_mapping.f1, False), "book embedding")
print("======================================\nf3 f2 False\n")
test_model("validation", Book_embedding_approach(distance_mapping.f3, score_mapping.f2, False), "book embedding")
print("======================================\nf3 f3 False\n")
test_model("validation", Book_embedding_approach(distance_mapping.f3, score_mapping.f3, False), "book embedding")
print("======================================\nf4 f1 False\n")
test_model("validation", Book_embedding_approach(distance_mapping.f4, score_mapping.f1, False), "book embedding")
print("======================================\nf4 f2 False\n")
test_model("validation", Book_embedding_approach(distance_mapping.f4, score_mapping.f2, False), "book embedding")
print("======================================\nf4 f3 False\n")
test_model("validation", Book_embedding_approach(distance_mapping.f4, score_mapping.f3, False), "book embedding")
print("======================================\nf5 f1 False\n")
test_model("validation", Book_embedding_approach(distance_mapping.f5, score_mapping.f1, False), "book embedding")
print("======================================\nf5 f2 False\n")
test_model("validation", Book_embedding_approach(distance_mapping.f5, score_mapping.f2, False), "book embedding")
print("======================================\nf5 f3 False\n")
test_model("validation", Book_embedding_approach(distance_mapping.f5, score_mapping.f3, False), "book embedding")
print("======================================\nf2 f1 True\n")
test_model("validation", Book_embedding_approach(distance_mapping.f2, score_mapping.f1, True), "book embedding")
print("======================================\nf2 f2 True\n")
test_model("validation", Book_embedding_approach(distance_mapping.f2, score_mapping.f2, True), "book embedding")
print("======================================\nf2 f3 True\n")
test_model("validation", Book_embedding_approach(distance_mapping.f2, score_mapping.f3, True), "book embedding")
print("======================================\nf3 f1 True\n")
test_model("validation", Book_embedding_approach(distance_mapping.f3, score_mapping.f1, True), "book embedding")
print("======================================\nf3 f2 True\n")
test_model("validation", Book_embedding_approach(distance_mapping.f3, score_mapping.f2, True), "book embedding")
print("======================================\nf3 f3 True\n")
test_model("validation", Book_embedding_approach(distance_mapping.f3, score_mapping.f3, True), "book embedding")
print("======================================\nf4 f1 True\n")
test_model("validation", Book_embedding_approach(distance_mapping.f4, score_mapping.f1, True), "book embedding")
print("======================================\nf4 f2 True\n")
test_model("validation", Book_embedding_approach(distance_mapping.f4, score_mapping.f2, True), "book embedding")
print("======================================\nf4 f3 True\n")
test_model("validation", Book_embedding_approach(distance_mapping.f4, score_mapping.f3, True), "book embedding")
print("======================================\nf5 f1 True\n")
test_model("validation", Book_embedding_approach(distance_mapping.f5, score_mapping.f1, True), "book embedding")
print("======================================\nf5 f2 True\n")
test_model("validation", Book_embedding_approach(distance_mapping.f5, score_mapping.f2, True), "book embedding")
print("======================================\nf5 f3 True\n")
test_model("validation", Book_embedding_approach(distance_mapping.f5, score_mapping.f3, True), "book embedding")