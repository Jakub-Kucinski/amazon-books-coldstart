from src.amazon_books_coldstart.models.Simple_approach import Simple_approach
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

class Score_mapping:
    
    def f1(self, x):
        return x

    def f2(self, x):
        return x - 3.5
    
    def f3(self, x):
        return 1. / (1. + np.e ** (x - 3.5))

distance_mapping = Distance_mapping()
score_mapping = Score_mapping()

print("======================================\nTrue f1 f1\n")
test_model("validation", Simple_approach(True, distance_mapping.f1, score_mapping.f1))
print("======================================\nTrue f2 f1\n")
test_model("validation", Simple_approach(True, distance_mapping.f2, score_mapping.f1))
print("======================================\nTrue f3 f1\n")
test_model("validation", Simple_approach(True, distance_mapping.f3, score_mapping.f1))
print("======================================\nTrue f4 f1\n")
test_model("validation", Simple_approach(True, distance_mapping.f4, score_mapping.f1))
print("======================================\nTrue f1 f2\n")
test_model("validation", Simple_approach(True, distance_mapping.f1, score_mapping.f2))
print("======================================\nTrue f2 f2\n")
test_model("validation", Simple_approach(True, distance_mapping.f2, score_mapping.f2))
print("======================================\nTrue f3 f2\n")
test_model("validation", Simple_approach(True, distance_mapping.f3, score_mapping.f2))
print("======================================\nTrue f4 f2\n")
test_model("validation", Simple_approach(True, distance_mapping.f4, score_mapping.f2))
print("======================================\nTrue f1 f3\n")
test_model("validation", Simple_approach(True, distance_mapping.f1, score_mapping.f3))
print("======================================\nTrue f2 f3\n")
test_model("validation", Simple_approach(True, distance_mapping.f2, score_mapping.f3))
print("======================================\nTrue f3 f3\n")
test_model("validation", Simple_approach(True, distance_mapping.f3, score_mapping.f3))
print("======================================\nTrue f4 f3\n")
test_model("validation", Simple_approach(True, distance_mapping.f4, score_mapping.f3))
print("======================================\nFalse f1 f1\n")
test_model("validation", Simple_approach(False, distance_mapping.f1, score_mapping.f1))
print("======================================\nFalse f2 f1\n")
test_model("validation", Simple_approach(False, distance_mapping.f2, score_mapping.f1))
print("======================================\nFalse f3 f1\n")
test_model("validation", Simple_approach(False, distance_mapping.f3, score_mapping.f1))
print("======================================\nFalse f4 f1\n")
test_model("validation", Simple_approach(False, distance_mapping.f4, score_mapping.f1))
print("======================================\nFalse f1 f2\n")
test_model("validation", Simple_approach(False, distance_mapping.f1, score_mapping.f2))
print("======================================\nFalse f2 f2\n")
test_model("validation", Simple_approach(False, distance_mapping.f2, score_mapping.f2))
print("======================================\nFalse f3 f2\n")
test_model("validation", Simple_approach(False, distance_mapping.f3, score_mapping.f2))
print("======================================\nFalse f4 f2\n")
test_model("validation", Simple_approach(False, distance_mapping.f4, score_mapping.f2))
print("======================================\nFalse f1 f3\n")
test_model("validation", Simple_approach(False, distance_mapping.f1, score_mapping.f3))
print("======================================\nFalse f2 f3\n")
test_model("validation", Simple_approach(False, distance_mapping.f2, score_mapping.f3))
print("======================================\nFalse f3 f3\n")
test_model("validation", Simple_approach(False, distance_mapping.f3, score_mapping.f3))
print("======================================\nFalse f4 f3\n")
test_model("validation", Simple_approach(False, distance_mapping.f4, score_mapping.f3))