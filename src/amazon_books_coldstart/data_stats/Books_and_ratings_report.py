import pandas as pd
from ydata_profiling import ProfileReport

books = pd.read_csv("data/02_intermediate/train_books.csv")
ratings = pd.read_csv("data/02_intermediate/train_ratings.csv")

profile = ProfileReport(books, title="Books' profiling report")
profile.to_file("data/08_reporting/Books_report.html")

profile = ProfileReport(ratings, title="Ratings' profiling report")
profile.to_file("data/08_reporting/Ratings_report.html")
