from pathlib import Path

import click
import pandas as pd


def read_books(path="data/01_raw/books_data.csv"):
    books = pd.read_csv(path)
    return books


def read_ratings(path="data/01_raw/Books_rating.csv"):
    ratings = pd.read_csv(path)
    return ratings


def filter_data(
    ratings,
    books,
    remove_no_description=True,
    remove_no_publisher=True,
    remove_no_author=True,
    remove_no_title=True,
    remove_no_category=True,
):
    if remove_no_description:
        books = books[books["description"].notna()]
    if remove_no_publisher:
        books = books[books["publisher"].notna()]
    if remove_no_author:
        books = books[books["authors"].notna()]
    if remove_no_title:
        books = books[books["Title"].notna()]
    if remove_no_category:
        books = books[books["categories"].notna()]
    books = books.join(
        ratings[["Id", "Title"]].drop_duplicates().set_index("Title"),
        on="Title",
        how="inner",
    )

    books = books[books["Title"].isin(ratings["Title"])]
    ratings = ratings[ratings["Title"].isin(books["Title"])]
    books = books[books["Id"].isin(ratings["Id"])]
    ratings = ratings[ratings["Id"].isin(books["Id"])]
    return ratings, books


def split_data_by_books(ratings, books, train=0.7, validation=0.2, test=0.1):
    if abs(1 - train + validation + test) < 1e-5:
        raise ValueError("Train, validation and test sizes must sum to 1")
    ratings = ratings[["Id", "User_id", "review/score"]]
    ratings = ratings.rename(
        columns={"Id": "book_id", "User_id": "user_id", "review/score": "score"}
    )
    books = books[
        ["Id", "Title", "authors", "description", "publisher", "categories", "image"]
    ]
    books = books.rename(
        columns={
            "Id": "book_id",
            "Title": "title",
            "authors": "authors",
            "description": "description",
            "publisher": "publisher",
            "categories": "categories",
            "image": "image_url",
        }
    )
    books = books.sample(frac=1, random_state=42)
    train_size = int(len(books) * train)
    validation_size = int(len(books) * validation)
    train_books = books[:train_size]
    validation_books = books[train_size : train_size + validation_size]
    test_books = books[train_size + validation_size :]
    train_ratings = ratings[ratings["book_id"].isin(train_books["book_id"])]
    validation_ratings = ratings[ratings["book_id"].isin(validation_books["book_id"])]
    test_ratings = ratings[ratings["book_id"].isin(test_books["book_id"])]
    return (
        train_ratings,
        validation_ratings,
        test_ratings,
        train_books,
        validation_books,
        test_books,
    )


@click.command()
@click.option("--data_path", default="data/01_raw/")
@click.option("--destination_path", default="data/02_intermediate/")
@click.option("--ratings_file", default="Books_rating.csv")
@click.option("--books_file", default="books_data.csv")
@click.option("--dont_remove_description", is_flag=True, default=False)
@click.option("--dont_remove_publisher", is_flag=True, default=False)
@click.option("--dont_remove_author", is_flag=True, default=False)
@click.option("--dont_remove_title", is_flag=True, default=False)
@click.option("--dont_remove_category", is_flag=True, default=False)
@click.option("--train", default=0.7)
@click.option("--validation", default=0.2)
@click.option("--test", default=0.1)
@click.option("--verbose", is_flag=True, default=False)
def main(
    data_path,
    destination_path,
    ratings_file,
    books_file,
    dont_remove_description,
    dont_remove_publisher,
    dont_remove_author,
    dont_remove_title,
    dont_remove_category,
    train,
    validation,
    test,
    verbose,
):
    data_path = Path(data_path)
    if verbose:
        print("Reading data")
    ratings = read_ratings(data_path / ratings_file)
    books = read_books(data_path / books_file)
    if verbose:
        print("Filtering data")
    ratings, books = filter_data(
        ratings,
        books,
        remove_no_description=not dont_remove_description,
        remove_no_publisher=not dont_remove_publisher,
        remove_no_author=not dont_remove_author,
        remove_no_title=not dont_remove_title,
        remove_no_category=not dont_remove_category,
    )
    if verbose:
        print("Splitting data")
    (
        train_ratings,
        validation_ratings,
        test_ratings,
        train_books,
        validation_books,
        test_books,
    ) = split_data_by_books(
        ratings, books, train=train, validation=validation, test=test
    )
    if verbose:
        print("Saving data")
    destination_path = Path(destination_path)
    train_ratings.to_csv(destination_path / "train_ratings.csv", index=False)
    validation_ratings.to_csv(destination_path / "validation_ratings.csv", index=False)
    test_ratings.to_csv(destination_path / "test_ratings.csv", index=False)
    train_books.to_csv(destination_path / "train_books.csv", index=False)
    validation_books.to_csv(destination_path / "validation_books.csv", index=False)
    test_books.to_csv(destination_path / "test_books.csv", index=False)


if __name__ == "__main__":
    main()
