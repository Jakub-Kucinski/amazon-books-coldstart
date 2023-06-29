# Table of contents
- [Table of contents](#table-of-contents)
- [Installation](#installation)
  - [Environment setup](#environment-setup)
  - [Pre-commit installation](#pre-commit-installation)
  - [Dependencies update](#dependencies-update)
- [Pipelines](#pipelines)
  - [Data download](#data-download)
  - [Data preprocessing](#data-preprocessing)
  - [Embeddings creation](#embeddings-creation)
    - [Usage](#usage)

# Installation

## Environment setup
```bash
conda env create -f conda.yml
conda activate amazon-books-coldstart
poetry install
```

## Pre-commit installation

```shell
pre-commit install
pre-commit autoupdate
```

To check all files without committing simply run:
```shell
pre-commit run --all-files
```

## Dependencies update
To update your environment with the latest dependencies from `pyproject.toml`, use
```shell
poetry update
```

To add new package to the poetry environment, use
```bash
poetry add <package-name>
```
To remove a package, use
```bash
poetry remove <package-name>
```

# Pipelines
First please install the repo by running:
```shell
python setup.py develop
```

## Data download

Due to considerable amount of effort necessary to implement downloading dataset from Kaggle, we decided to let the user do it manually.
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews/download?datasetVersionNumber=1) and unzip it inside directory `data/01_raw/`.
After this process there should be two files available:
* `data/01_raw/books_data.csv`
* `data/01_raw/Books_rating.csv`

## Data preprocessing

To filter out missing data and split books and reviews into train, validation and test sets, run
```shell
python src/amazon_books_coldstart/data_preprocessing/pandas_preprocessing.py
```

## Embeddings creation

```shell
python src/amazon_books_coldstart/embeddings_creation/create_embeddings.py
```

### Usage

In `src/amazon_books_coldstart/models/booksindex.py` class `BooksIndex` is implemented. It gives a possibility to search by description, for example:

```python
from src.amazon_books_coldstart.models.booksindex import BooksIndex

index = BooksIndex("data/03_primary/train.index", "data/03_primary/train_id_2_row.json")
distances, neighbors = index.find_neighbors("Fantasy book description", 10)
```
