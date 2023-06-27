# Table of contents
- [Table of contents](#table-of-contents)
- [Installation](#installation)
  - [Environment setup](#environment-setup)
  - [Pre-commit installation](#pre-commit-installation)
  - [Dependencies update](#dependencies-update)
- [Pipelines](#pipelines)
  - [Data download](#data-download)
  - [Data preprocess](#data-preprocess)

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

## Data download

Due to considerable amount of effort necessary to implement downloading dataset from Kaggle, we decided to let the user do it manually.
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews/download?datasetVersionNumber=1) and unzip it inside directory `data/01_raw/`.
After this process there should be two files available:
* `data/01_raw/books_data.csv`
* `data/01_raw/Books_rating.csv`

## Data preprocess

In order to run this pipeline it is necessary to run all cells from notebook `src/amazon-books-coldstart/data_preprocessing/preprocess.ipynb`.
The last variable `dr` is an instance of class `Data_reader` which contains the data.
