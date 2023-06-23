# Table of contents
- [Table of contents](#table-of-contents)
- [Installation](#installation)
  - [Environment setup](#environment-setup)
  - [Pre-commit installation](#pre-commit-installation)
  - [Dependencies update](#dependencies-update)

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
