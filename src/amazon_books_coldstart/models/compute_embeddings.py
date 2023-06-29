import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def flatten(lst):
    return [item for sublist in lst for item in sublist]


class BooksDataset(Dataset):
    def __init__(
        self,
        df_books,
        embeddings,
        book_2_row,
        author_2_idx,
        category_2_idx,
        publisher_2_idx,
        device,
    ):
        self.author_2_idx = author_2_idx
        self.category_2_idx = category_2_idx
        self.publisher_2_idx = publisher_2_idx

        self.device = device
        self.df_books = df_books

        self.embeddings = embeddings
        self.book_2_row = book_2_row

    def __len__(self):
        return self.df_books.shape[0]

    def prepare_sample(self, book, device):
        embedding = self.embeddings[self.book_2_row[book["book_id"]]]
        author_ids = [
            self.author_2_idx.get(author)
            for author in book["authors"].strip("[]").split(",")
        ]
        author_ids = [
            author_id if author_id is not None else 0 for author_id in author_ids
        ]
        category_ids = [
            self.category_2_idx.get(category)
            for category in book["categories"].strip("[]").split(",")
        ]
        category_ids = [
            category_id if category_id is not None else 0
            for category_id in category_ids
        ]
        publisher_id = self.publisher_2_idx.get(book["publisher"])
        if publisher_id is None:
            publisher_id = 0

        return (
            torch.tensor(embedding, device=device),
            torch.tensor(author_ids, dtype=torch.long, device=device),
            torch.tensor(category_ids, dtype=torch.long, device=device),
            torch.tensor(publisher_id, dtype=torch.long, device=device),
        )

    def __getitem__(self, idx):
        book = self.df_books.iloc[idx]
        sample = self.prepare_sample(book, self.device)
        return sample


def create_collate_fn(device):
    def collate_fn(batch):
        embeddings = torch.stack([sample[0] for sample in batch])
        author_ids = torch.cat([sample[1] for sample in batch], 0)
        author_offsets = torch.cumsum(
            torch.tensor(
                [0] + [len(sample[1]) for sample in batch],
                dtype=torch.long,
                device=device,
            ),
            dim=0,
        )[:-1]
        category_ids = torch.cat([sample[2] for sample in batch], 0)
        category_offsets = torch.cumsum(
            torch.tensor(
                [0] + [len(sample[2]) for sample in batch],
                dtype=torch.long,
                device=device,
            ),
            dim=0,
        )[:-1]
        publisher_ids = torch.stack([sample[3] for sample in batch])
        return (
            embeddings,
            author_ids,
            author_offsets,
            category_ids,
            category_offsets,
            publisher_ids,
        )

    return collate_fn


def compute_all_embeddings(model, device="cuda"):
    def compute_embeddings(dataloader):
        model_output = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                embeddings = model(batch)
                model_output.append(embeddings.detach().cpu().numpy())

        return np.concatenate(model_output, axis=0)

    author_2_id = json.load(open("data/04_features/author_ids.json", mode="r"))
    category_2_id = json.load(open("data/04_features/category_ids.json", mode="r"))
    publisher_2_id = json.load(open("data/04_features/publisher_ids.json", mode="r"))

    for prefix in ["train", "validation", "test"]:
        df_books = pd.read_csv(f"data/04_features/{prefix}_books.csv")
        book_2_row = json.load(
            open(f"data/04_features/{prefix}_book_2_row.json", mode="r")
        )
        embeddings = np.load(f"data/04_features/{prefix}_embeddings.npy")
        dataset = BooksDataset(
            df_books,
            embeddings,
            book_2_row,
            author_2_id,
            category_2_id,
            publisher_2_id,
            device,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            collate_fn=create_collate_fn(device),
        )
        model_output = compute_embeddings(dataloader)
        np.save(f"data/07_model_output/{prefix}_model_output.npy", model_output)
