import torch
from torch.utils.data import DataLoader, Dataset


def flatten(lst):
    return [item for sublist in lst for item in sublist]


class AmazonBooksDataset(Dataset):
    def __init__(self, df_books, df_ratings, embeddings, book_2_row, device):
        unique_authors = set(
            flatten(
                df_books["authors"].apply(lambda x: x.strip("[]").split(",")).tolist()
            )
        )
        self.author_2_idx = {
            author: idx + 1 for idx, author in enumerate(unique_authors)
        }

        unique_categories = set(
            flatten(df_books["categories"].apply(lambda x: x.strip("[]").split(",")))
        )
        self.category_2_idx = {
            category: idx + 1 for idx, category in enumerate(unique_categories)
        }

        unique_publishers = set(df_books["publisher"].unique())
        self.publisher_2_idx = {
            publisher: idx + 1 for idx, publisher in enumerate(unique_publishers)
        }

        self.device = device
        self.df_books = df_books

        df_pairs = df_ratings.merge(df_ratings, on="user_id", how="inner")
        df_pairs = df_pairs.drop(columns=["score_x", "score_y"])
        df_pairs = df_pairs[df_pairs["book_id_x"] != df_pairs["book_id_y"]]
        self.df_pairs = df_pairs

        self.embeddings = embeddings
        self.book_2_row = book_2_row

    def __len__(self):
        return self.df_pairs.shape[0]

    def prepare_sample(self, book, device):
        embedding = self.embeddings[self.book_2_row[book["book_id"]]]
        author_ids = [
            self.author_2_idx[author]
            for author in book["authors"].strip("[]").split(",")
        ]
        category_ids = [
            self.category_2_idx[category]
            for category in book["categories"].strip("[]").split(",")
        ]
        publisher_id = self.publisher_2_idx[book["publisher"]]

        return (
            torch.tensor(embedding, device=device),
            torch.tensor(author_ids, dtype=torch.long, device=device),
            torch.tensor(category_ids, dtype=torch.long, device=device),
            torch.tensor(publisher_id, dtype=torch.long, device=device),
        )

    def __getitem__(self, idx):
        row = self.df_pairs.iloc[idx]
        book1 = self.df_books[self.df_books["book_id"] == row["book_id_x"]].iloc[0]
        book2 = self.df_books[self.df_books["book_id"] == row["book_id_y"]].iloc[0]
        sample1 = self.prepare_sample(book1, self.device)
        sample2 = self.prepare_sample(book2, self.device)
        return (sample1, sample2)


def create_collate_fn(device):
    def collate_fn(batch):
        def collate_colmn(batch):
            embeddings = torch.stack([sample[0] for sample in batch])
            author_ids = torch.stack([sample[1] for sample in batch])
            author_offsets = torch.cumsum(
                torch.tensor(
                    [0] + [len(sample[1]) for sample in batch],
                    dtype=torch.long,
                    device=device,
                ),
                dim=0,
            )[:-1]
            category_ids = torch.stack([sample[2] for sample in batch])
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

        col1 = collate_colmn([sample[0] for sample in batch])
        col2 = collate_colmn([sample[1] for sample in batch])
        return (col1, col2)

    return collate_fn


def get_train_dataloader(
    df_books, df_ratings, embeddings, book_2_row, device, batch_size=128, shuffle=False
):
    dataset = AmazonBooksDataset(df_books, df_ratings, embeddings, book_2_row, device)
    collate_fn = create_collate_fn(device)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
