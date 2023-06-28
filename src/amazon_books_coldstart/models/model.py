import torch

from src.amazon_books_coldstart.config import OUTPUT_SIZE


class AmazonBooksModelv1(torch.nn.Module):
    def __init__(
        self,
        description_embeddings,
        title_embeddings,
        d_description,
        d_title,
        linear1_size=1024,
        linear2_size=512,
        linear3_size=128,
    ):
        super(AmazonBooksModelv1, self).__init__()
        self.description_embeddings = description_embeddings
        self.title_embeddings = title_embeddings
        self.linear1 = torch.nn.Linear(d_description + d_title, linear1_size)
        self.linear2 = torch.nn.Linear(linear1_size, linear2_size)
        self.linear3 = torch.nn.Linear(linear3_size, linear3_size)

    def forward(self, id):
        description = self.description_embeddings(id)
        title = self.title_embeddings(id)
        x = torch.cat((description, title), dim=1)
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        x = torch.nn.functional.relu(x)
        x = self.linear3(x)
        return x


class AmazonBooksModelv2(torch.nn.Module):
    def __init__(
        self,
        description_embeddings,
        title_embeddings,
        d_description,
        d_title,
        linear1_size=1024,
        linear2_size=256,
    ):
        super(AmazonBooksModelv2, self).__init__()
        self.description_embeddings = description_embeddings
        self.title_embeddings = title_embeddings
        self.linear1 = torch.nn.Linear(d_description + d_title, linear1_size)
        self.linear2 = torch.nn.Linear(linear1_size, linear2_size)

    def forward(self, id):
        description = self.description_embeddings(id)
        title = self.title_embeddings(id)
        x = torch.cat((description, title), dim=1)
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        return x


class AmazonBooksModelv3(torch.nn.Module):
    def __init__(
        self,
        d_description=OUTPUT_SIZE,
        n_authors=82462,
        n_publishers=None,
        n_categories=None,
        d_authors=128,
        d_publishers=128,
        d_categories=128,
        linear1_size=1024,
        linear2_size=256,
    ):
        super(AmazonBooksModelv3, self).__init__()
        self.authors_embeddings = torch.nn.EmbeddingBag(
            n_authors, d_authors, padding_idx=0, mode="mean"
        )
        self.publishers_embeddings = torch.nn.Embedding(
            n_publishers, d_publishers, padding_idx=0
        )
        self.categories_embeddings = torch.nn.EmbeddingBag(
            n_categories, d_categories, padding_idx=0, mode="mean"
        )
        self.linear1 = torch.nn.Linear(
            d_description + d_authors + d_publishers + d_categories, linear1_size
        )
        self.linear2 = torch.nn.Linear(linear1_size, linear2_size)

    def forward(self, description_embedding, authors, publishers, categories):
        authors = self.authors_embeddings(authors)
        publishers = self.publishers_embeddings(publishers)
        categories = self.categories_embeddings(categories)
        x = torch.cat((description_embedding, authors, publishers, categories), dim=1)
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        return x
