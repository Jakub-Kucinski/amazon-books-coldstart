import json

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from pytorch_metric_learning import losses
from tqdm import tqdm

import src.amazon_books_coldstart.models.model as model_classes
from src.amazon_books_coldstart.config import OUTPUT_SIZE
from src.amazon_books_coldstart.models.dataset import get_train_dataloader


def train_single_epoch(
    model,
    loss_func,
    train_loader,
    optimizer,
    epoch,
    log_interval=100,
    save_checkpoint_every=1000,
):
    model.train()
    for batch_idx, (data, data2) in tqdm(
        enumerate(train_loader), total=len(train_loader), leave=False
    ):
        optimizer.zero_grad()
        embeddings = model(*data)
        embeddings2 = model(*data2)
        loss = loss_func(embeddings, embeddings2)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print("Epoch {} Iteration {}: Loss = {}".format(epoch, batch_idx, loss))
        if batch_idx % save_checkpoint_every == 0:
            save_model(model, f"data/06_models/{epoch}_{batch_idx}.pth")


def wrap_loss_function(loss_function):
    loss_func = losses.SelfSupervisedLoss(loss_function)
    return loss_func


def start_training(
    model_name,
    loss,
    train_loader,
    model_params=dict(),
    optimizer_params=dict(),
    num_epochs=5,
    save_checkpoint_every=1000,
):
    loss_func = wrap_loss_function(loss)

    model_constructor = getattr(model_classes, model_name)
    model = model_constructor(**model_params).to(device)

    optimizer = optim.Adam(model.parameters(), **optimizer_params)

    for epoch in tqdm(range(1, num_epochs + 1)):
        train_single_epoch(
            model,
            loss_func,
            train_loader,
            optimizer,
            epoch,
            save_checkpoint_every=save_checkpoint_every,
        )


def save_model(model, file_path):
    torch.save(model, file_path)


def load_model(file_path):
    model = torch.load(file_path)
    return model


torch.manual_seed(2137)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
loss = losses.SupConLoss()
model_params = {
    "d_description": OUTPUT_SIZE,
    "n_authors": 82460 + 1,
    "n_publishers": 10227 + 1,
    "n_categories": 2823 + 1,
    "d_authors": 128,
    "d_publishers": 128,
    "d_categories": 128,
    "linear1_size": 1024,
    "linear2_size": 256,
}
optimizer_params = {"lr": 0.0001}
df_books = pd.read_csv("data/02_intermediate/train_books.csv")
df_ratings = pd.read_csv("data/02_intermediate/train_ratings.csv")
embeddings = np.load("data/03_primary/train_embeddings.npy")
book_2_row = json.load(open("data/03_primary/train_id_2_row.json", mode="r"))
train_loader = get_train_dataloader(
    df_books, df_ratings, embeddings, book_2_row, device, batch_size=512, shuffle=True
)
start_training(
    "AmazonBooksModelv3",
    loss,
    train_loader,
    model_params,
    optimizer_params,
    num_epochs=5,
    save_checkpoint_every=200,
)
