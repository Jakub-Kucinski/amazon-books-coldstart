import torch
import torch.optim as optim
from pytorch_metric_learning import losses
from TODO import Dataloader
from tqdm import tqdm

import src.amazon_books_coldstart.models as models
from src.amazon_books_coldstart.config import OUTPUT_SIZE


def train_single_epoch(model, loss_func, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, data2) in enumerate(train_loader):
        data = [feature.to(device) for feature in data]
        data2 = [feature.to(device) for feature in data2]
        optimizer.zero_grad()
        embeddings = model(data)
        embeddings2 = model(data2)
        loss = loss_func(embeddings, embeddings2)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print("Epoch {} Iteration {}: Loss = {}".format(epoch, batch_idx, loss))


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
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    loss_func = wrap_loss_function(loss)

    model_constructor = getattr(models, model_name)
    model = model_constructor(**model_params).to(device)

    optimizer = optim.Adam(model.parameters(), **optimizer_params)

    for epoch in tqdm(range(1, num_epochs + 1)):
        train_single_epoch(model, loss_func, device, train_loader, optimizer, epoch)


loss = losses.SupConLoss()
model_params = {
    "d_description": OUTPUT_SIZE,
    "n_authors": 82462,
    "n_publishers": None,
    "n_categories": None,
    "d_authors": 128,
    "d_publishers": 128,
    "d_categories": 128,
    "linear1_size": 1024,
    "linear2_size": 256,
}
optimizer_params = {"lr": 0.001}
train_loader = Dataloader()
start_training("AmazonBooksModelv3", loss, train_loader, model_params, optimizer_params)
