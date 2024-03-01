"""
Helper functions for modeling training and evaluation
"""

import collections
import math
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from PIL import Image
from sklearn.decomposition import PCA
from torch.utils.data import (ConcatDataset, DataLoader, Dataset, Subset,
                              WeightedRandomSampler)
from torchvision import models, transforms
from tqdm import tqdm

fst = lambda x: itemgetter(0)(list(x) if hasattr(x, "__iter__") else x)
snd = lambda x: itemgetter(1)(list(x) if hasattr(x, "__iter__") else x)


def create_model(arch="eff-b1", num_classes=1000):
    if arch == "eff-b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
    elif arch == "eff-b1":
        model = models.efficientnet_b1(
            weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1
        )
    elif arch == "eff-b2":
        model = models.efficientnet_b2(
            weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1
        )
    elif arch == "eff-b3":
        model = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )
    elif arch == "eff-b4":
        model = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )
    elif arch == "eff-b5":
        model = models.efficientnet_b5(
            weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1
        )

    model = model.cuda()
    return model


def create_optimizer(model, lr=1e-5):
    opt = optim.AdamW(model.parameters(), lr=lr)
    return opt


def update(model, opt, crit, train_loader, val_loader=None, epochs=10, save_name=None):
    train_loss = []
    val_loss = []
    for i in range(epochs):
        model.train()
        epoch_train_loss = []
        epoch_val_loss = []
        for j, (x, y) in tqdm(enumerate(train_loader)):
            logits = model(x.cuda())
            loss = crit(logits, y.cuda())
            loss.backward()
            opt.step()
            epoch_train_loss.append(loss.detach().cpu().item())
        train_loss.append(np.mean(epoch_train_loss))

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for j, (x, y) in tqdm(enumerate(val_loader)):
                    logits = model(x.cuda())
                    loss = crit(logits, y.cuda())
                    epoch_val_loss.append(loss.detach().cpu().item())
                val_loss.append(np.mean(epoch_val_loss))

                if i % 10 == 0:
                    print(
                        f"{i} finished\t train={train_loss[-1]:.4f} val={val_loss[-1]:.4f}"
                    )
                    if save_name is not None:
                        torch.save(model.state_dict(), "models/" + save_name)

    if save_name is not None:
        torch.save(model.state_dict(), "models/" + save_name)

    if val_loader is None:
        return train_loss, val_loss

    return train_loss


def evaluate(model, dataset, num_batches=None, batch_size=32):
    model = model.cuda()
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    with torch.no_grad():
        all_logits = []
        for i, (x, *_) in tqdm(enumerate(loader), total=num_batches):
            if num_batches is not None and i > num_batches:
                break
            logits = model(x.cuda())
            all_logits.append(logits.detach().cpu().numpy())

    return np.concatenate(all_logits)


def temp_scale(logits, labels, plot=True):
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    temperature = torch.nn.Parameter(torch.ones(1))
    criterion = torch.nn.CrossEntropyLoss()

    # Removing strong_wolfe line search results in jump after 50 epochs
    optimizer = torch.optim.LBFGS(
        [temperature], lr=0.001, max_iter=10000, line_search_fn="strong_wolfe"
    )

    temps = []
    losses = []

    def _eval():
        loss = criterion(torch.div(logits, temperature), labels)
        loss.backward()
        temps.append(temperature.item())
        losses.append(loss.item())
        return loss

    optimizer.step(_eval)

    if plot:
        print("Final T_scaling factor: {:.2f}".format(temperature.item()))

    if plot:
        plt.figure(figsize=(9, 2))
        plt.subplot(121)
        plt.plot(list(range(len(temps))), temps)

        plt.subplot(122)
        plt.plot(list(range(len(losses))), losses)
        plt.show()
    return temperature.detach()


def AC_metric(scores):
    scores = np.asarray(scores)
    return scores.max(1).mean()


def DOC_metric(retrain_scores, base_scores):
    retrain_scores = np.asarray(retrain_scores)
    base_scores = np.asarray(base_scores)
    assert retrain_scores.shape == base_scores.shape
    m = retrain_scores.shape[0]

    metric = 0
    for i in range(m):
        metric += abs(retrain_scores[i].max() - base_scores[i].max())

    return metric / m


def KL_divergence(p, q):
    res = np.sum(np.where(p != 0, p * np.log(p / q), 0))
    return res


def f_metric(retrain_scores, base_scores, divergence=KL_divergence):
    retrain_scores = np.asarray(retrain_scores)
    base_scores = np.asarray(base_scores)
    assert retrain_scores.shape == base_scores.shape
    m = retrain_scores.shape[0]

    metric = 0
    for i in range(m):
        metric += divergence(retrain_scores[i], base_scores[i])

    return metric / m
