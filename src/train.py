import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import torchvision
from torchvision.models import (EfficientNet_B1_Weights,
                                EfficientNet_B2_Weights,
                                EfficientNet_B3_Weights,
                                EfficientNet_B4_Weights,
                                EfficientNet_B5_Weights, efficientnet_b1,
                                efficientnet_b2, efficientnet_b3,
                                efficientnet_b4, efficientnet_b5)
from torchvision.transforms import transforms
from tqdm import tqdm

plt.style.use("bmh")
from pathlib import Path

import pandas as pd
import wilds
from PIL import Image
from sklearn import metrics
from wilds.common.data_loaders import get_eval_loader, get_train_loader

DATASETS = [
    "iwildcam",
    # 'camelyon',
    # 'rxrx1',
    "fmow",
]
MODELS = [
    "efficientnet-b1",
    "efficientnet-b2",
    "efficientnet-b3",
    "efficientnet-b4",
    "efficientnet-b5",
]


@click.command()
@click.option("--dataset", default="iwildcam")
@click.option("--arch", default="efficientnet-b3", type=click.Choice(MODELS))
@click.option("--save_dir", default="/u/luchar/data-valuation/models/")
@click.option("--pretrain", is_flag=True, default=True)
@click.option("--epochs", default=10)
@click.option("--lr", default=1e-4)
@click.option("--debug", is_flag=True, default=False)
def main(dataset, arch, save_dir, pretrain, epochs, lr, debug):
    dataset = wilds.get_dataset(dataset=dataset, root_dir="/u/luchar/data/wilds/")

    num_class = dataset._n_classes
    splits = dataset._split_dict.keys()

    _transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    data_splits = {
        split: dataset.get_subset(split, transform=_transforms) for split in splits
    }

    # subset = lambda ds, indices: data.Subset(ds, indices)
    # if debug:
    # first_128 = lambda ds: subset(ds=ds, indices=128)
    # data_splits = {k: first_128(v) for k, v in data_splits.items()}

    for split, dataset in data_splits.items():
        print(split, f"{len(dataset)}")

    loader_params = dict(batch_size=16, num_workers=8, pin_memory=True)

    get_loader = lambda split: get_train_loader if split == "train" else get_eval_loader
    loader_splits = {
        split: get_loader(split)("standard", dataset, **loader_params)
        for split, dataset in data_splits.items()
    }

    model = get_model(arch, pretrain, num_class)
    model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model = fit(
        model,
        loader_splits["train"],
        criterion,
        optimizer,
        epochs=epochs,
        val_loader=loader_splits["id_val"],
        debug=debug,
    )

    torch.save(
        {
            "dataset": dataset,
            "architecture": arch,
            "epochs": epochs,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        Path(save_dir)
        / f'{dataset}-{arch}-{epochs}-{"pretrain" if pretrain else ""}.pt',
    )


def get_model(arch, pretrain, num_class):
    if arch == "efficientnet-b1":
        weights = EfficientNet_B1_Weights.IMAGENET1K_V1
        model = efficientnet_b1
    elif arch == "efficientnet-b2":
        weights = EfficientNet_B2_Weights.IMAGENET1K_V1
        model = efficientnet_b2
    elif arch == "efficientnet-b3":
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        model = efficientnet_b3
    elif arch == "efficientnet-b4":
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1
        model = efficientnet_b4
    elif arch == "efficientnet-b5":
        weights = EfficientNet_B5_Weights.IMAGENET1K_V1
        model = efficientnet_b5
    else:
        raise ValueError("Architecture not recognized")

    model = model(weights=weights) if pretrain else model()
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_class)
    return model


def fit(model, loader, criterion, optimizer, epochs=10, val_loader=None, debug=False):
    for e in range(epochs):
        print("\nepoch", e)
        model.train()
        for i, (img, label, _) in enumerate(loader):
            if debug and i > 64:
                break
            logit = model(img.cuda())
            loss = criterion(logit, label.cuda())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print("\t", i, end="")
        if not debug and val_loader is not None:
            model.eval()
            print("validating...", end="")
            val_scores, val_labels = get_scores(model, val_loader)
            accuracy(val_scores, val_labels)

    return model


def get_scores(model, loader):
    model.eval()
    scores, labels = [], []
    for i, (x, y, _) in enumerate(loader):
        scores.append(model(x.cuda()).detach().cpu())
        labels.append(y)
    return torch.softmax(torch.concat(scores), 1), torch.concat(labels)


def accuracy(scores, labels):
    acc = metrics.accuracy_score(scores.argmax(1), labels)
    print(f"accuracy: {acc:.1%}")
    return acc


if __name__ == "__main__":
    main()
