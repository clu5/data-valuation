import argparse
import json
import math
import os
from collections import defaultdict
from importlib import reload
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_absolute_error)
from tqdm import tqdm

plt.style.use("bmh")
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import (CIFAR10, EMNIST, MNIST, QMNIST, SVHN,
                                  FashionMNIST)
from torchvision.models import (EfficientNet_B0_Weights,
                                EfficientNet_B1_Weights,
                                EfficientNet_B2_Weights,
                                EfficientNet_B3_Weights,
                                EfficientNet_B4_Weights, ResNet18_Weights,
                                efficientnet_b0, efficientnet_b1,
                                efficientnet_b2, efficientnet_b3,
                                efficientnet_b4, resnet18)
from torchvision.transforms import Resize

import models
import valuation


def resize(x, size=(32, 32)):
    """
    Resize a tensor or array to the specified size.

    Parameters:
        x (torch.Tensor or np.ndarray): The input tensor or array to resize.
        size (tuple, optional): The target size as (width, height). Default is (32, 32).

    Returns:
        torch.Tensor: The resized tensor.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return Resize(size)(x)


def make_mnistm(data_dir):
    """
    Create or load MNIST-M dataset from the specified data directory.

    If the dataset exists in the specified location, it loads from there. Otherwise, it creates the dataset
    and saves it to the specified location.

    Parameters:
        data_dir (Path): The directory where the dataset is or will be stored.

    Returns:
        tuple: A tuple containing:
            - mnistm_data (torch.Tensor): The data for MNIST-M.
            - mnistm_targets (torch.Tensor): The corresponding labels for MNIST-M.
    """
    if (data_dir / "mnistm_data.pt").exists():
        mnistm_data = torch.load(data_dir / "mnistm_data.pt")
        mnistm_targets = torch.load(data_dir / "mnistm_targets.pt")
        return mnistm_data, mnistm_targets
    print("making MNIST-M data".center(40, "-"))
    mnistm_dir = data_dir / "mnist_m"
    df = pd.read_csv(
        mnistm_dir / "mnist_m_train_labels.txt",
        sep=" ",
        header=None,
        names=["image", "label"],
    )
    mnistm_data = resize(
        np.moveaxis(
            np.stack(
                [
                    np.array(Image.open(mnistm_dir / "mnist_m_train" / img))
                    for img in tqdm(df.image.values)
                ]
            ),
            -1,
            1,
        )
    )
    mnistm_targets = torch.tensor(df.label.values)
    torch.save(mnistm_data, data_dir / "mnistm_data.pt")
    torch.save(mnistm_targets, data_dir / "mnistm_targets.pt")
    print("finished MNIST-M data".center(40, "-"))
    return mnist_data, mnist_targets


def make_dida(data_dir):
    """
    Create or load the DIDA dataset from the specified data directory.

    If the dataset exists in the specified location, it loads from there. Otherwise, it creates the dataset
    and saves it to the specified location.

    Parameters:
        data_dir (Path): The directory where the dataset is or will be stored.

    Returns:
        tuple: A tuple containing:
            - dida_data (torch.Tensor): The data for the DIDA dataset.
            - dida_targets (torch.Tensor): The corresponding labels for the DIDA dataset.
    """
    if (data_dir / "dida_data.pt").exists():
        dida_data = torch.load(data_dir / "dida_data.pt")
        dida_targets = torch.load(data_dir / "dida_targets.pt")
        return dida_data, dida_targets
    print("making DIDA data".center(40, "-"))
    dida_dir = data_dir / "DIDA-70k"
    dida_paths = {int(p.stem): list(p.glob("*.jpg")) for p in (dida_dir.glob("[!.]*"))}
    dida_data_dict = {}
    dida_targets_dict = {}
    for label, image_paths in tqdm(dida_paths.items()):
        images, targets = [], []
        targets = []
        for p in tqdm(image_paths):
            images.append(resize(np.moveaxis(np.array(Image.open(p)), -1, 0)))
            targets.append(label)
        dida_data_dict[label] = torch.stack(images)
        dida_targets_dict[label] = torch.tensor(targets)
    dida_data = torch.tensor(torch.cat(list(dida_data_dict.values())))
    dida_targets = torch.cat(list(dida_targets_dict.values()))
    torch.save(dida_data, data_dir / "dida_data.pt")
    torch.save(dida_targets, data_dir / "dida_targets.pt")
    print("finished DIDA data".center(40, "-"))
    return dida_data, dida_targets


def make_data(data_dir):
    """
    Create or download a set of datasets and return them in a dictionary.

    This function creates or downloads multiple datasets, including MNIST, EMNIST, QMNIST, SVHN, DIDA,
    and MNIST-M, resizing them to a common shape and structure.

    Parameters:
        data_dir (Path): The directory to store or load the datasets.

    Returns:
        dict: A dictionary with dataset names as keys and corresponding data and targets as values.
    """
    mnist = MNIST(root=data_dir, train=True, download=True)
    emnist = EMNIST(root=data_dir, split="digits", train=True, download=True)
    qmnist = QMNIST(root=data_dir, what="test50k", train=True, download=True)
    svhn = SVHN(root=data_dir, split="train", download=True)
    dida_data, dida_targets = make_dida(data_dir)
    mnistm_data, mnistm_targets = make_mnistm(data_dir)
    return {
        "MNIST": {
            "data": resize(mnist.data.unsqueeze(1).repeat(1, 3, 1, 1)),
            "targets": mnist.targets,
        },
        "EMNIST": {
            "data": resize(
                torch.fliplr(torch.rot90(emnist.data, k=1, dims=[1, 2]))
                .unsqueeze(1)
                .repeat(1, 3, 1, 1)
            ),
            "targets": emnist.targets,
        },
        "QMNIST": {
            "data": resize(qmnist.data.unsqueeze(1).repeat(1, 3, 1, 1)),
            "targets": qmnist.targets[:, 0],
        },
        "SVHN": {
            "data": resize(svhn.data),
            "targets": torch.tensor(svhn.labels),
        },
        "DIDA": {
            "data": dida_data,
            "targets": dida_targets,
        },
        "MNIST-M": {
            "data": mnistm_data,
            "targets": mnistm_targets,
        },
    }


def embed(data):
    """
    Embed the data using a pre-trained ResNet18 model.

    Parameters:
        data (torch.Tensor): The data to embed, expected to be normalized to [0, 1].

    Returns:
        torch.Tensor: The embedded output from the ResNet18 model.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).cuda()
    model.eval()
    loader = DataLoader(TensorDataset(data / 255), batch_size=32)
    outputs = []
    for batch in loader:
        outputs.append(model(batch[0].cuda()).detach().cpu())
    return torch.cat(outputs)


def get_valuation(buyer_pca, seller):
    """
    Calculate relevance and volume for valuation between a buyer and seller.

    Parameters:
        buyer_pca (PCA): The PCA model fit on the buyer's data.
        seller (torch.Tensor): The seller's data to evaluate.

    Returns:
        tuple: A tuple containing:
            - relevance (float): The relevance score.
            - volume (float): The volume score.
    """
    rel = valuation.get_relevance(buyer_pca, seller)
    # vol = valuation.get_volume(np.cov(buyer_pca.transform(seller).T))
    vol = valuation.get_volume(buyer_pca.transform(seller).T)
    return rel, max(vol, 1e-5)


def make_data_loader(data, targets, batch_size=32):
    """
    Create a DataLoader from given data and targets.

    Parameters:
        data (torch.Tensor or np.ndarray): The data to load.
        targets (torch.Tensor or np.ndarray): The targets associated with the data.
        batch_size (int, optional): The size of each batch. Default is 32.

    Returns:
        DataLoader: A DataLoader instance with the given data and targets.
    """
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)
    return DataLoader(TensorDataset(data / 255, targets), batch_size=batch_size)


def train(buyer_data, buyer_targets, seller_data, seller_targets, args):
    """
    Train two models (classification and regression) using given buyer and seller data, and return metrics and losses.

    Parameters:
        buyer_data (torch.Tensor): Data for the buyer model.
        buyer_targets (torch.Tensor): Targets for the buyer model.
        seller_data (torch.Tensor): Data for the seller model.
        seller_targets (torch.Tensor): Targets for the seller model.
        args (argparse.Namespace): Command-line arguments containing training settings.

    Returns:
        dict: A dictionary containing:
            - "metrics": A dictionary with keys 'accuracy' and 'MAE' representing accuracy and mean absolute error.
            - "losses": A dictionary with keys 'classification' and 'regression' representing classification and regression losses.
    """
    buyer_loader = make_data_loader(buyer_data, buyer_targets)
    seller_loader = make_data_loader(seller_data, seller_targets)
    cls_model = models.CNN().cuda()
    reg_model = models.CNN(regressor=True).cuda()
    cls_opt = torch.optim.SGD(cls_model.parameters(), lr=args.learning_rate)
    reg_opt = torch.optim.SGD(reg_model.parameters(), lr=args.learning_rate)
    cls_loss = models.fit(
        cls_model, seller_loader, cls_opt, epochs=1 if args.debug else args.epochs
    )
    reg_loss = models.fit(
        reg_model,
        seller_loader,
        reg_opt,
        epochs=1 if args.debug else args.epochs,
        classification=False,
    )
    losses = {"classification": cls_loss, "regression": reg_loss}
    cls_pred, reg_pred, targets = [], [], []
    for x, y in buyer_loader:
        cls_pred.append(cls_model(x.cuda()).detach().cpu())
        reg_pred.append(reg_model(x.cuda()).detach().cpu())
        targets.append(y)
    targets = torch.cat(targets)
    metrics = {
        "accuracy": accuracy_score(targets, torch.cat(cls_pred).argmax(1)),
        "MAE": mean_absolute_error(targets, torch.cat(reg_pred)),
    }
    return {"metrics": metrics, "losses": losses}


def main(args):
    """
    Main function to perform the valuation and plot results for multiple datasets.

    Parameters:
        args (argparse.Namespace): Command-line arguments containing configuration settings.
    """
    print(args)
    data_dir = Path(args.data_dir)
    datasets = make_data(data_dir)
    print("loaded datasets".center(40, "-"))
    embeddings = {}
    for k, v in datasets.items():
        print(k, "number of samples", len(v["data"]), len(v["targets"]))
        embeddings[k] = embed(v["data"][: args.num_valuation])
    print("finished embeddings ".center(40, "-"))
    valuations = defaultdict(dict)
    pca = PCA(n_components=args.num_components, svd_solver="randomized", whiten=False)
    for buyer, v in embeddings.items():
        pca.fit(v)
        for seller, w in embeddings.items():
            relevance, diversity = get_valuation(pca, w)
            valuations[buyer][seller] = {
                "relevance": relevance,
                "diversity": diversity,
            }
    print("finished valuations".center(40, "-"))
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    if not args.debug:
        with open(results_dir / "valuations.json", "w") as f:
            json.dump(dict(valuations), f, indent=4)
    for buyer, results in valuations.items():
        plt.figure(figsize=(8, 5))
        plt.xlabel("Relvance", fontsize=20)
        plt.ylabel("Diversity", fontsize=20)
        plt.xlim(0, 1.1)
        plt.tick_params(labelsize=14)
        plt.title(f"Buyer: {buyer}", fontsize=20, pad=12)
        for seller, value in results.items():
            plt.scatter(
                value["relevance"], value["diversity"], s=300, marker="o", label=seller
            )
        plt.legend(fontsize=20, bbox_to_anchor=(1.5, 1))
        if not args.debug:
            plt.savefig(results_dir / f"{buyer}-valuation.png", bbox_inches="tight")
    print("finished plots".center(40, "-"))
    return
    performances = defaultdict(dict)
    for buyer, v in tqdm(datasets.items()):
        for seller, w in tqdm(datasets.items()):
            if buyer == seller:
                continue
            buyer_data = v["data"][: args.num_train]
            buyer_targets = v["targets"][: args.num_train]
            seller_data = w["data"][: args.num_train]
            seller_targets = w["targets"][: args.num_train]
            assert (
                buyer_data.shape[0] == buyer_targets.shape[0]
            ), f"{buyer} mismatch {buyer_data.shape[0]} != {buyer_targets.shape[0]}"
            assert (
                seller_data.shape[0] == seller_targets.shape[0]
            ), f"{seller} mismatch {seller_data.shape[0]} != {seller_targets.shape[0]}"
            performances[buyer][seller] = train(
                buyer_data, buyer_targets, seller_data, seller_targets, args
            )
            if args.debug:
                break
        if args.debug:
            break
    if not args.debug:
        with open(results_dir / "performances.json", "w") as f:
            json.dump(dict(performances), f, indent=4, default=float)
    print("finished training".center(40, "-"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="digits.py",
        description="Runs digits experiment",
        epilog="Data valuation",
    )
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument(
        "-nc",
        "--num_components",
        default=5,
        help="number of buyer components to use in valuation",
    )
    parser.add_argument(
        "-nv",
        "--num_valuation",
        default=1000,
        help="number of points per dataset to value",
    )
    parser.add_argument(
        "-nt",
        "--num_train",
        default=50000,
        help="number of points per dataset to train",
    )
    parser.add_argument("-e", "--epochs", default=30, help="number of training epochs")
    parser.add_argument("-lr", "--learning_rate", default=1e-3, help="learning rate")
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()
    main(args)
