import collections
import math
from copy import deepcopy
from operator import itemgetter

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torch.utils.data import (ConcatDataset, DataLoader, Dataset, Subset,
                              WeightedRandomSampler)
from torchvision import models, transforms
from torchvision.datasets import (CIFAR10, EMNIST, MNIST, SVHN, USPS,
                                  FashionMNIST)
from tqdm import tqdm


def make_imagenet_csv(data_dir):
    """
    Maps ImageNet UID to class names and indices
    """
    index_to_names = {
        index: [n.strip().replace(" ", "_") for n in names.split(",")]
        for index, names in eval(
            open(data_dir / "imagenet1000_clsidx_to_labels.txt").read()
        ).items()
    }

    id_to_class = {
        row.split()[0]: row.split()[2]
        for row in open(data_dir / "map_clsloc.txt").readlines()
    }

    id_to_index = {}
    for _id, _cls in id_to_class.items():
        if _id == "n03126707":  # crane (machine)
            id_to_index[_id] = 517
        elif _id == "n02012849":  # crane (bird)
            id_to_index[_id] = 134
        elif _id == "n03710637":  # maillot
            id_to_index[_id] = 638
        elif _id == "n03710721":  # maillot tank suit
            id_to_index[_id] = 639
        elif _id == "n03773504":  # missile
            id_to_index[_id] = 657
        elif _id == "n04008634":  # projectile missile
            id_to_index[_id] = 744
        elif _id == "n02445715":  # skunk
            id_to_index[_id] = 361
        elif _id == "n02443114":  # polecat
            id_to_index[_id] = 358
        else:
            for k, v in index_to_names.items():
                if _cls in v:
                    id_to_index[_id] = k

    df = pd.DataFrame.from_dict(id_to_index, orient="index", columns=["class_index"])
    df["class_name"] = df.index.map(id_to_class)
    df.class_name = df.class_name.map(lambda s: s.lower())
    return df


class simple_DS(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = self.images[i]
        if self.transforms is not None:
            image = self.transforms(image)
        if self.labels is not None:
            label = self.labels[i]
            return image, label
        else:
            return image


class DS(Dataset):
    def __init__(
        self,
        image_dir,
        class_df=None,
        extensions=[".jpg", ".jpeg", ".JPEG", ".JPG"],
        transforms=None,
        test_transforms=None,
        eval_mode=False,
        domain="imagenet-val-set",
    ):
        if domain is None:
            self.domain = image_dir.stem
        else:
            self.domain = domain
        self.class_df = class_df
        self.image_dir = image_dir
        self.images = []
        for ext in extensions:
            if self.domain == "imagenet-val-set":
                self.images.extend(list(image_dir.glob(f"[!.]*{ext}")))
            else:
                self.images.extend(list(image_dir.glob(f"[!.]*/[!.]*{ext}")))
        self.len = len(self.images)
        print(f"found {self.len} images")
        self.transforms = transforms
        self.test_transforms = test_transforms
        self.eval_mode = eval_mode

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        fname = self.images[i]
        image = self.load_image(fname)
        if self.eval_mode:
            if self.test_transforms is not None:
                image = self.test_transforms(image)
        else:
            if self.transforms is not None:
                image = self.transforms(image)

        if self.class_df is not None:
            label = self.extract_label(fname)
        else:
            label = -1  # dummy target

        return image, label

    def load_image(self, fname):
        return Image.open(fname)

    def extract_label(self, fname):
        if self.domain == "imagenet-val-set":
            uid = fname.stem.split("_")[3]
        elif self.domain == "imagenetv2-matched-frequency-format-val":
            index = int(fname.parent.stem)
            uid = self.class_df[self.class_df.class_index == index].index.values[0]
        else:
            uid = fname.parent.name
        return self.class_df.loc[uid].class_index


def subsample(ds, n=100, test_split=0.5):
    """
    Randomly subsets or partition a dataset into validation and test
    """
    index = np.random.choice(np.arange(len(ds)), n)
    if test_split > 0:
        split = round(n * (1 - test_split))
        val_index = index[:split]
        test_index = index[split:]
        print(len(test_index))
        val_ds = Subset(deepcopy(ds), val_index)
        test_ds = Subset(deepcopy(ds), test_index)
        test_ds.dataset.eval_mode = True
        return val_ds, test_ds
    else:
        return Subset(ds, index)


def subset(dataset, classes):
    """
        Patitions MNIST/FashionMNIST dataset by class labels
    """
    class_mask = {
        k: np.array(dataset.targets) == v for k, v in dataset.class_to_idx.items()
    }
    dummy = np.array([False] * len(dataset))
    for c in classes:
        dummy ^= class_mask[c]
    return dummy
