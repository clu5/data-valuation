import argparse
import json
import os
import math
from importlib import reload
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error
import matplotlib
import matplotlib.pyplot as plt; plt.style.use('bmh')
from torchvision.transforms import Resize
import pandas as pd
import torch
from torchvision.datasets import EMNIST, MNIST, SVHN, FashionMNIST, CIFAR10, QMNIST
from torchvision.models import (
    resnet18, ResNet18_Weights,
    efficientnet_b0, EfficientNet_B0_Weights, 
    efficientnet_b1, EfficientNet_B1_Weights, 
    efficientnet_b2, EfficientNet_B2_Weights, 
    efficientnet_b3, EfficientNet_B3_Weights, 
    efficientnet_b4, EfficientNet_B4_Weights, 
)
from torch.utils.data import TensorDataset, DataLoader
import valuation, models


def resize(x, size=(32, 32)): 
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return Resize(size)(x)


def make_mnistm(data_dir):
    if (data_dir / 'mnistm_data.pt').exists():
        mnistm_data = torch.load(data_dir / 'mnistm_data.pt')
        mnistm_targets = torch.load(data_dir / 'mnistm_targets.pt')
        return mnistm_data, mnistm_targets
    print('making MNIST-M data'.center(40, '-'))
    mnistm_dir = data_dir / 'mnist_m'
    df = pd.read_csv(mnistm_dir / 'mnist_m_train_labels.txt', sep=' ', header=None, names=['image', 'label'])
    mnistm_data = resize(np.moveaxis(np.stack([np.array(Image.open(mnistm_dir / 'mnist_m_train' / img)) for img in tqdm(df.image.values)]), -1, 1))
    mnistm_targets = torch.tensor(df.label.values)
    torch.save(mnistm_data, data_dir / 'mnistm_data.pt')
    torch.save(mnistm_targets, data_dir / 'mnistm_targets.pt')
    print('finished MNIST-M data'.center(40, '-'))
    return mnist_data, mnist_targets


def make_dida(data_dir):
    if (data_dir / 'dida_data.pt').exists():
        dida_data = torch.load(data_dir / 'dida_data.pt')
        dida_targets = torch.load(data_dir / 'dida_targets.pt')
        return dida_data, dida_targets
    print('making DIDA data'.center(40, '-'))
    dida_dir = data_dir / 'DIDA-70k'
    dida_paths = {int(p.stem): list(p.glob('*.jpg')) for p in (dida_dir.glob('[!.]*'))}
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
    torch.save(dida_data, data_dir / 'dida_data.pt')
    torch.save(dida_targets, data_dir / 'dida_targets.pt')
    print('finished DIDA data'.center(40, '-'))
    return dida_data, dida_targets


def make_data(data_dir):
    mnist = MNIST(root=data_dir, train=True, download=True)
    emnist = EMNIST(root=data_dir, split='digits', train=True, download=True)
    qmnist = QMNIST(root=data_dir, what='test50k', train=True, download=True)
    svhn = SVHN(root=data_dir, split='train', download=True)
    dida_data, dida_targets = make_dida(data_dir)
    mnistm_data, mnistm_targets = make_mnistm(data_dir)
    return {
        'MNIST': {
            'data': resize(mnist.data.unsqueeze(1).repeat(1, 3, 1, 1)),
            'targets': mnist.targets,
        },
        'EMNIST': {
            'data': resize(torch.fliplr(torch.rot90(emnist.data, k=1, dims=[1, 2])).unsqueeze(1).repeat(1, 3, 1, 1)),
            'targets': emnist.targets,
        },
        'QMNIST': {
            'data': resize(qmnist.data.unsqueeze(1).repeat(1, 3, 1, 1)),
            'targets': qmnist.targets[:, 0],
        },
        'SVHN': {
            'data': resize(svhn.data),
            'targets': torch.tensor(svhn.labels),
        },
        'DIDA': {
            'data': dida_data,
            'targets': dida_targets,
        },
        'MNIST-M': {
            'data': mnistm_data,
            'targets': mnistm_targets,
        },
    }


def embed(data):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).cuda()
    model.eval()
    loader = DataLoader(TensorDataset(data / 255), batch_size=32)
    outputs = []
    for batch in loader:
        outputs.append(model(batch[0].cuda()).detach().cpu())
    return torch.cat(outputs)


def get_valuation(buyer_pca, seller):
    rel = valuation.get_relevance(buyer_pca, seller)
    vol = valuation.get_volume(np.cov(buyer_pca.transform(seller).T))
    return rel, max(vol, 1e-5)


def make_data_loader(data, targets, batch_size=32):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)
    return DataLoader(TensorDataset(data / 255, targets), batch_size=batch_size)


def train(buyer_data, buyer_targets, seller_data, seller_targets, args):
    buyer_loader = make_data_loader(buyer_data, buyer_targets)
    seller_loader = make_data_loader(seller_data, seller_targets)
    cls_model = models.CNN().cuda()
    reg_model = models.CNN(regressor=True).cuda()
    cls_opt = torch.optim.SGD(cls_model.parameters(), lr=args.learning_rate)
    reg_opt = torch.optim.SGD(reg_model.parameters(), lr=args.learning_rate)
    cls_loss = models.fit(cls_model, seller_loader, cls_opt, epochs=1 if args.debug else args.epochs)
    reg_loss = models.fit(reg_model, seller_loader, reg_opt, epochs=1 if args.debug else args.epochs, classification=False)
    losses = {'classification': cls_loss, 'regression': reg_loss}
    cls_pred, reg_pred, targets = [], [], []
    for x, y in buyer_loader:
        cls_pred.append(cls_model(x.cuda()).detach().cpu())
        reg_pred.append(reg_model(x.cuda()).detach().cpu())
        targets.append(y)
    targets = torch.cat(targets)
    metrics = {
        'accuracy': accuracy_score(targets, torch.cat(cls_pred).argmax(1)),
        'MAE': mean_absolute_error(targets, torch.cat(reg_pred)),
    }
    return {'metrics': metrics, 'losses': losses}


def main(args):
    print(args)
    data_dir = Path(args.data_dir)
    datasets = make_data(data_dir)
    print('loaded datasets'.center(40, '-'))
    embeddings = {}
    for k, v in datasets.items(): 
        print(k, 'number of samples', len(v['data']), len(v['targets']))
        embeddings[k] = embed(v['data'][:args.num_valuation])
    print('finished embeddings '.center(40, '-'))
    valuations = defaultdict(dict)
    pca = PCA(n_components=args.num_components, svd_solver='randomized', whiten=False)
    for buyer, v in embeddings.items():
        pca.fit(v)
        for seller, w in embeddings.items():
            relevance, diversity = get_valuation(pca, w)
            valuations[buyer][seller] = {
                'relevance': relevance,
                'diversity': diversity,
            }
    print('finished valuations'.center(40, '-'))
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    if not args.debug:
        with open(results_dir / 'valuations.json', 'w') as f:
            json.dump(dict(valuations), f, indent=4)
    for buyer, results in valuations.items():
        plt.figure(figsize=(6, 6))
        plt.xlabel('Relvance', fontsize=20)
        plt.ylabel('Diversity', fontsize=20)
        plt.xlim(0, 1.1)
        plt.tick_params(labelsize=14)
        plt.title(f'Buyer: {buyer}', fontsize=20, pad=12)
        for seller, value in results.items():
            plt.scatter(value['relevance'], value['diversity'], s=300, marker='o', label=seller)
        plt.legend(fontsize=20, bbox_to_anchor=(1.7, 1))
        if not args.debug:
            plt.savefig(results_dir / f'{buyer}-valuation.png', bbox_inches='tight')
    print('finished plots'.center(40, '-'))
    performances = defaultdict(dict)
    for buyer, v in tqdm(datasets.items()):
        for seller, w in tqdm(datasets.items()):
            if buyer == seller:
                continue
            buyer_data = v['data'][:args.num_train]
            buyer_targets = v['targets'][:args.num_train]
            seller_data = w['data'][:args.num_train]
            seller_targets = w['targets'][:args.num_train]
            assert buyer_data.shape[0] == buyer_targets.shape[0], f'{buyer} mismatch {buyer_data.shape[0]} != {buyer_targets.shape[0]}'
            assert seller_data.shape[0] == seller_targets.shape[0], f'{seller} mismatch {seller_data.shape[0]} != {seller_targets.shape[0]}'
            performances[buyer][seller] = train(buyer_data, buyer_targets, seller_data, seller_targets, args)
            if args.debug:
                break
        if args.debug:
            break
    if not args.debug:
        with open(results_dir / 'performances.json', 'w') as f:
            json.dump(dict(performances), f, indent=4, default=float)
    print('finished training'.center(40, '-'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='digits.py',
        description='Runs digits experiment',
        epilog='Data valuation',
    )
    parser.add_argument('--data_dir', default='../data')
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('-nc', '--num_components', default=5, help='number of buyer components to use in valuation')
    parser.add_argument('-nv', '--num_valuation', default=1000, help='number of points per dataset to value')
    parser.add_argument('-nt', '--num_train', default=50000, help='number of points per dataset to train')
    parser.add_argument('-e', '--epochs', default=30, help='number of training epochs')
    parser.add_argument('-lr', '--learning_rate', default=1e-3, help='learning rate')
    parser.add_argument('-d', '--debug', action='store_true')
    args = parser.parse_args()
    main(args)