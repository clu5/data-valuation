import copy
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import click
from tqdm import tqdm
import torch
import torch.utils.data as data
import torchvision
from torchvision.transforms import transforms
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('bmh')
import pandas as pd
from sklearn import metrics
from pathlib import Path
from PIL import Image
import wilds
from wilds.common.data_loaders import get_train_loader, get_eval_loader
import train


def get_q_hat(calibration_scores, labels, alpha=0.05):
    if not isinstance(calibration_scores, torch.Tensor):
        calibration_scores = torch.tensor(calibration_scores)

    n = calibration_scores.shape[0]

    #  sort scores and returns values and index that would sort classes
    values, indices = calibration_scores.sort(dim=1, descending=True)

    #  sum up all scores cummulatively and return to original index order
    cum_scores = values.cumsum(1).gather(1, indices.argsort(1))[range(n), labels]

    #  get quantile with small correction for finite sample sizes
    q_hat = torch.quantile(cum_scores, np.ceil((n + 1) * (1 - alpha)) / n)
    clipped_q_hat = torch.minimum(q_hat, torch.tensor(1-1e-9))

    return clipped_q_hat


def make_prediction_set(scores, q_hat):
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores)
    assert q_hat < 1, 'q_hat must be below 1'

    n = scores.shape[0]

    values, indices = scores.sort(dim=1, descending=True)

    #  number of each confidence prediction set to acheive coverage
    set_sizes = (values.cumsum(1) > q_hat).int().argmax(dim=1)

    confidence_sets = [indices[i][0:(set_sizes[i] + 1)] for i in range(n)]

    return [x.tolist() for x in confidence_sets]

get_sizes = lambda x: np.array([len(x_i) for x_i in x])
# for param in model.parameters():
    # param.requires_grad = False

@click.command()
@click.option('-cp', '--checkpoint_path', default='models/rxrx1-b3.pt')
@click.option('-dr', '--data_root', default='/u/luchar/data/wilds/')
# @click.option('-lp', '--linear-probing')
@click.option('-d', '--debug', is_flag=True, default=False)
def main(checkpoint_path, data_root, debug):
    checkpoint_path = Path(checkpoint_path)
    _dataset, arch = checkpoint_path.stem.split('-')
    checkpoint = torch.load(checkpoint_path)
    dataset = wilds.get_dataset(
        dataset=_dataset,
        root_dir=data_root,
    )
    num_classes = dataset._n_classes

    model = getattr(models, f'efficientnet_{arch}')()
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )
    model.load_state_dict(checkpoint['model'])
    model = model.cuda()

    compose_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    id_train_data = dataset.get_subset('train', transform=compose_transforms)
    id_test_data = dataset.get_subset('id_test', transform=compose_transforms)
    od_val_data = dataset.get_subset('val', transform=compose_transforms)
    od_test_data = dataset.get_subset('test', transform=compose_transforms)

    subset = lambda ds, indices: data.Subset(ds, indices)
    loader = lambda ds: data.DataLoader(ds, batch_size=16, num_workers=8, pin_memory=True, shuffle=False)

    cal_split = int(len(id_test_data)*0.1)
    id_cal_data = subset(id_test_data, range(cal_split))
    id_test_data = subset(id_test_data, range(cal_split, len(id_test_data)))


    if debug:
        N = 1000
        id_cal_data = subset(id_cal_data, range(min(N, len(id_cal_data))))
        id_test_data = subset(id_test_data, range(min(N, len(id_test_data))))
        od_val_data = subset(od_val_data, range(min(N, len(od_val_data))))
        od_test_data = subset(od_test_data, range(min(N, len(od_test_data))))

    id_test_scores, id_test_labels = train.get_scores(model, loader(id_test_data))
    id_cal_scores, id_cal_labels = train.get_scores(model, loader(id_cal_data))
    od_val_scores, od_val_labels = train.get_scores(model, loader(od_val_data))
    od_test_scores, od_test_labels = train.get_scores(model, loader(od_test_data))

    print('original id accuracy')
    train.accuracy(id_test_scores, id_test_labels)
    print('original od accuracy')
    train.accuracy(od_test_scores, od_test_labels)

    alpha = 0.10
    q_hat = get_q_hat(id_cal_scores, id_cal_labels, alpha=alpha)
    id_test_pset = make_prediction_set(id_test_scores, q_hat)
    od_val_pset = make_prediction_set(od_val_scores, q_hat)
    od_test_pset = make_prediction_set(od_test_scores, q_hat)
    id_test_size = get_sizes(id_test_pset)
    od_val_size = get_sizes(od_val_pset)
    od_test_size = get_sizes(od_test_pset)

    #budget_per_round = int(0.3 * len(od_val_data))

    q1 = np.quantile(od_test_size, 0.25)
    q2 = np.quantile(od_test_size, 0.5)
    q3 = np.quantile(od_test_size, 0.75)

    model = model.cpu()
    torch.cuda.empty_cache()

    criterion = torch.nn.CrossEntropyLoss()
    lr = 1e-4
    fit_kwargs = dict(criterion=criterion, epochs=20 if not debug else 2, debug=debug, patience=5)

    bought_index = [i for i, size in enumerate(od_test_size) if size < q1]
    random_index = np.random.randint(0, len(od_test_size), size=len(bought_index))
    bought_data = subset(od_test_data, bought_index)
    random_data = subset(od_test_data, random_index)
    model_bought = copy.deepcopy(model).cuda()
    model_random = copy.deepcopy(model).cuda()
    opt_bought = torch.optim.Adam(model_bought.parameters(), lr=lr)
    opt_random = torch.optim.Adam(model_random.parameters(), lr=lr)
    model_bought = train.fit(model_bought, loader(bought_data), val_loader=loader(id_test_data), optimizer=opt_bought, **fit_kwargs)
    model_random = train.fit(model_random, loader(random_data), val_loader=loader(id_test_data), optimizer=opt_random, **fit_kwargs)
    id_bought_scores, id_bought_labels = train.get_scores(model_bought, loader(id_test_data))
    id_random_scores, id_random_labels = train.get_scores(model_random, loader(id_test_data))
    print('\nq1 finetuned id accuracy (bought) = ', end=' ')
    train.accuracy(id_bought_scores, id_bought_labels)
    print('\nq1 finetuned id accuracy (random) = ', end=' ')
    train.accuracy(id_random_scores, id_random_labels)
    od_bought_scores, od_bought_labels = train.get_scores(model_bought, loader(od_test_data))
    od_random_scores, od_random_labels = train.get_scores(model_random, loader(od_test_data))
    print('\nq1 finetuned od accuracy (bought) = ', end=' ')
    train.accuracy(od_bought_scores, od_bought_labels)
    print('\nq1 finetuned od accuracy (random) = ', end=' ')
    train.accuracy(od_random_scores, od_random_labels)

    del model_bought, model_random, opt_bought, opt_random
    torch.cuda.empty_cache()

    bought_index = [i for i, size in enumerate(od_test_size) if q1 < size < q2]
    random_index = np.random.randint(0, len(od_test_size), size=len(bought_index))
    bought_data = subset(od_test_data, bought_index)
    random_data = subset(od_test_data, random_index)
    model_bought = copy.deepcopy(model).cuda()
    model_random = copy.deepcopy(model).cuda()
    opt_bought = torch.optim.Adam(model_bought.parameters(), lr=lr)
    opt_random = torch.optim.Adam(model_random.parameters(), lr=lr)
    model_bought = train.fit(model_bought, loader(bought_data), val_loader=loader(id_test_data), optimizer=opt_bought, **fit_kwargs)
    model_random = train.fit(model_random, loader(random_data), val_loader=loader(id_test_data), optimizer=opt_random, **fit_kwargs)
    id_bought_scores, id_bought_labels = train.get_scores(model_bought, loader(id_test_data))
    id_random_scores, id_random_labels = train.get_scores(model_random, loader(id_test_data))
    print('\nq2 finetuned id accuracy (bought) = ', end=' ')
    train.accuracy(id_bought_scores, id_bought_labels)
    print('\nq2 finetuned id accuracy (random) = ', end=' ')
    train.accuracy(id_random_scores, id_random_labels)
    od_bought_scores, od_bought_labels = train.get_scores(model_bought, loader(od_test_data))
    od_random_scores, od_random_labels = train.get_scores(model_random, loader(od_test_data))
    print('\nq2 finetuned od accuracy (bought) = ', end=' ')
    train.accuracy(od_bought_scores, od_bought_labels)
    print('\nq2 finetuned od accuracy (random) = ', end=' ')
    train.accuracy(od_random_scores, od_random_labels)

    del model_bought, model_random, opt_bought, opt_random
    torch.cuda.empty_cache()

    bought_index = [i for i, size in enumerate(od_test_size) if q2 < size < q3]
    random_index = np.random.randint(0, len(od_test_size), size=len(bought_index))
    bought_data = subset(od_test_data, bought_index)
    random_data = subset(od_test_data, random_index)
    model_bought = copy.deepcopy(model).cuda()
    model_random = copy.deepcopy(model).cuda()
    opt_bought = torch.optim.Adam(model_bought.parameters(), lr=lr)
    opt_random = torch.optim.Adam(model_random.parameters(), lr=lr)
    model_bought = train.fit(model_bought, loader(bought_data), val_loader=loader(id_test_data), optimizer=opt_bought, **fit_kwargs)
    model_random = train.fit(model_random, loader(random_data), val_loader=loader(id_test_data), optimizer=opt_random, **fit_kwargs)
    id_bought_scores, id_bought_labels = train.get_scores(model_bought, loader(id_test_data))
    id_random_scores, id_random_labels = train.get_scores(model_random, loader(id_test_data))
    print('\nq3 finetuned id accuracy (bought) = ', end=' ')
    train.accuracy(id_bought_scores, id_bought_labels)
    print('\nq3 finetuned id accuracy (random) = ', end=' ')
    train.accuracy(id_random_scores, id_random_labels)
    od_bought_scores, od_bought_labels = train.get_scores(model_bought, loader(od_test_data))
    od_random_scores, od_random_labels = train.get_scores(model_random, loader(od_test_data))
    print('\nq3 finetuned od accuracy (bought) = ', end=' ')
    train.accuracy(od_bought_scores, od_bought_labels)
    print('\nq3 finetuned od accuracy (random) = ', end=' ')
    train.accuracy(od_random_scores, od_random_labels)

    del model_bought, model_random, opt_bought, opt_random
    torch.cuda.empty_cache()

    bought_index = [i for i, size in enumerate(od_test_size) if q3 < size ]
    random_index = np.random.randint(0, len(od_test_size), size=len(bought_index))
    bought_data = subset(od_test_data, bought_index)
    random_data = subset(od_test_data, random_index)
    model_bought = copy.deepcopy(model).cuda()
    model_random = copy.deepcopy(model).cuda()
    opt_bought = torch.optim.Adam(model_bought.parameters(), lr=lr)
    opt_random = torch.optim.Adam(model_random.parameters(), lr=lr)
    model_bought = train.fit(model_bought, loader(bought_data), val_loader=loader(id_test_data), optimizer=opt_bought, **fit_kwargs)
    model_random = train.fit(model_random, loader(random_data), val_loader=loader(id_test_data), optimizer=opt_random, **fit_kwargs)
    id_bought_scores, id_bought_labels = train.get_scores(model_bought, loader(id_test_data))
    id_random_scores, id_random_labels = train.get_scores(model_random, loader(id_test_data))
    print('\nq4 finetuned id accuracy (bought) = ', end=' ')
    train.accuracy(id_bought_scores, id_bought_labels)
    print('\nq4 finetuned id accuracy (random) = ', end=' ')
    train.accuracy(id_random_scores, id_random_labels)
    od_bought_scores, od_bought_labels = train.get_scores(model_bought, loader(od_test_data))
    od_random_scores, od_random_labels = train.get_scores(model_random, loader(od_test_data))
    print('\nq4 finetuned od accuracy (bought) = ', end=' ')
    train.accuracy(od_bought_scores, od_bought_labels)
    print('\nq4 finetuned od accuracy (random) = ', end=' ')
    train.accuracy(od_random_scores, od_random_labels)

if __name__ == '__main__':
    main()
