import collections
import math
from PIL import Image
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler, ConcatDataset

from tqdm import tqdm
from operator import itemgetter
fst = lambda x: itemgetter(0)(list(x) if hasattr(x, '__iter__') else x)
snd = lambda x: itemgetter(1)(list(x) if hasattr(x, '__iter__') else x)


def make_imagenet_csv(data_dir):
    index_to_names = {
        index: [n.strip().replace(' ', '_') for n in names.split(',')] 
        for index, names in eval(open(data_dir / 'imagenet1000_clsidx_to_labels.txt').read()).items()
    }

    id_to_class = {row.split()[0]: row.split()[2] for row in open(data_dir / 'map_clsloc.txt').readlines()}

    id_to_index = {}
    for _id, _cls in id_to_class.items():
        if _id == 'n03126707': # crane (machine)
            id_to_index[_id] = 517
        elif _id == 'n02012849': # crane (bird)
            id_to_index[_id] = 134
        elif _id == 'n03710637': # maillot
            id_to_index[_id] = 638
        elif _id == 'n03710721': # maillot tank suit
            id_to_index[_id] = 639
        elif _id == 'n03773504': # missile
            id_to_index[_id] = 657
        elif _id == 'n04008634': # projectile missile
            id_to_index[_id] = 744
        elif _id == 'n02445715': # skunk
            id_to_index[_id] = 361
        elif _id == 'n02443114': # polecat
            id_to_index[_id] = 358 
        else:
            for k, v in index_to_names.items():
                if _cls in v:
                    id_to_index[_id] = k
                
    df = pd.DataFrame.from_dict(id_to_index, orient='index', columns=['class_index'])
    df['class_name'] = df.index.map(id_to_class)
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
        extensions=['.jpg', '.jpeg', '.JPEG', '.JPG'], 
        transforms=None, 
        test_transforms=None, 
        eval_mode=False, 
        domain='imagenet-val-set',
    ):
        if domain is None:
            self.domain = image_dir.stem
        else:
            self.domain = domain
        self.class_df = class_df
        self.image_dir = image_dir
        self.images = []
        for ext in extensions:
            if self.domain == 'imagenet-val-set':
                self.images.extend(list(image_dir.glob(f'[!.]*{ext}'))) 
            else:
                self.images.extend(list(image_dir.glob(f'[!.]*/[!.]*{ext}'))) 
        self.len = len(self.images)
        print(f'found {self.len} images')
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
        if self.domain == 'imagenet-val-set':
            uid = fname.stem.split('_')[3]
        elif self.domain == 'imagenetv2-matched-frequency-format-val':
            index = int(fname.parent.stem)
            uid = self.class_df[self.class_df.class_index == index].index.values[0]
        else:
            uid = fname.parent.name
        return self.class_df.loc[uid].class_index
    
def subsample(ds, n=100, test_split=0.5): 
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

    
def create_model(arch='eff-b1',  num_classes = 1000):
    if arch == 'eff-b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    elif arch == 'eff-b1':
        model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
    elif arch == 'eff-b2':
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    elif arch == 'eff-b3':
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    elif arch == 'eff-b4':
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    elif arch == 'eff-b5':
        model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1)
        
    model = model.cuda()
    return model


def create_optimizer(model, lr = 1e-5):
    opt = optim.AdamW(model.parameters(),lr=lr)
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
                    print(f'{i} finished\t train={train_loss[-1]:.4f} val={val_loss[-1]:.4f}')
                    if save_name is not None:
                        torch.save(model.state_dict(), 'models/' + save_name)
                    
    if save_name is not None:
        torch.save(model.state_dict(), 'models/' + save_name)

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
    optimizer = torch.optim.LBFGS([temperature], lr=0.001, max_iter=10000, line_search_fn="strong_wolfe")

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
    res = np.sum(np.where(p!=0, p*np.log(p/q), 0))
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


def covariance(scores, normalize=True):
    # scores = scores.T
    if normalize:
        scores = scores - scores.mean(0)
    norm = 1 / scores.shape[0]
    cov = scores.T @ scores
    # cov = scores @ scores.T
    return norm * cov


def svd(covariance_matrix):
    eig_val, eig_vec = np.linalg.eig(covariance_matrix)
    eig_val = eig_val.real
    return eig_val, eig_vec


def compute_volume(cov, epsilon=1e-8):
    return np.sqrt(np.linalg.det(cov) + epsilon)
    
    
def compute_volumes(datasets, d=1):
    # d = datasets[0].shape[1]
    # for i in range(len(datasets)):
        # datasets[i] = datasets[i].reshape(-1 ,d)

    # X = np.concatenate(datasets, axis=0).reshape(-1, d)
    volumes = np.zeros(len(datasets))
    for i, dataset in enumerate(datasets):
        volumes[i] = np.sqrt(np.linalg.det( dataset.T @ dataset ) + 1e-8)

    # volume_all = np.sqrt(np.linalg.det(X.T @ X) + 1e-8).round(3)
    return volumes#, volume_all


def compute_X_tilde_and_counts(X, omega=0.1):
    """
    Compresses the original feature matrix X to  X_tilde with the specified omega.

    Returns:
       X_tilde: compressed np.ndarray
       cubes: a dictionary of cubes with the respective counts in each dcube
    """
    X = torch.tensor(X)
    
    D = X.shape[1]

    # assert 0 < omega <= 1, "omega must be within range [0,1]."

    m = math.ceil(1.0 / omega) # number of intervals for each dimension

    cubes = collections.Counter() # a dictionary to store the freqs
    # key: (1,1,..)  a d-dimensional tuple, each entry between [0, m-1]
    # value: counts

    Omega = collections.defaultdict(list)
    # Omega = {}
    
    min_ds = torch.min(X, axis=0).values

    # a dictionary to store cubes of not full size
    for x in X:
        cube = []
        for d, xd in enumerate(x - min_ds):
            d_index = math.floor(xd / omega)
            cube.append(d_index)

        cube_key = tuple(cube)
        cubes[cube_key] += 1

        Omega[cube_key].append(x)

        '''
        if cube_key in Omega:
            
            # Implementing mean() to compute the average of all rows which fall in the cube
            
            Omega[cube_key] = Omega[cube_key] * (1 - 1.0 / cubes[cube_key]) + 1.0 / cubes[cube_key] * x
            # Omega[cube_key].append(x)
        else:
             Omega[cube_key] = x
        '''
    X_tilde = torch.stack([torch.stack(list(value)).mean(axis=0) for key, value in Omega.items()])

    # X_tilde = stack(list(Omega.values()))

    return X_tilde, cubes



def compute_robust_volumes(X_tildes, dcube_collections):
        
    N = sum([len(X_tilde) for X_tilde in X_tildes])
    alpha = 1.0 / (10 * N) # it means we set beta = 10
    # print("alpha is :{}, and (1 + alpha) is :{}".format(alpha, 1 + alpha))

    # volumes, volume_all = compute_volumes(X_tildes, d=X_tildes[0].shape[1])
    volumes = compute_volumes(X_tildes, d=X_tildes[0].shape[1])
    robust_volumes = np.zeros_like(volumes)
    for i, (volume, hypercubes) in enumerate(zip(volumes, dcube_collections)):
        rho_omega_prod = 1.0
        for cube_index, freq_count in hypercubes.items():
            # if freq_count == 1: continue # volume does not monotonically increase with omega
            # commenting this if will result in volume monotonically increasing with omega
            rho_omega = (1 - alpha**(freq_count + 1)) / (1 - alpha)

            rho_omega_prod *= rho_omega

        robust_volumes[i] = (volume * rho_omega_prod).round(3)
    return robust_volumes

def get_volume(cov, omega=0.1):
    X_tilde, cubes = compute_X_tilde_and_counts(cov, omega=omega)
    vol = compute_robust_volumes([X_tilde], [cubes])
    return vol[0]


def div_rel_func(buyer_eig_vals, buyer_eig_vecs, seller_cov, threshold=1e-2, n_components=2):
    buyer_vals = buyer_eig_vals[:n_components]
    # buyer_vecs = np.abs(buyer_eig_vecs[:n_components])
    buyer_vecs = buyer_eig_vecs[:n_components]
    seller_vals = np.linalg.norm(seller_cov @ buyer_vecs.T, axis=0)

    # Diversity
    div_components = np.abs(buyer_vals - seller_vals) / np.maximum(buyer_vals, seller_vals)

    #  Relevance
    rel_components = np.minimum(buyer_vals, seller_vals) / np.maximum(buyer_vals, seller_vals)

    keep_mask = buyer_vals >= threshold

    div = np.prod(np.where(keep_mask, div_components, 1)) ** (1 / keep_mask.sum())
    rel = np.prod(np.where(keep_mask, rel_components, 1)) ** (1 / keep_mask.sum())
    return rel, div #, keep_mask.sum()

def fit_buyer(buyer_features, n_components=2, svd_solver='randomized', whiten=True):
    X_b = buyer_features.float()
    X_b -= X_b.mean(0)
    buyer_cov = np.cov(X_b.T)
    pca = PCA(n_components=n_components, svd_solver=svd_solver, whiten=whiten)
    pca.fit(X_b)
    buyer_values = pca.explained_variance_  # eigenvalues
    buyer_components = pca.components_  # eigenvectors
    return pca, buyer_cov, buyer_values, buyer_components


def project_seller(seller_features, buyer_pca):
    X_s = np.array(seller_features)
    X_s -= X_s.mean(0)
    seller_cov = np.cov(X_s.T)
    proj_seller_cov = buyer_pca.transform(seller_cov)
    return seller_cov, proj_seller_cov
    
    
def get_valuation(buyer_values, buyer_components, seller_cov, proj_seller_cov, threshold=0.1, n_components=2, omega=0.3):
    rel, div = div_rel_func(buyer_values, buyer_components, seller_cov, threshold=threshold, n_components=n_components)
    vol = get_volume(proj_seller_cov, omega=omega)
    return rel, div, vol
    
    
def get_relevance(buyer_pca, seller_data, threshold=1e-2):
    # buyer_vecs = buyer_eig_vecs[:n_components]
    buyer_vals = buyer_pca.explained_variance_
    # seller_vals = np.linalg.norm(np.cov(buyer_pca.transform(seller_data).T), axis=0)
    seller_vals = np.linalg.norm(np.cov(buyer_pca.transform(seller_data).T), axis=0)
    # seller_vals = np.linalg.norm(seller_cov @ buyer_pca.components_.T, axis=0)
    rel_components = np.minimum(buyer_vals, seller_vals) / np.maximum(buyer_vals, seller_vals)
    keep_mask = buyer_vals >= threshold
    rel = np.prod(np.where(keep_mask, rel_components, 1)) ** (1 / keep_mask.sum())
    return rel