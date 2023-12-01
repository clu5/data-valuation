"""
Main implmentation for diversity and relevance measures for data valuation
"""
import collections
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple
from scipy.linalg import cosm

from vendi_score import vendi
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from torch.utils.data import (ConcatDataset, DataLoader, Dataset, Subset,
                              WeightedRandomSampler)
from torchvision import models, transforms


def covariance(X, normalize=True):
    """
    Computes covariance matrix
    """
    if normalize:
        X = X - X.mean(0)
    n = X.shape[0]
    norm = (n - 1)
    gram = np.dot(X.T, X)
    return gram / norm

def svd(covariance_matrix):
    eig_val, eig_vec = np.linalg.eig(covariance_matrix)
    eig_val = eig_val.real
    return eig_val, eig_vec

def compute_volume(cov, epsilon=1e-8):
    return np.sqrt(np.linalg.det(cov) + epsilon)

def compute_volumes(datasets, d=1):
    volumes = np.zeros(len(datasets))
    for i, dataset in enumerate(datasets):
        volumes[i] = np.sqrt(np.linalg.det(dataset.T @ dataset) + 1e-8)
    return volumes

def compute_projected_volumes(datasets, projection, d=1):
    volumes = np.zeros(len(datasets))
    for i, dataset in enumerate(datasets):
        volumes[i] = np.sqrt(np.linalg.det(dataset.T @ dataset) + 1e-8)
    return volumes


def compute_X_tilde_and_counts(X: torch.Tensor, omega: float = 0.1) -> Tuple[np.ndarray, Dict]:
    """
    https://github.com/opendataval/opendataval/blob/main/opendataval/dataval/volume/rvs.py

    Compresses the original feature matrix X to  X_tilde with the specified omega.

    Returns
    -------
    np.ndarray
       compressed form of X as a d-cube 
    dict[tuple, int]
       cubes: a dictionary of cubes with the respective counts in each dcube
    """
    assert 0 < omega <= 1, "omega must be within range [0,1]."

    # Dictionary to store frequency for each cube
    cubes = collections.Counter()
    omega_dict = collections.defaultdict(list)
    min_ds = np.min(X, axis=0)

    # a dictionary to store cubes of not full size
    for entry in X:
        cube_key = tuple(math.floor(ent.item() / omega) for ent in entry - min_ds)
        cubes[cube_key] += 1
        omega_dict[cube_key].append(entry)

    X_tilde = np.stack([np.mean(value, axis=0) for value in omega_dict.values()])
    return X_tilde, cubes


def compute_robust_volumes(X_tilde: np.ndarray, hypercubes: dict[tuple, int]):

    """
    https://github.com/opendataval/opendataval/blob/main/opendataval/dataval/volume/rvs.py
    """
    alpha = 1.0 / (10 * len(X_tilde))  # it means we set beta = 10

    flat_data = X_tilde.reshape(-1, X_tilde.shape[1])
    (sign, volume) = np.linalg.slogdet(np.dot(flat_data.T, flat_data))
    robustness_factor = 1.0

    for freq_count in hypercubes.values():
        robustness_factor *= (1 - alpha ** (freq_count + 1)) / (1 - alpha)

    return sign, volume, robustness_factor


def get_volume(X, omega=0.1, norm=False):
    """
    From https://github.com/ZhaoxuanWu/VolumeBased-DataValuation/blob/main/volume.py
    """
    X_tilde, cubes = compute_X_tilde_and_counts(X, omega=omega)
    sign, vol, robustness_factor = compute_robust_volumes(X_tilde, cubes)
    robust_vol = robustness_factor * vol
    return dict(robust_vol=robust_vol, sign=sign, vol=vol, robustness_factor=robustness_factor) 


def cluster_valuation(buyer_data, seller_data, k_means=None, n_clusters=10, n_components=25):
    if k_means is None:
        k_means = KMeans(n_clusters=n_clusters, n_init='auto')
        k_means.fit(buyer_data)
    buyer_clusters = {k: buyer_data[k_means.predict(buyer_data) == k] for k in range(n_clusters)}
    seller_clusters = {k: seller_data[k_means.predict(seller_data) == k] for k in range(n_clusters)}
    cluster_rel = {}
    cluster_vol = {}
    # for j in tqdm(range(n_clusters)):
    for j in range(n_clusters):
        cluster_pca = PCA(n_components=n_components, svd_solver='randomized', whiten=False)
        cluster_pca.fit(buyer_clusters[j])
        ws = []
        rs = []
        vs = []
        for i in range(n_clusters):
            if seller_clusters[i].shape[0] == 0 or seller_clusters[i].shape[0] == 1:
                ws.append(0)
                rs.append(0)
                vs.append(0)
            else:
                ws.append(seller_clusters[i].shape[0] / seller_data.shape[0])
                rs.append(valuation.get_relevance(cluster_pca, seller_clusters[i]))
                # vs.append(valuation.get_volume(np.cov(cluster_pca.transform(seller_clusters[i]).T)))
                vs.append(valuation.get_volume(cluster_pca.transform(seller_clusters[i])))
        cluster_rel[j] = np.average(rs, weights=ws)
        cluster_vol[j] = np.average(vs, weights=ws)
    buyer_weights = [v.shape[0] / buyer_data.shape[0] for v in buyer_clusters.values()]
    # print(buyer_weights)
    rel = np.average(list(cluster_rel.values()), weights=buyer_weights)
    vol = np.average(list(cluster_vol.values()), weights=buyer_weights)
    return rel, vol


def get_value(
    buyer_data, seller_data, threshold=0.1, n_components=10, 
    verbose=False, norm_volume=True, omega=0.1, dtype = np.float32,
    only_return_vol = True, decomp=None, decomp_kwargs={},
    use_smallest_components = False,
    include_vendi_score = False,
):
    start_time = time.perf_counter()
    buyer_data = np.array(buyer_data, dtype=dtype)
    seller_data = np.array(seller_data, dtype=dtype)
    seller_cov = np.cov(seller_data, rowvar=False)
    buyer_cov = np.cov(buyer_data, rowvar=False)
    buyer_val, buyer_vec = np.linalg.eig(buyer_cov)
    order = np.argsort(buyer_val)[::-1]
    sorted_buyer_val = buyer_val[order]
    sorted_buyer_vec = buyer_vec[:, order]
    if use_smallest_components:
        slice_index = np.s_[-n_components:]
    else:
        slice_index = np.s_[:n_components]
    
    buyer_values = sorted_buyer_val.real[slice_index]
    buyer_components = sorted_buyer_vec.real[:, slice_index]
    if verbose:
        print(f'{slice_index=}')
        print(f'{buyer_values=}')
        print(buyer_components.shape)
    if decomp is not None:
        D = decomp(**decomp_kwargs)
        # D = decomp(n_components=n_components, **decomp_kwargs)
        D.fit(buyer_data)
        D.mean_ = np.zeros(seller_data.shape[1]) # dummy mean 
        seller_values = np.linalg.norm(D.transform(seller_cov), axis=0)
    else:
        seller_values = np.linalg.norm(seller_cov @ buyer_components, axis=0)
    if verbose:
        print(seller_values.shape)
        print(f'{seller_values=}')
        
    # only include directions with value above this threshold
    keep_mask = buyer_values >= threshold
    
    if verbose: 
        print(f'{keep_mask.nonzero()[0].shape[0]=}')
        
    C = np.maximum(buyer_values, seller_values)  
    div_components = np.abs(buyer_values - seller_values) / C
    rel_components = np.minimum(buyer_values, seller_values) / C
    div = np.prod(np.where(keep_mask, div_components, 1)) ** (1 / max(1, keep_mask.sum()))
    rel = np.prod(np.where(keep_mask, rel_components, 1)) ** (1 / max(1, keep_mask.sum()))
    if verbose:
        print(np.prod(seller_values))

    if norm_volume:
        Norm = Normalizer(norm='l2')
        seller_data = Norm.fit_transform(seller_data)
    if decomp is not None:
        vol = get_volume(D.transform(seller_data), omega=omega)
    else:
        vol = get_volume(seller_data @ buyer_components, omega=omega)
    if only_return_vol:
        vol = vol['robust_vol']

    if include_vendi_score:
        if decomp is not None:
            vs = vendi.score_dual(D.transform(seller_data), normalize=True)
        else:
            vs = vendi.score_dual(seller_data, normalize=True)

    # Compute the cosine similarity and L2 Distance
    buyer_mean = np.mean(buyer_cov, axis=0)
    seller_mean = np.mean(seller_cov, axis=0)
    cos = np.dot(buyer_mean, seller_mean) / (np.linalg.norm(buyer_mean) * np.linalg.norm(seller_mean))
    l2 = - np.linalg.norm(buyer_mean - seller_mean) # negative since we want the ordering to match

    end_time = time.perf_counter()
    if verbose:
        print('time', end_time - start_time)
    return dict(cosine=cos, diversity=div, l2=l2, relevance=rel, volume=vol, vendi=vs)