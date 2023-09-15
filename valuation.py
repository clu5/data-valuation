"""
Main implmentation for diversity and relevance measures for data valuation
"""
import collections
import math

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torch.utils.data import (ConcatDataset, DataLoader, Dataset, Subset,
                              WeightedRandomSampler)
from torchvision import models, transforms


def covariance(X, normalize=True):
    """
    Computes covariance matrix
    """
    if normalize:
        X = X - X.mean(0)
    norm = 1 / X.shape[0]
    cov = X.T @ X
    return norm * cov


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


def compute_X_tilde_and_counts(X, omega=0.1):
    """
    From https://github.com/ZhaoxuanWu/VolumeBased-DataValuation/blob/main/volume.py

    Compresses the original feature matrix X to  X_tilde with the specified omega.

    Returns:
       X_tilde: compressed np.ndarray
       cubes: a dictionary of cubes with the respective counts in each dcube
    """
    X = torch.tensor(X)

    D = X.shape[1]

    # assert 0 < omega <= 1, "omega must be within range [0,1]."

    m = math.ceil(1.0 / omega)  # number of intervals for each dimension

    cubes = collections.Counter()  # a dictionary to store the freqs
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

        """
        if cube_key in Omega:

            # Implementing mean() to compute the average of all rows which fall in the cube

            Omega[cube_key] = Omega[cube_key] * (1 - 1.0 / cubes[cube_key]) + 1.0 / cubes[cube_key] * x
            # Omega[cube_key].append(x)
        else:
             Omega[cube_key] = x
        """
    X_tilde = torch.stack(
        [torch.stack(list(value)).mean(axis=0) for key, value in Omega.items()]
    )

    # X_tilde = stack(list(Omega.values()))

    return X_tilde, cubes


def compute_robust_volumes(X_tildes, dcube_collections):
    """
    From https://github.com/ZhaoxuanWu/VolumeBased-DataValuation/blob/main/volume.py
    """
    N = sum([len(X_tilde) for X_tilde in X_tildes])
    alpha = 1.0 / (10 * N)  # it means we set beta = 10
    # print("alpha is :{}, and (1 + alpha) is :{}".format(alpha, 1 + alpha))

    # volumes, volume_all = compute_volumes(X_tildes, d=X_tildes[0].shape[1])
    volumes = compute_volumes(X_tildes, d=X_tildes[0].shape[1])
    robust_volumes = np.zeros_like(volumes)
    for i, (volume, hypercubes) in enumerate(zip(volumes, dcube_collections)):
        rho_omega_prod = 1.0
        for cube_index, freq_count in hypercubes.items():
            # if freq_count == 1: continue # volume does not monotonically increase with omega
            # commenting this if will result in volume monotonically increasing with omega
            rho_omega = (1 - alpha ** (freq_count + 1)) / (1 - alpha)

            rho_omega_prod *= rho_omega

        robust_volumes[i] = (volume * rho_omega_prod).round(3)
    return robust_volumes


def get_volume(cov, omega=0.1):
    """
    From https://github.com/ZhaoxuanWu/VolumeBased-DataValuation/blob/main/volume.py
    """

    X_tilde, cubes = compute_X_tilde_and_counts(cov, omega=omega)
    vol = compute_robust_volumes([X_tilde], [cubes])
    return vol[0]


def div_rel_func(
    buyer_eig_vals, buyer_eig_vecs, seller_cov, threshold=1e-2, n_components=2
):
    buyer_vals = buyer_eig_vals[:n_components]
    # buyer_vecs = buyer_eig_vecs[:, :n_components]
    buyer_vecs = buyer_eig_vecs[:n_components]
    seller_vals = np.linalg.norm(seller_cov @ buyer_vecs.T, axis=0)

    # Diversity based on difference of values
    div_components = np.abs(buyer_vals - seller_vals) / np.maximum(
        buyer_vals, seller_vals
    )

    #  Relevance
    rel_components = np.minimum(buyer_vals, seller_vals) / np.maximum(
        buyer_vals, seller_vals
    )

    # only include directions with value above this threshold
    keep_mask = buyer_vals >= threshold

    div = np.prod(np.where(keep_mask, div_components, 1)) ** (1 / keep_mask.sum())
    rel = np.prod(np.where(keep_mask, rel_components, 1)) ** (1 / keep_mask.sum())
    return rel, div  # , keep_mask.sum()


def fit_buyer(buyer_features, n_components=2, svd_solver="randomized", whiten=True):
    """
    Compute PCA for buyer's data
    """
    X_b = buyer_features.float()
    X_b -= X_b.mean(0)
    buyer_cov = np.cov(X_b, rowvar=False)
    pca = PCA(n_components=n_components, svd_solver=svd_solver, whiten=whiten)
    pca.fit(X_b)
    buyer_values = pca.explained_variance_  # eigenvalues
    buyer_components = pca.components_  # eigenvectors
    return pca, buyer_cov, buyer_values, buyer_components


def project_seller(seller_features, buyer_pca):
    """
    Projects seller's data onto buyer's principal components
    """
    X_s = np.array(seller_features)
    X_s -= X_s.mean(0)
    seller_cov = np.cov(X_s, rowvar=False)
    proj_seller_cov = buyer_pca.transform(seller_cov)
    return seller_cov, proj_seller_cov


def get_valuation(
    buyer_values,
    buyer_components,
    seller_cov,
    proj_seller_cov,
    threshold=0.1,
    n_components=2,
    omega=0.3,
):
    """
        buyer_values: buyer's eigenvalues from PCA
        buyer_components: buyer's principal components
        seller_cov: seller's covariance matrix
        proj_seller_cov: seller's covariance matrix projected onto buyer's components
        threshold: only include values in valuation computation above this threshold
        n_components: number of principal components to consider
        omega: parameter in [0, 1] for volume-based diversity that controls duplication robustness
    """
    rel, div = div_rel_func(
        buyer_values,
        buyer_components,
        seller_cov,
        threshold=threshold,
        n_components=n_components,
    )
    vol = get_volume(proj_seller_cov, omega=omega)
    return rel, div, vol


def get_relevance(buyer_pca, seller_data, threshold=1e-2):
    buyer_vals = buyer_pca.explained_variance_
    seller_vals = np.linalg.norm(np.cov(buyer_pca.transform(seller_data).T), axis=0)

    # alternative implmentation TODO: cleanup or delete
    # buyer_vecs = buyer_eig_vecs[:n_components]
    # seller_vals = np.linalg.norm(np.cov(buyer_pca.transform(seller_data).T), axis=0)
    # seller_vals = np.linalg.norm(seller_cov @ buyer_pca.components_.T, axis=0)

    rel_components = np.minimum(buyer_vals, seller_vals) / np.maximum(
        buyer_vals, seller_vals
    )
    keep_mask = buyer_vals >= threshold
    rel = np.prod(np.where(keep_mask, rel_components, 1)) ** (1 / keep_mask.sum())
    return rel
