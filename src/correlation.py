import json
import random
import os
import argparse
from pathlib import Path
from collections import defaultdict
from itertools import chain
from functools import partial

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, homogeneity_score
from tqdm import tqdm
from joblib import Parallel, delayed

import utils
import valuation
from trainer import Trainer


def sample_by_class(metadata, num_sellers, num_samples, alpha=0.1):
    """Sample seller data according to dirichlet class distribution"""
    class_ids = metadata['class_id'].values
    class_counts = np.bincount(class_ids.astype(int))
    class_prop = class_counts / class_counts.sum()
    
    # Generate dirichlet distributions for each seller
    class_weights = np.random.dirichlet(alpha + class_prop, size=num_sellers)
    
    # Create sample weights based on class distribution
    sample_weights = {i: v[class_ids] for i, v in enumerate(class_weights)}
    sample_weights = {k: v / v.sum() for k, v in sample_weights.items()}
    
    # Sample indices for each seller
    rang = np.arange(len(metadata))
    seller_indices = {
        k: np.random.choice(rang, size=num_samples, replace=False, p=v)
        for k, v in sample_weights.items()
    }
    
    return seller_indices

def sample_by_domain(metadata_dict, num_sellers, num_samples):
    """
    Sample seller data from different domains (e.g., ImageNet-C with different corruptions).
    
    Args:
        metadata_dict: Dictionary with corruption_type -> metadata dataframe
        num_sellers: Number of sellers to generate
        num_samples: Number of samples per seller
    
    Returns:
        Dictionary of seller indices
    """
    corruption_types = list(metadata_dict.keys())
    
    # Assign corruptions to sellers
    seller_corruptions = {}
    for i in range(num_sellers):
        # Each seller gets a different corruption type or a mix
        if i < len(corruption_types):
            seller_corruptions[i] = [corruption_types[i]]
        else:
            # For additional sellers, create mixtures of corruption types
            num_corruption_mix = random.randint(2, min(4, len(corruption_types)))
            seller_corruptions[i] = random.sample(corruption_types, num_corruption_mix)
    
    seller_indices = {}
    for seller_id, corruptions in seller_corruptions.items():
        # Calculate samples per corruption
        samples_per_corruption = num_samples // len(corruptions)
        
        indices = []
        for corruption in corruptions:
            metadata = metadata_dict[corruption]
            corruption_indices = range(len(metadata))
            
            # Sample randomly from this corruption
            selected = np.random.choice(
                corruption_indices, 
                size=min(samples_per_corruption, len(corruption_indices)), 
                replace=False
            )
            
            # Add corruption identifier to the indices
            indices.extend([(corruption, idx) for idx in selected])
        
        # If we don't have enough samples, sample more from random corruptions
        while len(indices) < num_samples and len(corruptions) > 0:
            corruption = random.choice(corruptions)
            metadata = metadata_dict[corruption]
            corruption_indices = range(len(metadata))
            
            if len(corruption_indices) > 0:
                selected = np.random.choice(corruption_indices, size=1, replace=False)
                indices.append((corruption, selected[0]))
        
        seller_indices[seller_id] = indices[:num_samples]
    
    return seller_indices

def sample_by_demographics(metadata, num_sellers, num_samples, attributes=None):
    """
    Sample seller data based on demographic attributes (gender, race, age).
    
    Args:
        metadata: DataFrame with demographic information
        num_sellers: Number of sellers to generate
        num_samples: Number of samples per seller
        attributes: List of demographic attributes to consider (default: all)
    
    Returns:
        Dictionary of seller indices
    """
    if attributes is None:
        attributes = ['gender', 'race', 'age']
    
    # Get unique values for each attribute
    attribute_values = {}
    for attr in attributes:
        attribute_values[attr] = metadata[attr].unique()
    
    # Create different demographic distributions for each seller
    seller_demographics = {}
    for i in range(num_sellers):
        seller_demographics[i] = {}
        for attr in attributes:
            values = attribute_values[attr]
            # Create a distribution emphasizing different values for different sellers
            weights = np.random.dirichlet([0.5] * len(values))
            # Increase weight of specific values to create diversity between sellers
            if i < len(values):
                weights[i % len(values)] += 0.5  # Boost one specific demographic
            weights /= weights.sum()
            seller_demographics[i][attr] = dict(zip(values, weights))
    
    # Sample indices for each seller
    seller_indices = {}
    for seller_id, demographics in seller_demographics.items():
        # Create sampling weights based on demographics
        weights = np.ones(len(metadata))
        for attr, value_weights in demographics.items():
            # Multiply weights by the weight of each sample's attribute value
            for value, weight in value_weights.items():
                weights[metadata[attr] == value] *= weight
        
        # Normalize weights
        weights /= weights.sum()
        
        # Sample indices
        rang = np.arange(len(metadata))
        seller_indices[seller_id] = np.random.choice(
            rang, size=min(num_samples, len(metadata)), replace=False, p=weights
        )
    
    return seller_indices


def get_random_class_mask(labels, num_classes=10):
    """
    Create a mask for selecting random classes based on type.
    
    Args:
        labels: Array-like containing class labels
        
    Returns:
        Tuple of (mask, selected_classes)
    """
    classes = torch.unique(labels)
    cls_sel = np.random.choice(classes, size=num_classes, replace=False)
    cls_sel = torch.tensor(cls_sel)
    mask = torch.isin(labels, cls_sel)
    return mask, cls_sel

def sample_dirichlet(
    classes, 
    num_sellers=10, 
    num_samples=1000, 
    alpha=0.1, 
    min_samples=1,
    debug=False,
):
    """
    Sample seller data according to dirichlet distribution.
    
    Args:
        classes: Array-like containing class labels to sample from
        num_sellers: Number of sellers to generate
        num_samples: Number of samples per seller
        alpha: Parameter controlling heterogeneity (lower = more heterogeneous)
        min_samples: Minimum samples per class
    
    Returns:
        Dictionary of seller indices
    """
    counts = np.ones(len(classes))
    classes = np.array(classes)
    class_count = np.unique(classes, return_counts=True)
    for idx, cnt in zip(*class_count):
        counts[idx] = cnt

    if debug:
        print(f"{class_count=}")

    prop = counts / counts.sum()
    weights = np.random.dirichlet(alpha + prop, size=num_sellers)
    sample_weights = {i: v[classes] for i, v in enumerate(weights)}
    sample_weights = {k: v / v.sum() for k, v in sample_weights.items()}
    if debug:
        print(f"{prop=}")

    n = len(classes)
    seller_indexes = {
        k: np.random.choice(np.arange(n), size=num_samples, replace=False, p=v)
        for k, v in sample_weights.items()
    }
    
    if min_samples > 0:
        for k in seller_indexes.keys():
            new_index = []
            for i, c in enumerate(class_count[0]):
                class_indices = np.where(classes == c)[0]
                np.random.shuffle(class_indices)
                new_index.extend(class_indices[:min_samples])

            new_index = new_index[:num_samples]
            seller_indexes[k][: len(new_index)] = new_index

    return seller_indexes


class Buyer:
    def __init__(self, data):
        self.data = data

    def get_query(self):
        pass

class Seller:
    def __init__(self, data):
        self.data = data

    def get_measurements(self):
        pass

class Trainer:
    pass


def load_embeddings(embedding_model="clip", domain='imagenet', embedding_dir='../embeddings'):
    """
    Load embeddings for the specified domain.
    
    Args:
        embedding_model: Embedding model name (e.g., 'clip')
        domain: Specific domain (e.g., 'imagenet', 'medmnist', 'face')
        embedding_dir: Directory containing embedding files
    
    Returns:
        Dictionary containing embeddings and labels
    """
    embedding_dir = Path(embedding_dir)
    
    if domain == 'imagenet':
        # embedding = torch.load(embedding_dir/f"imagenet-val-set_{embedding_model}.pt")
        # label_df = pd.read_csv(embedding_dir/f"imagenet-val-set_{embedding_model}.csv")

        # fix nans in sketch csv
        embedding = torch.load(embedding_dir/f"imagenetv2-matched-frequency-format-val_{embedding_model}.pt")
        label_df = pd.read_csv(embedding_dir/f"imagenetv2-matched-frequency-format-val_{embedding_model}.csv")

        print(label_df[label_df.isna().any(axis=1)])
        return embedding['embeddings'], label_df
        # seller_embeddings = {
        #     'ImageNet': f'imagenet-val-set_{embedding_model}.pt',
        #     # 'ImageNet V2': f'imagenetv2-matched-frequency-format-val_{embedding_model}.pt',
        #     # 'ImageNet-A': f'imagenet-a_{embedding_model}.pt',
        #     # 'ImageNet-R': f'imagenet-r_{embedding_model}.pt',
        #     # 'ImageNet-Sketch': f'imagenet-sketch_{embedding_model}.pt',
        #     # 'ImageNet-C (noise)': f'imagenet-c-gaussian_noise_{embedding_model}.pt',
        #     # 'ImageNet-C (jpeg)': f'imagenet-c-jpeg_compression_{embedding_model}.pt',
        #     # 'ImageNet-C (blur)': f'imagenet-c-defocus_blur_{embedding_model}.pt',
        # }

        # return dict(
        #     buyer=torch.load(embedding_dir / buyer_embedding),
        #     sellers={n: torch.load(embedding_dir / p) for n, p in seller_embeddings.items()}
        # )
        
    elif domain == 'medmnist':
        medmnist_variants = {
            "MedMNIST (blood)": "clip_embedding_medmnist_bloodmnist_224.pt",
            "MedMNIST (breast)": "clip_embedding_medmnist_breastmnist_224.pt",
            "MedMNIST (chest)": "clip_embedding_medmnist_chestmnist_224.pt",
            "MedMNIST (derma)": "clip_embedding_medmnist_dermamnist_224.pt",
            "MedMNIST (path)": "clip_embedding_medmnist_pathmnist_224.pt",
            "MedMNIST (retina)": "clip_embedding_medmnist_retinamnist_224.pt",
            "MedMNIST (tissue)": "clip_embedding_medmnist_tissuemnist_224.pt",
            "MedMNIST (organ)": "clip_embedding_medmnist_organamnist_224.pt",
        }
        
        if domain not in medmnist_variants:
            raise ValueError(f"Unknown MedMNIST domain: {domain}. Available domains: {list(medmnist_variants.keys())}")
        
        file_path = embedding_dir / medmnist_variants[domain]
        print(f"Loading {domain} embeddings from {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {file_path}")
            
        data = torch.load(file_path)
        return data
        
    elif domain == 'face':
        # For face data, we would expect specific embedding files
        file_path = embedding_dir / f"clip_embedding_{domain.lower()}.pt"
        print(f"Loading {domain} embeddings from {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {file_path}")
            
        data = torch.load(file_path)
        return data
    
    else:
        raise ValueError(f"Unknown domain: {domain}. Available domains: 'imagenet', 'medmnist', 'face'")


def run_experiment(
    embedding_model='clip',
    domain='imagenet',
    embedding_dir='../embeddings',
    num_test=1000,
    num_buyer=100,
    num_classes=100,
    num_sellers=3,
    num_samples=1000,
    num_train=1000,
    num_trials=1,
    baselines=['LavaEvaluator', 'KNNShapley'],
    n_components=5,
    min_samples=1,
    alpha=0.5,
    output_file=None,
    debug=False
):
    """
    Run experiments using embeddings.
    
    Args:
        domain: Specific domain/variant
        embedding_dir: Directory containing embedding files
        num_test: Number of test samples
        num_buyer: Number of buyer samples
        num_sellers: Number of sellers
        num_samples: Number of samples per seller
        num_train: Number of training samples for valuation
        num_trials: Number of trials
        baselines: List of valuation methods
        n_components: Number of components for measurements
        min_samples: Minimum samples per class
        alpha: Parameter for Dirichlet sampling
        output_file: Path to save results
        debug: Whether to run in debug mode with fewer sellers and trials
    
    Returns:
        Results dictionary
    """
    if debug:
        num_sellers = 5
        num_trials = 1
        num_samples = 500
        output_file = None
        print("Running in debug mode with fewer sellers and trials.")
    
    # Load data
    embeddings, label_df = load_embeddings(embedding_model=embedding_model, domain=domain, embedding_dir=embedding_dir)
    x_data = embeddings
    y_data = torch.tensor(label_df.class_index.values)
    
    if debug:
        print('min', y_data.min())
        print('max', y_data.max())
    
    num_classes = len(torch.unique(y_data))
    results = {
        'num_test': num_test,
        'num_buyer': num_buyer,
        'num_sellers': num_sellers,
        'num_samples': num_samples,
        'num_classes': num_classes,
        'num_trials': num_trials,
        'n_components': n_components,
        'alpha': alpha,
        'classes_selected': [],
        'trial_results': [],
        'correlations': {}
    }
    
    for random_state in range(num_trials):
        print(str(random_state).center(40, '-'))
        
        # Choose random classes for buyer distribution
        class_mask, classes_selected = get_random_class_mask(y_data, num_classes=num_classes)
        index = np.arange(len(y_data))
        class_index = index[class_mask]
        np.random.shuffle(class_index)
        print("Buyer has classes:", classes_selected)
        print(f'{len(class_index)=}')

        # Take samples for buyer query and test set
        buyer_index = class_index[:num_buyer]
        assert len(buyer_index) == num_buyer, f"Expected {num_buyer=} buyer samples, but got {len(buyer_index)=}. Check {num_buyer=} and num_test values."
        test_index = class_index[num_buyer:num_test + num_buyer]
        x_buyer = x_data[buyer_index]
        # y_buyer = y_data[buyer_index]
        x_test = x_data[test_index]
        y_test = y_data[test_index]

        # Use leftover data to sample sellers 
        samp_index = np.concatenate([
            class_index[num_test + num_buyer:], index[~class_mask]
        ])
        np.random.shuffle(samp_index)
        x_samp = x_data[samp_index]
        y_samp = y_data[samp_index]

        if debug:
            print(f"{x_test.shape=}")
            print(f"{y_test.shape=}")
            print(f"{y_samp.shape=}")

        # Generate sellers from dirichlet distribution 
        sellers = sample_dirichlet(
            y_samp, num_sellers=num_sellers, 
            num_samples=num_samples, 
            min_samples=min_samples,
            alpha=alpha,
        )
        assert len(buyer_index) == num_buyer, f"Expected {num_buyer} buyer samples, but got {len(buyer_index)}. Check num_buyer and num_test values."

        # Create trainer for each seller and parallelize fitting model and return utility for each seller
        def train_and_evaluate(seller_id, x_seller, y_seller, x_test, y_test):
            model = SGDClassifier(loss='log_loss', max_iter=200) 
            clf = make_pipeline(StandardScaler(), model)
            clf.fit(x_seller, y_seller)
            return seller_id, accuracy_score(y_test, clf.predict(x_test))

        n_jobs = min(len(sellers), 16) 

        print('Computing utilities'.center(40, '-'))
        seller_utilities = dict(Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(train_and_evaluate)(j, x_samp[index], y_samp[index], x_test, y_test)
            for j, index in sellers.items()
        ))
        if debug:
            print(seller_utilities)
        print('Computing data measurements'.center(40, '-'))
        seller_measurements = dict(Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(valuation.get_measurements)( x_buyer, x_samp[index], seller_id=j, n_components=n_components)
            for j, index in sellers.items()
        ))

        if debug:
            print(seller_measurements)

        seller_results = {}
        for seller_id, index in sellers.items():
            labels = y_samp[index]
            seller_results[seller_id] = {
                'utility': seller_utilities[seller_id],
                'measurements': seller_measurements[seller_id],
                'percent_relevant': torch.isin(labels, classes_selected).sum().item() / len(labels)
            }

        results['classes_selected'].append(classes_selected.tolist())
        results['trial_results'].append(seller_results)
        
    # Calculate correlations
    results['correlations'] = []
    for trial_result in results['trial_results']:
        seller_ids = list(trial_result.keys())
        utilities = [trial_result[seller_id]['utility'] for seller_id in seller_ids]
        measurement_types = list(trial_result[seller_ids[0]]['measurements'].keys())
        measurements = {m: [trial_result[k]['measurements'][m] for k in seller_ids] for m in measurement_types}
        
        trial_correlations = {}
        for measurement_type, measurement_values in measurements.items():
            trial_correlations[measurement_type] = utils.calculate_correlation(utilities, measurement_values)
        if debug:
            print(f"Trial {random_state} correlations: {trial_correlations}")
        results['correlations'].append(trial_correlations)
    
    # Save results if output file is provided
    if output_file:
        output_path = Path(output_file)
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, default=float, indent=2)
        print(f"Results saved to {output_path}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Correlation experiments between data measurements and utility.')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory containing result files')
    parser.add_argument('--output_dir', type=str, default='figures',
                        help='Directory to save plots')
    parser.add_argument('--domain', type=str, choices=['imagenet', 'medmnist', 'face'], default='imagenet',
                        help='Type of embedding data')
    parser.add_argument('--embedding_dir', type=str, default='embeddings',
                        help='Directory containing embedding files')
    parser.add_argument('--num_test', type=int, default=1000,
                        help='Number of test samples')
    parser.add_argument('--num_buyer', type=int, default=100,
                        help='Number of buyer samples')
    parser.add_argument('--num_sellers', type=int, default=100,
                        help='Number of sellers')
    parser.add_argument('--num_samples', type=int, default=5000,
                        help='Number of samples per seller')
    parser.add_argument('--num_train', type=int, default=1000,
                        help='Number of training samples for valuation')
    parser.add_argument('--num_trials', type=int, default=10,
                        help='Number of trials')
    parser.add_argument('--baselines', type=str, nargs='+', default=['LavaEvaluator', 'KNNShapley'],
                        help='Valuation methods to use')
    parser.add_argument('--n_components', type=int, default=5,
                        help='Number of components for measurements')
    parser.add_argument('--min_samples', type=int, default=1,
                        help='Minimum samples per class')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Parameter for Dirichlet sampling')
    parser.add_argument('--output_file', type=str, default=None,
                        help='File to save experiment results')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with fewer sellers and trials')
    
    args = parser.parse_args()
    
    run_experiment(
        domain=args.domain,
        embedding_dir=args.embedding_dir,
        num_test=args.num_test,
        num_buyer=args.num_buyer,
        num_sellers=args.num_sellers,
        num_samples=args.num_samples,
        num_train=args.num_train,
        num_trials=args.num_trials,
        baselines=args.baselines,
        n_components=args.n_components,
        min_samples=args.min_samples,
        alpha=args.alpha,
        output_file=args.output_file,
        debug=args.debug
    )