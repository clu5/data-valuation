import valuation
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
from collections import defaultdict
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
from pathlib import Path

def rank(d: dict, buyer):
    sorted_values = sorted([(k, v) for k, v in d.items()], key=lambda x: x[1])[::-1]
    ranking = zip(itertools.count(start=1), [x[0] for x in sorted_values])
    for r, v in ranking:
        if v == buyer:
            return r

def main():
    parser = argparse.ArgumentParser(description="Ranking script for data valuation")
    parser.add_argument("--model_type", type=str, choices=["clip", "dino", "efficientnet"], default="clip", help="Type of model")
    parser.add_argument("--embedding_dir", type=str, default='embeddings', help="Directory containing embedding files")
    parser.add_argument("--search_pattern", type=str, default="imagenet*_{model_type}", help="Search pattern for embedding and CSV files")
    parser.add_argument("--filter_class", type=str, help="Class to filter by")
    parser.add_argument("--num_buy", type=int, default=100, help="Number of buy samples")
    parser.add_argument("--num_sell", type=int, default=10000, help="Number of sell samples")
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials")
    parser.add_argument("--n_components", type=int, default=10, help="Number of components")
    parser.add_argument("--fig_output_dir", type=str, default="figures", help="Directory to save figures")
    parser.add_argument("--measures", type=str, nargs='+', default=["l2", "cosine", "correlation", "overlap", "volume", "vendi", "dispersion", "difference"], help="Measures to use")
    args = parser.parse_args()

    # Load and preprocess data
    datasets = {}
    embedding_dir = Path(args.embedding_dir)
    search_pattern = args.search_pattern.format(model_type=args.model_type)
    for emb_path in embedding_dir.glob(f"{search_pattern}.pt"):
        csv_path = emb_path.with_suffix('.csv')
        embeddings = torch.load(emb_path)["embeddings"]
        df = pd.read_csv(csv_path)
        if args.filter_class:
            indices = df[df['class'] == args.filter_class].index
            embeddings = embeddings[indices]
        datasets[emb_path.stem] = {"embeddings": embeddings, "data": df}

    # Print found embeddings
    print(f"Found embeddings with search pattern {search_pattern}:".center(80, "="))
    for name, data in datasets.items():
        print(f"{name:<30}: {data['embeddings'].shape}")

    print("\nStarting ranking experiments...")

    # Ranking experiments
    rank_results = defaultdict(list)
    first_buyer = list(datasets.keys())[0]
    first_buyer_results = defaultdict(list)
    
    for j in tqdm(range(args.num_trials), desc="Trials"):
        for buyer in datasets.keys():
            results = defaultdict(list)
            B = datasets[buyer]["embeddings"]
            for seller, v in datasets.items():
                S = datasets[seller]["embeddings"]
                _, buy = train_test_split(B, test_size=args.num_buy, random_state=j)
                _, sell = train_test_split(
                    S, test_size=min(len(S) - 1, args.num_sell), random_state=j
                )
                results[seller] = valuation.get_measurements(buy, sell, n_components=args.n_components)

            for m in args.measures:
                rank_results[m].append(
                    rank(dict(zip(results.keys(), [v[m] for v in results.values()])), buyer)
                )
                if buyer == first_buyer:
                    first_buyer_results[m].append({seller: results[seller][m] for seller in datasets.keys()})

    # Print average rankings
    print("\nAverage rankings across buyers for each measurement:")
    for m in args.measures:
        avg_rank = np.mean(rank_results[m])
        print(f"{m:<15}: {avg_rank:.2f}")

    # Plotting code for the first buyer
    num_measures = len(args.measures)
    num_cols = 4
    num_rows = (num_measures + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 3.5 * num_rows))

    for ax, val in zip(axes.flat, args.measures):
        names = list(datasets.keys())
        values = [np.mean([trial[seller] for trial in first_buyer_results[val]]) for seller in names]
        index = np.argsort(values)
        sorted_names = [names[i][:16] for i in index]
        sorted_values = [values[i] for i in index]
        color = ["C1" if names[i] == first_buyer else "C0" for i in index]

        if val in ["correlation", "overlap", "difference", "cosine"]:
            ax.set_xlim(0, 1)

        ax.barh(
            sorted_names,
            sorted_values,
            color=color,
        )
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), fontsize="x-large")
        ax.set_yticks(ax.get_yticks(), ax.get_yticklabels(), fontsize="xx-large")
        ax.set_title(val.capitalize(), fontsize="xx-large")
        ax.grid(axis='x', linestyle='--')
    fig.suptitle(f"Model Type: {args.model_type} - Buyer: {first_buyer}", fontsize="xx-large")
    fig.tight_layout(w_pad=0)
    save_name = f"{args.fig_output_dir}/ranking_{args.search_pattern.split('_')[0]}_{args.model_type}.png"
    fig.savefig(save_name, bbox_inches="tight", dpi=800)
    print(f"Figure saved to {save_name}")

if __name__ == "__main__":
    main()