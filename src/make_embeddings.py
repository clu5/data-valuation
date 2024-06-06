import os
import torch
import clip
from PIL import Image
import argparse
import pandas as pd
from pathlib import Path

def load_class_labels(labels_csv):
    """
    Load class labels and their indices from a CSV file.

    Args:
    labels_csv (str): Path to the CSV file mapping class directories to label indices.

    Returns:
    dict: A dictionary mapping class names to indices.
    """
    labels_df = pd.read_csv(labels_csv, index_col=0)
    label_map = {row['class_name']: row['class_index'] for index, row in labels_df.iterrows()}
    return label_map

def load_images_from_directory(directory, label_map, debug):
    """
    Load and verify images from directories organized by type and level of corruption.

    Args:
    directory (str): Path to the root directory containing directories organized by corruption type and level.
    label_map (dict): Dictionary mapping class names to indices.
    debug (bool): Whether to load only the first image for testing.

    Returns:
    dict: A dictionary where keys are tuples (corruption_type, level) and values are lists of (label_index, PIL.Image).
    """
    corruption_images = {}
    for corruption_type in os.listdir(directory):
        corruption_path = Path(directory) / corruption_type
        if corruption_path.is_dir():
            for level in os.listdir(corruption_path):
                level_path = corruption_path / level
                if level_path.is_dir():
                    key = (corruption_type, level)
                    corruption_images[key] = []
                    for class_dir in os.listdir(level_path):
                        if class_dir in label_map:
                            class_path = level_path / class_dir
                            for filename in os.listdir(class_path):
                                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                                    file_path = class_path / filename
                                    try:
                                        with Image.open(file_path) as img:
                                            img = img.convert("RGB")
                                            corruption_images[key].append((label_map[class_dir], img))
                                        if debug:
                                            break
                                    except (IOError, SyntaxError) as e:
                                        print(f"Error opening or verifying image {filename} at {file_path}: {e}")
                            if debug:
                                break
                    if debug:
                        break
    return corruption_images

def extract_and_save_features(images, output_dir):
    """
    Extract image features using the CLIP model for each type and level of corruption, and save them.

    Args:
    images (dict): A dictionary mapping (corruption_type, level) to lists of (label_index, PIL.Image).
    output_dir (str): The directory to save the extracted features.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    for key, img_list in images.items():
        all_features = []
        labels = []
        for label_index, image in img_list:
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                features = model.encode_image(image_tensor)
            all_features.append(features)
            labels.append(label_index)

        all_features_tensor = torch.cat(all_features, dim=0)

        # Saving features in a flat directory structure
        corruption_type, level = key
        output_path = Path(output_dir) / f"{corruption_type}_{level}.pt"
        torch.save({'embeddings': all_features_tensor, 'labels': torch.tensor(labels)}, output_path)

def main(directory, labels_csv, output_dir, debug):
    label_map = load_class_labels(labels_csv)
    images = load_images_from_directory(directory, label_map, debug)
    extract_and_save_features(images, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and save features from ImageNet-C dataset using CLIP, organized by type and level of corruption.")
    parser.add_argument("directory", type=str, help="Directory containing ImageNet-C organized by corruption type and level.")
    parser.add_argument("labels_csv", type=str, help="CSV file containing class labels and their corresponding indices.")
    parser.add_argument("output_dir", type=str, help="Directory to save the extracted features.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode to process only the first image of the first class of the first level of the first type.")
    args = parser.parse_args()
    main(args.directory, args.labels_csv, args.output_dir, args.debug)


