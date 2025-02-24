"""
Image Embedding Generator using HuggingFace Models

This script provides functionality to generate embeddings from images using various
pre-trained models (EfficientNet, CLIP, DINO). It supports batch processing,
recursive directory searching, and label mapping for ImageNet datasets.

Example command-line usage:
    # Basic usage with defaults (CLIP model, CPU/CUDA auto-detect)
    python embed_images.py --input $DATA_DIR/imagenet-a
    
    # Debug mode (uses EfficientNet, small batch size)
    python embed_images.py --input $DATA_DIR/imagenet-a --debug
    
    # Full configuration
    python embed_images.py \
        --input $DATA_DIR/imagenet-a \
        --output-dir results/embeddings \
        --model clip \
        --batch-size 64 \
        --max-images 1000 \
        --labels-csv imagenet_labels.csv \
        --shuffle \
        --no-cuda

Example usage as imported module:
    ```python
    from embed_images import ImageEmbedder, embed_dataset, get_image_paths
    
    # Single image embedding
    embedder = ImageEmbedder(model_name='clip', device='cuda')
    embedding = embedder.get_embeddings('path/to/image.jpg')
    
    # Process multiple images
    paths = get_image_paths('path/to/dataset', max_images=1000)
    embeddings = embed_dataset(
        dataset_path='path/to/dataset',
        model_name='clip',
        batch_size=32,
        max_images=1000
    )
    ```
"""
import torch
from PIL import Image
from transformers import (
    EfficientNetImageProcessor,
    EfficientNetModel,
    CLIPProcessor,
    CLIPModel,
    AutoImageProcessor,
    AutoModel
)
from typing import Union, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


class ImageEmbedder:
    """A class to generate embeddings from images using different models."""
    
    def __init__(self, model_name: str = "efficientnet", device: Optional[str] = None):
        """
        Initialize the ImageEmbedder with specified model.
        
        Args:
            model_name: One of "efficientnet", "clip", or "dino"
            device: Device to run the model on ("cuda" or "cpu"). If None, will automatically detect.
        """
        self.model_name = model_name.lower()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the appropriate model and processor
        if self.model_name == "efficientnet":
            self.model = EfficientNetModel.from_pretrained("google/efficientnet-b0")
            self.processor = EfficientNetImageProcessor.from_pretrained("google/efficientnet-b0")
        
        elif self.model_name == "clip":
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        elif self.model_name == "dino":
            self.model = AutoModel.from_pretrained("facebook/dino-vits16")
            self.processor = AutoImageProcessor.from_pretrained("facebook/dino-vits16")
        
        else:
            raise ValueError(f"Model {model_name} not supported. Choose from: efficientnet, clip, or dino")
        
        self.model.to(self.device)
        self.model.eval()

    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Load an image from path."""
        return Image.open(image_path).convert('RGB')

    def process_images(self, images: Union[Image.Image, List[Image.Image]]):
        """Process images based on the selected model."""
        if not isinstance(images, list):
            images = [images]
            
        if self.model_name == "clip":
            return self.processor(images=images, return_tensors="pt").to(self.device)
        else:
            return self.processor(images=images, return_tensors="pt").to(self.device)

    @torch.no_grad()
    def get_embeddings(self, 
                      images: Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]]) -> np.ndarray:
        """
        Generate embeddings for the given images.
        
        Args:
            images: Single image or list of images (can be paths or PIL Images)
            
        Returns:
            numpy.ndarray: Image embeddings
        """
        # Convert to list if single image
        if not isinstance(images, list):
            images = [images]
            
        # Load images if paths are provided
        processed_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                processed_images.append(self.load_image(img))
            else:
                processed_images.append(img)
                
        # Process images
        inputs = self.process_images(processed_images)
        
        # Get embeddings based on model type
        if self.model_name == "clip":
            embeddings = self.model.get_image_features(**inputs)
        elif self.model_name == "dino":
            embeddings = self.model(**inputs).last_hidden_state[:, 0, :]
        else:  # efficientnet
            embeddings = self.model(**inputs).pooler_output
            
        return embeddings.cpu().numpy()

def get_image_paths(
    dataset_path: Union[str, Path],
    file_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'),
    max_images: Optional[int] = None,
    shuffle: bool = False
) -> List[Path]:
    """
    Recursively find all image files in a directory and its subdirectories.
    
    Args:
        dataset_path: Root directory to search
        file_extensions: Tuple of valid image file extensions
        max_images: Maximum number of images to return (None for all)
        shuffle: Whether to randomly shuffle the image paths
        
    Returns:
        List[Path]: List of paths to image files
    """
    dataset_path = Path(dataset_path)
    
    # Recursively find all files with matching extensions
    image_paths = [
        p for p in dataset_path.rglob('*') 
        if p.suffix.lower() in file_extensions
    ]
    print(f"Found {len(image_paths)} total images")
    
    if not image_paths:
        raise ValueError(f"No images found in {dataset_path} with extensions {file_extensions}")
    
    # Optionally shuffle the paths
    if shuffle:
        import random
        random.shuffle(image_paths)
    
    # Limit number of images if specified
    if max_images is not None:
        image_paths = image_paths[:max_images]
    
    return image_paths

def embed_dataset(
    dataset_path: Union[str, Path],
    model_name: str = "efficientnet",
    batch_size: int = 32,
    file_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'),
    max_images: Optional[int] = None,
    shuffle: bool = False
) -> np.ndarray:
    """
    Embed an entire dataset of images.
    
    Args:
        dataset_path: Path to the directory containing images
        model_name: Name of the model to use
        batch_size: Number of images to process at once
        file_extensions: Tuple of valid file extensions to process
        
    Returns:
        numpy.ndarray: Array of embeddings for all images
    """
    # Get image paths using the new function
    image_paths = get_image_paths(
        dataset_path=dataset_path,
        file_extensions=file_extensions,
        max_images=max_images,
        shuffle=shuffle
    )
    
    embedder = ImageEmbedder(model_name)
    all_embeddings = []
    
    # Process images in batches
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i + batch_size]
        batch_embeddings = embedder.get_embeddings(batch_paths)
        all_embeddings.append(batch_embeddings)
        
    return np.vstack(all_embeddings)

def get_save_name(input_path: Path, model_name: str, debug: bool = False) -> str:
    """
    Generate save name from input path and model name.
    
    Args:
        input_path: Path to the input directory
        model_name: Name of the model used (e.g., 'clip', 'dino')
        
    Returns:
        str: Generated save name
    """
    base_name = input_path.stem
    # Special handling for ImageNet-C corruptions
    if 'imagenet-c' in str(input_path).lower():
        # Extract corruption name from path
        parts = input_path.parts
        corruption_idx = parts.index('imagenet-c') + 1
        if corruption_idx < len(parts):
            corruption_name = parts[corruption_idx]
            save_name = f"imagenet-c_{corruption_name}_{model_name}"
    else:
        # Regular dataset naming
        save_name = f"{base_name}_{model_name}"
    if debug:
        save_name = save_name + '_debug'
    return save_name

def create_image_mapping(
    image_paths: List[Path],
    labels_df: Optional[pd.DataFrame] = None,
    dataset_name: str = 'imagenet',
    debug: bool = False,
) -> pd.DataFrame:
    """
    Create a DataFrame mapping image paths to indices and labels.
    
    Args:
        image_paths: List of image file paths
        labels_df: DataFrame containing ImageNet labels with columns [class_index, class_name]
        dataset_name: Dataset name for identification
    
    Returns:
        pd.DataFrame with columns: [path, index, class_id, class_name, class_index]
    """
    # Create basic mapping
    mapping = pd.DataFrame({
        'dataset_name': dataset_name,
        'path': [str(p) for p in image_paths],
        'index': range(len(image_paths)),
    })
    
    # Extract label from path if labels_df is provided
    if labels_df is not None:
        # Extract class ID from path (parent folder name)
        mapping['class_id'] = mapping['path'].apply(lambda x: Path(x).parent.name)
        
        # Map class IDs to names and indices using labels_df
        # The index in labels_df is 0,1,2... so we need to use the first column
        class_name_dict = dict(zip(labels_df.iloc[:, 0], labels_df['class_name']))
        class_index_dict = dict(zip(labels_df.iloc[:, 0], labels_df['class_index']))
        
        mapping['class_name'] = mapping['class_id'].map(class_name_dict)
        mapping['class_index'] = mapping['class_id'].map(class_index_dict)
        
        if debug:
            # Print diagnostics in debug mode
            print("\nDebug: First few rows of labels_df:")
            print(labels_df.head())
            print("\nDebug: Example mappings:")
            print(f"First class ID in mapping: {mapping['class_id'].iloc[0]}")
            print(f"Available class IDs in dictionary: {list(class_name_dict.keys())[:5]}")
    
    return mapping


def save_embeddings(
    embeddings: np.ndarray,
    paths: List[Path],
    output_path: Path,
    labels_df: Optional[pd.DataFrame] = None,
    debug: bool = False,
):
    """
    Save embeddings, paths, and labels to files.
    
    Args:
        embeddings: Numpy array of embeddings
        paths: List of image paths
        output_path: Path to save embeddings
        labels_df: DataFrame containing ImageNet labels (optional)
        debug: Whether to print debug information
    """
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings tensor
    torch.save({
        'embeddings': torch.from_numpy(embeddings),
        'paths': [str(p) for p in paths]
    }, output_path)
    
    # Create and save mapping CSV
    mapping_df = create_image_mapping(
        paths, 
        labels_df, 
        dataset_name=output_path.stem,
        debug=debug,
    )
    csv_path = output_path.with_suffix('.csv')
    mapping_df.to_csv(csv_path, index=False)
    
    return mapping_df


def main():
    import argparse
    import os
    from datetime import datetime
    import pandas as pd
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Generate image embeddings using various models')
    parser.add_argument('--input', type=str, 
                       default=os.path.join(os.getenv('DATA_DIR', ''), 'imagenet-a'),
                       help='Input directory containing images')
    parser.add_argument('--output-dir', type=str, 
                       default='embeddings',
                       help='Output directory for saved embeddings')
    parser.add_argument('--labels-csv', type=str,
                       default=None,
                       help='Path to CSV file containing ImageNet labels')
    parser.add_argument('--model', type=str,
                       choices=['efficientnet', 'clip', 'dino'],
                       default='clip',
                       help='Model to use for embeddings')
    parser.add_argument('--batch-size', type=int,
                       default=32,
                       help='Batch size for processing')
    parser.add_argument('--max-images', type=int,
                       default=None,
                       help='Maximum number of images to process')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode (process only 4 images)')
    parser.add_argument('--shuffle', action='store_true',
                       help='Shuffle the dataset (default: False)')
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    print(f'{input_path=}')
    print(f'{output_dir=}')
    
    # Generate output filename
    save_name = get_save_name(input_path, args.model)
    output_path = output_dir / f"{save_name}.pt"
    
    # Load labels if provided
    labels_df = None
    if args.labels_csv:
        print(f"Loading labels from {args.labels_csv}")
        labels_df = pd.read_csv(args.labels_csv)
    
    # Validate input directory
    if not input_path.exists():
        raise ValueError(f"Input directory {input_path} does not exist")
    
    # In debug mode, override max_images
    if args.debug:
        args.max_images = 4
        print("Debug mode: Processing only 4 images")
    
    # Get image paths
    print(f"Searching for images in {input_path}")
    image_paths = get_image_paths(
        dataset_path=input_path,
        max_images=args.max_images,
        shuffle=args.shuffle
    )
    print(f"Processing {len(image_paths)} images")
    
    if args.debug:
        print("Debug mode: Image paths to process:")
        for path in image_paths:
            print(f"  {path}")
    

    print('CUDA available: ', torch.cuda.is_available())
    
    # Generate embeddings
    start_time = datetime.now()
    print(f"Generating embeddings using {args.model} model")
    embeddings = embed_dataset(
        dataset_path=input_path,
        model_name=args.model,
        batch_size=args.batch_size,
        max_images=args.max_images,
        shuffle=args.shuffle
    )
    
    # Print timing information
    duration = datetime.now() - start_time
    print(f"Processing completed in {duration}")
    print(f"Embeddings shape: {embeddings.shape}")
   
    # Save embeddings if not in debug mode
    # if not args.debug:
    print(f"Saving embeddings to {output_path}")
    mapping_df = save_embeddings(
        embeddings,
        image_paths,
        output_path,
        labels_df=labels_df,
        debug=args.debug,
    )
    print(f"Created mapping file: {output_path.with_suffix('.csv')}")
    if args.debug:
        print("\nFirst few rows of mapping:")
        print(mapping_df.head())
    # else:
    #     print("Debug mode: Skipping save")

if __name__ == "__main__":
    main()