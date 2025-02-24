#!/bin/bash
set -e  # Exit on error

# Default paths
DATA_DIR="${DATA_DIR:-/mnt/data}"
OUTPUT_DIR="embeddings"
LABELS_CSV="$DATA_DIR/imagenet.csv"

# Uncomment the ImageNet-C corruptions you want to process:
declare -a SELECTED_CORRUPTIONS=(
    # "brightness"
    "contrast"
    "defocus_blur"
    # "elastic_transform"
    "fog"
    # "frost"
    # "gaussian_blur"
    "gaussian_noise"
    # "glass_blur"
    # "impulse_noise"
    "jpeg_compression"
    # "motion_blur"
    "pixelate"
    # "saturate"
    # "shot_noise"
    # "snow"
    # "spatter"
    # "speckle_noise"
    # "zoom_blur"
)

# Available models
declare -a ALL_MODELS=(
    "clip"
    "efficientnet"
    "dino"
)

SELECTED_MODELS=("${ALL_MODELS[@]}")

# Parse command line arguments
FORCE=false
DEBUG=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE=true
            shift
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --labels-csv)
            LABELS_CSV="$2"
            shift 2
            ;;
        --models)
            # Split comma-separated list into array
            IFS=',' read -ra SELECTED_MODELS <<< "$2"
            # Validate models
            for model in "${SELECTED_MODELS[@]}"; do
                if [[ ! " ${ALL_MODELS[@]} " =~ " ${model} " ]]; then
                    echo "Error: Invalid model '$model'"
                    echo "Available models: ${ALL_MODELS[*]}"
                    exit 1
                fi
            done
            shift 2
            ;;
        --list-models)
            echo "Available models:"
            printf "  %s\n" "${ALL_MODELS[@]}"
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: $0 [--force] [--debug] [--output-dir dir] [--labels-csv file] [--models model1,model2] [--list-models]"
            exit 1
            ;;
    esac
done

# Regular datasets to process
declare -a DATASETS=(
    "imagenet-a"
    "imagenet-val-set"
    "imagenetv2-matched-frequency-format-val"
    "imagenet-r"
    "imagenet-sketch"
)

# Function to process a dataset with a specific model
process_dataset() {
    local dataset=$1
    local model=$2
    local input_path="$DATA_DIR/$dataset"
    local output_file="$OUTPUT_DIR/${dataset}_${model}.pt"
    
    # Check if dataset exists
    if [[ ! -d "$input_path" ]]; then
        echo "Warning: Dataset directory $input_path not found, skipping..."
        return
    fi
    
    # Check if output already exists and force flag
    if [[ -f "$output_file" && "$FORCE" = false ]]; then
        echo "Embeddings for $dataset ($model) already exist at $output_file, skipping..."
        return
    fi
    
    echo "Processing $dataset with $model..."
    
    # Build Python command with optional debug flag
    cmd="python3 src/embed_images.py \
        --input "$input_path" \
        --output-dir "$OUTPUT_DIR" \
        --labels-csv "$LABELS_CSV" \
        --model "$model""
    
    if [[ "$DEBUG" = true ]]; then
        cmd+=" --debug"
    fi
    
    eval $cmd
    
    if [[ $? -eq 0 ]]; then
        echo "Successfully processed $dataset with $model"
    else
        echo "Error processing $dataset with $model"
        exit 1
    fi
}

# Function to process imagenet-c corruptions with a specific model
process_imagenetc() {
    local model=$1
    local imagenetc_dir="$DATA_DIR/imagenet-c"
    
    # Check if imagenet-c directory exists
    if [[ ! -d "$imagenetc_dir" ]]; then
        echo "Warning: ImageNet-C directory not found at $imagenetc_dir, skipping..."
        return
    fi
    
    # Process each corruption type
    for corruption in "${SELECTED_CORRUPTIONS[@]}"; do
        local level3_dir="$imagenetc_dir/$corruption/3"
        local output_file="$OUTPUT_DIR/imagenet-c_${corruption}_${model}.pt"
        
        # Check if level 3 exists
        if [[ ! -d "$level3_dir" ]]; then
            echo "Warning: Level 3 not found for $corruption, skipping..."
            continue
        fi
        
        # Check if output exists and force flag
        if [[ -f "$output_file" && "$FORCE" = false ]]; then
            echo "Embeddings for ImageNet-C $corruption ($model) already exist, skipping..."
            continue
        fi
        
        echo "Processing ImageNet-C $corruption with $model..."
        
        # Build Python command with optional debug flag
        cmd="python3 src/embed_images.py \
            --input "$level3_dir" \
            --output-dir "$OUTPUT_DIR" \
            --labels-csv "$LABELS_CSV" \
            --model "$model""
        
        if [[ "$DEBUG" = true ]]; then
            cmd+=" --debug"
        fi
        
        eval $cmd
        
        if [[ $? -eq 0 ]]; then
            echo "Successfully processed ImageNet-C $corruption with $model"
        else
            echo "Error processing ImageNet-C $corruption with $model"
            exit 1
        fi
    done
}

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "Starting embedding generation..."
echo "Output directory: $OUTPUT_DIR"
echo "Labels CSV: $LABELS_CSV"
echo "Selected models: ${SELECTED_MODELS[*]}"
if [ ${#SELECTED_CORRUPTIONS[@]} -gt 0 ]; then
    echo "Selected corruptions: ${SELECTED_CORRUPTIONS[*]}"
fi
if [[ "$FORCE" = true ]]; then
    echo "Force mode: enabled"
fi
if [[ "$DEBUG" = true ]]; then
    echo "Debug mode: enabled"
fi
echo

# Process each model
for model in "${SELECTED_MODELS[@]}"; do
    echo "Processing with model: $model"
    echo "========================="
    
    # Process regular datasets
    for dataset in "${DATASETS[@]}"; do
        process_dataset "$dataset" "$model"
        echo
    done
    
    # Process ImageNet-C corruptions
    echo "Processing ImageNet-C corruptions with $model..."
    process_imagenetc "$model"
    echo
done

echo "All processing completed!"