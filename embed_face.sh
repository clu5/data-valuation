#!/bin/bash
set -e  # Exit on error

# Default paths
DATA_DIR="${DATA_DIR:-/mnt/data}"
OUTPUT_DIR="embeddings"
LABELS_CSV="$DATA_DIR/fairface_labels.csv"

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
    "fairface"
    "utkface"
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
        --model "$model" \
        --domain face"
    
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

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "Starting embedding generation..."
echo "Output directory: $OUTPUT_DIR"
echo "Labels CSV: $LABELS_CSV"
echo "Selected models: ${SELECTED_MODELS[*]}"
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
done

echo "All processing completed!"