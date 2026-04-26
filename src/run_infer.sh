#!/bin/bash

# Model
MODELS=("Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-3B-Instruct" "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-14B-Instruct" "Qwen/Qwen2.5-32B-Instruct")

# Run inference
for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT=$(basename ${MODEL})
    echo "Using model: ${MODEL_SHORT}"

    # DATA_PATH
    INPUT_PATH="./data/input/dataset.json"
    OUTPUT_PATH="./data/output/${MODEL_SHORT}.json"

    if [ -f "${OUTPUT_PATH}" ]; then
        echo "Output file ${OUTPUT_PATH} already exists, skipping ${MODEL_SHORT}"
        continue
    fi
    OUTPUT_DIR=$(dirname "${OUTPUT_PATH}")
    mkdir -p "${OUTPUT_DIR}"

    # Run
    echo "Starting inference"
    python code/infer.py \
        --model "${MODEL}" \
        --tokenizer "${MODEL}" \
        --data_file "${INPUT_PATH}" \
        --output_path "${OUTPUT_PATH}"

    echo "Completed processing ${MODEL_SHORT}"
done