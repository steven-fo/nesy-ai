#!/bin/bash

# Model
MODELS=("Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-3B-Instruct" "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-14B-Instruct" "Qwen/Qwen2.5-32B-Instruct")

# Input data
DATA_TYPES=("mmlu" "mathqa" "medqa")

# API Key
API_KEY="YOUR_API_KEY"


# Run eval
for MODEL in "${MODELS[@]}"; do
    for DATA_TYPE in "${DATA_TYPES[@]}"; do

        MODEL_SHORT=$(basename ${MODEL})
        echo "Using model: ${MODEL_SHORT}"

        # DATA_PATH
        DATA_FILE="./data/output/${DATA_TYPE}/${MODEL_SHORT}/slow.json"
        OUTPUT_PATH="./data/output/${DATA_TYPE}/${MODEL_SHORT}/slow_eval.json"

        if [ -f "${OUTPUT_PATH}" ]; then
            echo "Output file ${OUTPUT_PATH} already exists, skipping ${MODEL_SHORT} with ${DATA_TYPE} prompt..."
            continue
        fi
        OUTPUT_DIR=$(dirname "${OUTPUT_PATH}")
        mkdir -p "${OUTPUT_DIR}"


        # Run
        echo "Starting eval"
        python code/eval.py \
            --data_file "${DATA_FILE}" \
            --output_file "${OUTPUT_PATH}" \
            --api_key ${API_KEY}

        echo "Completed processing ${MODEL_SHORT}"
    done
done
