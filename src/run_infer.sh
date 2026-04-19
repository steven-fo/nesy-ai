#!/bin/bash

# Model
MODELS=("Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-3B-Instruct" "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-14B-Instruct" "Qwen/Qwen2.5-32B-Instruct")

# Prompt
PROMPT_TYPES=(fast slow)

# Input data
DATA_TYPES=("mmlu" "mathqa" "medqa")

# GPU
GPU_IDS="1,2"


# Run inference
for DATA_TYPE in "${DATA_TYPES[@]}"; do
    echo "Using data type: ${DATA_TYPE}"
    
    for PROMPT_TYPE in "${PROMPT_TYPES[@]}"; do
        echo "Using prompt type: ${PROMPT_TYPE}"
        
        for MODEL in "${MODELS[@]}"; do
            MODEL_SHORT=$(basename ${MODEL})
            echo "Using model: ${MODEL_SHORT}"

            # DATA_PATH
            INPUT_PATH="./data/input/${DATA_TYPE}.json"
            OUTPUT_PATH="./data/output/${DATA_TYPE}/${MODEL_SHORT}/${PROMPT_TYPE}.json"

            if [ -f "${OUTPUT_PATH}" ]; then
                echo "Output file ${OUTPUT_PATH} already exists, skipping ${MODEL_SHORT} with ${PROMPT_TYPE} prompt..."
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
                --gpu_ids "${GPU_IDS}" \
                --output_path "${OUTPUT_PATH}" \
                --prompt_type "${PROMPT_TYPE}"

            echo "Completed processing ${MODEL_SHORT} with ${PROMPT_TYPE} prompt."
        done
    done
done