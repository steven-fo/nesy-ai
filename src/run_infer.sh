#!/bin/bash

# Model
MODELS=("Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-3B-Instruct" "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-14B-Instruct" "Qwen/Qwen2.5-32B-Instruct")
EXPERIMENTS=("dual_process" "anchoring")

# Run inference
for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT=$(basename ${MODEL})
    echo "Using model: ${MODEL_SHORT}"

    for EXPERIMENT in "${EXPERIMENTS[@]}"; do
        # DATA_PATH
        INPUT_PATH="./data/input/valence_test.csv"
        if [ "${EXPERIMENT}" = "anchoring" ]; then
            OUTPUT_PATH="./data/output/${MODEL_SHORT}_anchored_results.json"
        else
            OUTPUT_PATH="./data/output/${MODEL_SHORT}_results.json"
        fi

        if [ -f "${OUTPUT_PATH}" ]; then
            echo "Output file ${OUTPUT_PATH} already exists, skipping ${MODEL_SHORT} (${EXPERIMENT})"
            continue
        fi
        OUTPUT_DIR=$(dirname "${OUTPUT_PATH}")
        mkdir -p "${OUTPUT_DIR}"

        # Run
        echo "Starting inference (${EXPERIMENT})"
        python code/infer.py \
            --model "${MODEL}" \
            --tokenizer "${MODEL}" \
            --data_file "${INPUT_PATH}" \
            --prompt_file "./config/prompt.yaml" \
            --experiment "${EXPERIMENT}" \
            --output_path "${OUTPUT_PATH}"

        echo "Completed processing ${MODEL_SHORT} (${EXPERIMENT})"
    done
done