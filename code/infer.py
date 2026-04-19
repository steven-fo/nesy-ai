import os
import json
import yaml
import argparse
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run inference using specified LLM model')
    parser.add_argument('--model', type=str, required=True, help='Model to use')
    parser.add_argument('--tokenizer', type=str, required=True, help='Tokenizer to use')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the input data file')
    parser.add_argument('--prompt_type', type=str, required=True, help='Type of prompt to use (fast/slow)')
    parser.add_argument('--gpu_ids', type=str, required=True, help='Comma separated list of GPU IDs to use')
    parser.add_argument('--output_path', type=str, required=True, help='Full path for output file')
    
    return parser.parse_args()

def get_prompt_prefix(prompt_type: str) -> str:
    with open('./config/prompt.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config[prompt_type]

def setup_model(model_path: str, tokenizer_path: str, gpu_ids: str) -> tuple[LLM, AutoTokenizer]:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    
    print(f"Loading model: {model_path}", flush=True)
    llm = LLM(model=model_path,
              tokenizer=tokenizer_path,
              trust_remote_code=True,
              dtype="bfloat16",
              tensor_parallel_size=len(gpu_ids.split(',')))
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return llm, tokenizer

def load_data(data_file: str) -> List[Dict[str, Any]]:
    print(f"Loading data from: {data_file}", flush=True)
    with open(data_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_prompts(data: List[Dict[str, Any]], tokenizer: AutoTokenizer, prompt_prefix: str, model_path: str, prompt_type: str) -> List[str]:
    print("Preparing prompts...", flush=True)
    prompts = []
    use_prefix = not ("qwq" in model_path.lower() and prompt_type == "complex")
    print(f"Using prefix: {'Yes' if use_prefix else 'No'} (model path: {model_path}, prompt type: {prompt_type})", flush=True)
    for item in data:
        if use_prefix:
            question = prompt_prefix + item['question']
        else:
            question = "Try to solve the following question: \n" + item['question']
        item['prefixed_question'] = question
        format = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(format)
    
    if prompts:
        print("\nFirst prompt example:", flush=True)
        print("Original question:", data[0]['question'], flush=True)
        print("Prefixed question:", data[0]['prefixed_question'], flush=True)
        print("Formatted prompt:", prompts[0], flush=True)
        print("", flush=True)
    
    return prompts

def generate_responses(llm: LLM, prompts: List[str]) -> List[Any]:
    print("Generating responses...", flush=True)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=4096,
    )
    return llm.generate(prompts, sampling_params)

def process_outputs(data: List[Dict[str, Any]], outputs: List[Any]) -> List[Dict[str, str]]:
    print("Processing outputs...", flush=True)
    llm_data = []
    for q, a in zip(data, outputs):
        llm_data.append({
            'question': q['question'],
            'prefixed_question': q.get('prefixed_question', q['question']),
            'answer': q['answer'],
            'llm_output': a.outputs[0].text
        })
    return llm_data

def save_results(results: List[Dict[str, str]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saving results to: {output_path}", flush=True)
    
def main():
    args = get_args()
    prompt_prefix = get_prompt_prefix(args.prompt_type)
    llm, tokenizer = setup_model(args.model, args.tokenizer, args.gpu_ids)
    data = load_data(args.data_file)
    prompts = prepare_prompts(data, tokenizer, prompt_prefix, args.model, args.prompt_type)
    outputs = generate_responses(llm, prompts)
    results = process_outputs(data, outputs)
    save_results(results, args.output_path)
    print("Processing completed!", flush=True)

if __name__ == "__main__":
    main()