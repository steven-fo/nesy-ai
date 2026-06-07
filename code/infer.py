import argparse
import json
import os
from typing import Any, Dict, List

import pandas as pd
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference using fast and slow thinking prompts")
    parser.add_argument("--model", type=str, required=True, help="Model to use")
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer to use")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output_path", type=str, required=True, help="Full path for output file")
    parser.add_argument("--prompt_file", type=str, default="config/prompt.yaml", help="Path to prompt YAML file")
    parser.add_argument("--experiment", type=str, default="dual_process", choices=["dual_process", "anchoring"], help="Inference experiment mode",)
    parser.add_argument("--size", type=int, default=1000, help="Number of rows to load from the input file")
    return parser.parse_args()


def setup_model(model_path: str, tokenizer_path: str) -> tuple[Any, AutoTokenizer]:
    print(f"Loading model and tokenizer: {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator, tokenizer


def load_data(data_file: str, size: int) -> List[Dict[str, Any]]:
    print(f"Loading data from: {data_file}", flush=True)
    df = pd.read_csv(data_file).head(size)
    data_list: List[Dict[str, Any]] = []
    for index, row in df.iterrows():
        data_list.append(
            {
                "id": index + 1,
                "situation": row["situation"],
                "vrd": row["vrd"],
                "text": row["text"],
                "valence": row["valence"],
            }
        )
    return data_list


def load_prompt_templates(prompt_file: str) -> Dict[str, str]:
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_config = yaml.safe_load(f) or {}

    if not isinstance(prompt_config, dict):
        raise ValueError("Prompt file must contain a mapping with keys: fast, slow, anchor")

    for required_key in ["fast", "slow", "anchor"]:
        if required_key not in prompt_config:
            raise ValueError(f"Missing prompt key in YAML: {required_key}")

    return {
        "fast": str(prompt_config["fast"]),
        "slow": str(prompt_config["slow"]),
        "anchor": str(prompt_config["anchor"]),
    }


def get_prompt_prefix(prompt_type: str, prompt_templates: Dict[str, str]) -> str:
    if prompt_type not in prompt_templates:
        raise ValueError(f"Unsupported prompt type: {prompt_type}")
    return prompt_templates[prompt_type]


def prepare_prompts(q_item: Dict[str, Any], tokenizer: AutoTokenizer, prompt_prefix: str) -> str:
    print("Preparing prompt for a single data item...", flush=True)
    formatted_prompt_content = prompt_prefix.format(
        situation=q_item["situation"],
        vrd=q_item["vrd"],
        text=q_item["text"],
    )

    final_model_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": formatted_prompt_content}],
        tokenize=False,
        add_generation_prompt=True,
    )

    return final_model_prompt


def get_anchor_text(valence: str) -> str:
    if valence == "Supports":
        return "Opposes"
    if valence == "Opposes":
        return "Supports"
    return "Supports"


def prepare_anchored_prompts(q_item: Dict[str, Any], tokenizer: AutoTokenizer, anchor_template: str, main_template: str) -> str:
    print("Preparing anchored prompt for a single data item...", flush=True)
    anchor_text = get_anchor_text(q_item["valence"])

    formatted_anchor_prompt = anchor_template.format(
        vrd=q_item["vrd"],
        text=q_item["text"],
        anchor=anchor_text,
        situation=q_item["situation"],
    )
    formatted_main_prompt = main_template.format(
        situation=q_item["situation"],
        vrd=q_item["vrd"],
        text=q_item["text"],
    )

    combined_prompt_content = f"{formatted_main_prompt}\n\n{formatted_anchor_prompt}"
    final_model_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": combined_prompt_content}],
        tokenize=False,
        add_generation_prompt=True,
    )

    return final_model_prompt


def generate_responses(generator: Any, prompts: str) -> Any:
    print("Generating responses...", flush=True)
    return generator(
        prompts,
        max_new_tokens=1024,
        do_sample=False,
        return_full_text=False,
    )


def process_outputs(
    data: List[Dict[str, Any]],
    fast_prompts: List[str],
    fast_outputs: List[Any],
    slow_prompts: List[str],
    slow_outputs: List[Any],
    experiment: str,
) -> List[Dict[str, Any]]:
    print("Processing outputs...", flush=True)
    llm_data: List[Dict[str, Any]] = []
    for q_item, fast_prompt, fast_output_item, slow_prompt, slow_output_item in zip(
        data, fast_prompts, fast_outputs, slow_prompts, slow_outputs
    ):
        generated_fast_text = fast_output_item[0]["generated_text"]
        generated_slow_text = slow_output_item[0]["generated_text"]

        llm_data.append(
            {
                "id": q_item["id"],
                "experiment": experiment,
                "situation": q_item["situation"],
                "vrd": q_item["vrd"],
                "text": q_item["text"],
                "valence": q_item["valence"],
                "fast_prompts": fast_prompt,
                "llm_raw_output_fast": generated_fast_text,
                "slow_prompts": slow_prompt,
                "llm_raw_output_slow": generated_slow_text,
            }
        )

    return llm_data


def run_inference(
    model_path: str,
    tokenizer_path: str,
    data_file: str,
    size: int,
    prompt_file: str,
    experiment: str,
) -> List[Dict[str, Any]]:
    prompt_templates = load_prompt_templates(prompt_file)
    prompt_prefix_fast = get_prompt_prefix("fast", prompt_templates)
    prompt_prefix_slow = get_prompt_prefix("slow", prompt_templates)
    prompt_prefix_anchor = get_prompt_prefix("anchor", prompt_templates)
    generator, tokenizer = setup_model(model_path, tokenizer_path)
    data = load_data(data_file, size)

    fast_prompts: List[str] = []
    slow_prompts: List[str] = []
    fast_outputs: List[Any] = []
    slow_outputs: List[Any] = []

    for q_item in data:
        fast_prompt = prepare_prompts(q_item, tokenizer, prompt_prefix_fast)

        if experiment == "anchoring":
            slow_prompt = prepare_anchored_prompts(
                q_item,
                tokenizer,
                anchor_template=prompt_prefix_anchor,
                main_template=prompt_prefix_slow,
            )
        elif experiment == "dual_process":
            slow_prompt = prepare_prompts(q_item, tokenizer, prompt_prefix_slow)
        else:
            raise ValueError(f"Unsupported experiment: {experiment}")

        fast_output = generate_responses(generator, fast_prompt)
        slow_output = generate_responses(generator, slow_prompt)

        fast_prompts.append(fast_prompt)
        slow_prompts.append(slow_prompt)
        fast_outputs.append(fast_output)
        slow_outputs.append(slow_output)

    return process_outputs(data, fast_prompts, fast_outputs, slow_prompts, slow_outputs, experiment)


def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saving {len(results)} processed items to: {output_path}", flush=True)


def main() -> List[Dict[str, Any]]:
    args = get_args()
    results = run_inference(
        args.model,
        args.tokenizer,
        args.data_file,
        args.size,
        args.prompt_file,
        args.experiment,
    )
    save_results(results, args.output_path)
    return results


if __name__ == "__main__":
    main()