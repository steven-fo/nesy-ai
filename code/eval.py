import os
import json
import yaml
import argparse
from tqdm import tqdm
from typing import List, Dict, Any
from zhipuai import ZhipuAI

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate LLM outputs using external APIs')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Input file path (including .json)')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output file path')
    parser.add_argument('--api_key', type=str, required=True,
                        help='ZhipuAI的API Key')
    return parser.parse_args()

def setup_client(api_key: str) -> tuple[Any, str]:
    client = ZhipuAI(api_key=api_key)
    model_name = "glm-4-plus"
    return client, model_name

def load_data(data_file: str) -> List[Dict[str, Any]]:
    print(f"Loading data from: {data_file}")
    with open(data_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_prompt_template() -> str:
    with open('./config/prompt.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config['evaluator']

def format_prompt(template: str, question: str, answer: str, llm_output: str) -> str:
    return (
        f"{template}\n"
        f"{question}\n"
        f"Correct answer: {answer}\n"
        f"LLM output: {llm_output}"
    )

def evaluate_response(client: Any, model_name: str, question: str, answer: str, llm_output: str) -> str:
    try:
        template = load_prompt_template()
        prompt = format_prompt(template, question, answer, llm_output)
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def save_result_streaming(result: Dict[str, Any], output_file: str, is_first: bool = False) -> None:
    mode = 'w' if is_first else 'a'
    with open(output_file, mode, encoding='utf-8') as f:
        if is_first:
            f.write('[\n  ')
        else:
            f.write(',\n  ')
        json_str = json.dumps(result, ensure_ascii=False, indent=2)
        json_str = json_str.replace('\n', '\n  ')
        f.write(json_str)

def process_data(data: List[Dict[str, Any]], client: Any, model_name: str, output_file: str) -> None:
    print("Processing data...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Print the first prompt as an example
    if data:
        template = load_prompt_template()
        first_prompt = format_prompt(template, data[0]['question'], data[0]['answer'], data[0]['llm_output'])
        print("\nExample prompt for the first item:")
        print("-" * 80)
        print(first_prompt)
        print("-" * 80)
    
    for i, item in enumerate(tqdm(data, desc="Evaluating responses")):
        eval_result = evaluate_response(
            client, model_name,
            item['question'],
            item['answer'],
            item['llm_output']
        )
        result = {
            'question': item['question'],
            'answer': item['answer'],
            'llm_output': item['llm_output'],
            'eval': eval_result
        }
        save_result_streaming(result, output_file, is_first=(i==0))
    
    # Close the JSON array with proper indentation
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write('\n]')

def main() -> None:
    print("Starting evaluation...")
    args = get_args()
    client, model_name = setup_client(args.api_key)
    data = load_data(args.data_file)
    process_data(data, client, model_name, args.output_file)
    print("Evaluation completed!")

if __name__ == "__main__":
    main()