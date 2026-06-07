import json
import os
import re
from typing import Any
import pandas as pd


def normalize_label(label_text: str) -> str:
    if label_text.lower().startswith("support"):
        return "Supports"
    if label_text.lower().startswith("oppose"):
        return "Opposes"
    if label_text.lower() == "either":
        return "Either"
    return "N/A"


def extract_label_from_raw_output(raw_output: Any) -> tuple[str, str]:
    if not isinstance(raw_output, str):
        return "N/A", "None"

    label_variations_pattern = r"(Support(?:s|es)?|Oppose(?:s)?|Either)"

    match_rule1 = re.search(
        r"(?:Fast thinking Label:|Slow thinking label:)\s*(" + label_variations_pattern + r")",
        raw_output,
        re.IGNORECASE,
    )
    if match_rule1:
        return normalize_label(match_rule1.group(1)), "Rule 1"

    match_rule2 = re.search(
        r"(?:Fast thinking label|Slow thinking label)[\s\S]*?(" + label_variations_pattern + r")",
        raw_output,
        re.IGNORECASE,
    )
    if match_rule2:
        return normalize_label(match_rule2.group(1)), "Rule 2"

    match_rule3 = re.search(
        r"Solution[\s\S]*?(" + label_variations_pattern + r")",
        raw_output,
        re.IGNORECASE | re.DOTALL,
    )
    if match_rule3:
        return normalize_label(match_rule3.group(1)), "Rule 3"

    match_rule4 = re.search(label_variations_pattern, raw_output, re.IGNORECASE)
    if match_rule4:
        return normalize_label(match_rule4.group(1)), "Rule 4"

    return "N/A", "None"


def parse_results(data_frame: pd.DataFrame, eval_type: str = "not") -> pd.DataFrame:
    df = data_frame.copy()

    if eval_type == "anchor":
        fast_raw_col = "llm_raw_output_fast_anchored"
        slow_raw_col = "llm_raw_output_slow_anchored"
        fast_label_col = "llm_fast_thinking_label_anchored"
        slow_label_col = "llm_slow_thinking_label_anchored"
        fast_rule_col = "llm_fast_thinking_rule_anchored"
        slow_rule_col = "llm_slow_thinking_rule_anchored"
    else:
        fast_raw_col = "llm_raw_output_fast"
        slow_raw_col = "llm_raw_output_slow"
        fast_label_col = "llm_fast_thinking_label"
        slow_label_col = "llm_slow_thinking_label"
        fast_rule_col = "llm_fast_thinking_rule"
        slow_rule_col = "llm_slow_thinking_rule"

    df[fast_label_col], df[fast_rule_col] = zip(*df[fast_raw_col].apply(extract_label_from_raw_output))
    df[slow_label_col], df[slow_rule_col] = zip(*df[slow_raw_col].apply(extract_label_from_raw_output))

    df["fast_correct"] = df[fast_label_col] == df["valence"].astype(int)
    df["slow_correct"] = df[slow_label_col] == df["valence"].astype(int)
    return df


def eval(data_frame: pd.DataFrame, dataset_name: str = "Dataset") -> dict[str, Any]:
    results: dict[str, Any] = {"Dataset": dataset_name}

    total_data_points = len(data_frame)
    fast_correct_count = int(data_frame["fast_correct"].sum())
    slow_correct_count = int(data_frame["slow_correct"].sum())

    fast_accuracy = fast_correct_count / total_data_points if total_data_points > 0 else 0.0
    slow_accuracy = slow_correct_count / total_data_points if total_data_points > 0 else 0.0

    results["Fast Correct Count"] = fast_correct_count
    results["Slow Correct Count"] = slow_correct_count
    results["Fast Accuracy"] = f"{fast_accuracy:.2%}"
    results["Slow Accuracy"] = f"{slow_accuracy:.2%}"

    capability_knowledge_retrieval = fast_accuracy
    capability_reasoning_adjustment = slow_accuracy - fast_accuracy
    results["Capability Knowledge Retrieval (CKR)"] = f"{capability_knowledge_retrieval:.2%}"
    results["Capability Reasoning Adjustment (CRA)"] = f"{capability_reasoning_adjustment:.2%}"

    overthinking_count = data_frame[(data_frame["fast_correct"] == 1) & (data_frame["slow_correct"] == 0)].shape[0]
    results["Overthinking Count"] = int(overthinking_count)

    fast_correct_total_for_rate = data_frame[data_frame["fast_correct"] == 1].shape[0]
    if fast_correct_total_for_rate > 0:
        overthinking_rate = overthinking_count / fast_correct_total_for_rate
        results["Overthinking Rate"] = f"{overthinking_rate:.2%}"
    else:
        results["Overthinking Rate"] = "N/A (No fast correct predictions)"

    correction_count = data_frame[(data_frame["fast_correct"] == 0) & (data_frame["slow_correct"] == 1)].shape[0]
    results["Correction Count"] = int(correction_count)

    fast_incorrect_total_for_rate = data_frame[data_frame["fast_correct"] == 0].shape[0]
    if fast_incorrect_total_for_rate > 0:
        correction_rate = correction_count / fast_incorrect_total_for_rate
        results["Correction Rate"] = f"{correction_rate:.2%}"
    else:
        results["Correction Rate"] = "N/A (No fast incorrect predictions)"

    return results


def evaluate_json_file(file_path: str, dataset_name: str = "Dataset", eval_type: str = "not") -> dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    df = pd.DataFrame(raw_data)
    parsed_df = parse_results(df, eval_type=eval_type)
    metrics = eval(parsed_df, dataset_name=dataset_name)
    return metrics


def save_metrics(metrics: dict[str, Any], output_file: str) -> None:
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main() -> dict[str, Any]:
    input_files = [
        "data/output/qwen2.5_1.5B_results.json",
        "data/output/qwen2.5_3B_results.json",
        "data/output/qwen2.5_7B_results.json",
        "data/output/qwen2.5_7B_anchored_results.json",
    ]
    output_file = "data/output/combined_evaluation_metrics.json"

    combined_metrics: dict[str, Any] = {"evaluations": []}

    for file_path in input_files:
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        metrics = evaluate_json_file(
            file_path,
            dataset_name=dataset_name,
            eval_type="anchor" if "anchored" in dataset_name else "not"       
        )
        combined_metrics["evaluations"].append(
            {
                "dataset": dataset_name,
                "metrics": metrics,
            }
        )

    save_metrics(combined_metrics, output_file)
    return combined_metrics


if __name__ == "__main__":
    main()
