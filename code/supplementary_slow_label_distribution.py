import json
import os
import re
from typing import Any
import pandas as pd

LABEL_PATTERN = r"(Support(?:s|es)?|Oppose(?:s)?|Either)"


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

	match_rule1 = re.search(
		r"(?:Fast thinking Label:|Slow thinking label:)\s*" + LABEL_PATTERN,
		raw_output,
		re.IGNORECASE,
	)
	if match_rule1:
		return normalize_label(match_rule1.group(1)), "Rule 1"

	match_rule2 = re.search(
		r"(?:Fast thinking label|Slow thinking label)[\s\S]*?" + LABEL_PATTERN,
		raw_output,
		re.IGNORECASE,
	)
	if match_rule2:
		return normalize_label(match_rule2.group(1)), "Rule 2"

	match_rule3 = re.search(
		r"Solution[\s\S]*?" + LABEL_PATTERN,
		raw_output,
		re.IGNORECASE | re.DOTALL,
	)
	if match_rule3:
		return normalize_label(match_rule3.group(1)), "Rule 3"

	match_rule4 = re.search(LABEL_PATTERN, raw_output, re.IGNORECASE)
	if match_rule4:
		return normalize_label(match_rule4.group(1)), "Rule 4"

	return "N/A", "None"


def load_results(path: str) -> list[dict[str, Any]]:
	with open(path, "r", encoding="utf-8") as file_handle:
		return json.load(file_handle)


def compute_slow_label_distribution(records: list[dict[str, Any]]) -> pd.DataFrame:
	rows = []
	for record in records:
		slow_label, slow_rule = extract_label_from_raw_output(record.get("llm_raw_output_slow"))
		fast_label, _ = extract_label_from_raw_output(record.get("llm_raw_output_fast"))
		rows.append(
			{
				"id": record.get("id"),
				"valence": record.get("valence"),
				"fast_label": fast_label,
				"slow_label": slow_label,
				"slow_rule": slow_rule,
				"overthinking": fast_label == record.get("valence") and slow_label != record.get("valence"),
			}
		)

	return pd.DataFrame(rows)


def main() -> None:
	input_files = [
		("Qwen1.5B", "data/input/qwen2.5_1.5B_results.json"),
		("Qwen3B", "data/input/qwen2.5_3B_results.json"),
		("Qwen7B", "data/input/qwen2.5_7B_results.json"),
	]

	all_model_data = []

	for dataset_name, input_path in input_files:
		records = load_results(input_path)
		df = compute_slow_label_distribution(records)
		overthinking_df = df[df["overthinking"] == True].copy()  # noqa: E712
		distribution = overthinking_df["slow_label"].value_counts().reindex(["Either", "Supports", "Opposes"], fill_value=0)
		
		overthinking_count = overthinking_df.shape[0]
		if overthinking_count > 0:
			either_percent = round((distribution.get("Either", 0) / overthinking_count) * 100, 2)
			supports_percent = round((distribution.get("Supports", 0) / overthinking_count) * 100, 2)
			opposes_percent = round((distribution.get("Opposes", 0) / overthinking_count) * 100, 2)
		else:
			either_percent = 0.0
			supports_percent = 0.0
			opposes_percent = 0.0

		all_model_data.append(
			{
				"Model": dataset_name,
				"Overthinking Count": int(overthinking_count),
				"Either (%)": f"{either_percent}%",
				"Supports (%)": f"{supports_percent}%",
				"Opposes (%)": f"{opposes_percent}%",
			}
		)

	final_df = pd.DataFrame(all_model_data)
	output_dir = "data/output/analysis"
	os.makedirs(output_dir, exist_ok=True)
	output_file_name = os.path.join(output_dir, "slow_label_distribution.json")
	final_df.to_json(output_file_name, orient="records", indent=4)
	print(f"DataFrame saved to {output_file_name}")


if __name__ == "__main__":
	main()