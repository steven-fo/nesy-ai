import json
import os
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def pct_str_to_float(s: Any) -> float:
	try:
		if s is None:
			return float("nan")
		if isinstance(s, str) and s.endswith("%"):
			return float(s.rstrip("%")) / 100.0
		return float(s)
	except Exception:
		return float("nan")


def load_combined_metrics(path: str) -> pd.DataFrame:
	with open(path, "r", encoding="utf-8") as f:
		data = json.load(f)

	records = []
	for item in data.get("evaluations", []):
		dataset = item.get("dataset")
		metrics = item.get("metrics", {}) or {}
		flat = {"Dataset": dataset}
		# copy metrics keys as-is
		for k, v in metrics.items():
			flat[k] = v
		records.append(flat)

	df = pd.DataFrame(records)
	return df


def plot_overthinking_and_correction(df: pd.DataFrame, outdir: str) -> None:
	chart_df = df[df["Dataset"] != "Qwen7B Anchor"].copy()
	if chart_df.empty:
		return

	# Convert percentage strings to float
	chart_df["Overthinking Rate"] = chart_df["Overthinking Rate"].apply(pct_str_to_float)
	chart_df["Correction Rate"] = chart_df["Correction Rate"].apply(pct_str_to_float)

	chart_melt = chart_df.melt(
		id_vars="Dataset",
		value_vars=["Overthinking Rate", "Correction Rate"],
		var_name="Metrik",
		value_name="Nilai (%)",
	)
	chart_melt = chart_melt.rename(columns={"Dataset": "Model"})

	plt.figure(figsize=(10, 6))
	ax = sns.lineplot(x="Model", y="Nilai (%)", hue="Metrik", data=chart_melt, marker="o")
	ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
	plt.tight_layout()
	os.makedirs(outdir, exist_ok=True)
	plt.savefig(os.path.join(outdir, "chart_overthinking_correction.png"))
	plt.close()


def plot_anchor(df: pd.DataFrame, outdir: str) -> None:
	chart_df = df[df["Dataset"].isin(["Qwen7B", "Qwen7B Anchor"])].copy()
	if chart_df.empty:
		return

	for col in ["Fast Accuracy", "Slow Accuracy", "Capability Reasoning Adjustment (CRA)"]:
		chart_df[col + " F"] = chart_df[col].apply(pct_str_to_float)

	chart_df = chart_df.rename(columns={"Fast Accuracy F": "Afast"})
	chart_df = chart_df.rename(columns={"Slow Accuracy F": "Aslow"})

	plot_df = chart_df.melt(
		id_vars="Dataset",
		value_vars=["Afast", "Aslow"],
		var_name="Metric",
		value_name="Value",
	)
	metric_order = ["Afast", "Aslow"]
	plot_df["Metric"] = pd.Categorical(plot_df["Metric"], categories=metric_order, ordered=True)

	plt.figure(figsize=(12, 7))
	sns.barplot(x="Dataset", y="Value", hue="Metric", data=plot_df, palette="coolwarm")
	plt.xlabel("Model", fontsize=14)
	plt.ylabel("Accuracy (%)", fontsize=14)
	plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))
	plt.grid(axis="y", linestyle="--", alpha=0.7)
	plt.xticks(rotation=0, ha="center", fontsize=12, fontstyle="normal")
	plt.yticks(fontsize=12)
	plt.legend(title="Metric", fontsize=12, title_fontsize=14)
	plt.tight_layout()
	os.makedirs(outdir, exist_ok=True)
	plt.savefig(os.path.join(outdir, "chart_anchor.png"))
	plt.close()


def plot_capability_improvements(df: pd.DataFrame, outdir: str) -> None:
	base_df = df[df["Dataset"] != "Qwen7B Anchor"].copy()
	if base_df.empty:
		return

	# Convert needed columns
	base_df["Slow Accuracy F"] = base_df["Slow Accuracy"].apply(pct_str_to_float)
	base_df["CKR F"] = base_df["Capability Knowledge Retrieval (CKR)"].apply(pct_str_to_float)
	base_df["CRA F"] = base_df["Capability Reasoning Adjustment (CRA)"].apply(pct_str_to_float)

	# Set index by Dataset for calculations
	base_df = base_df.set_index("Dataset")

	models = [m for m in ["Qwen1.5B", "Qwen3B", "Qwen7B"] if m in base_df.index]
	if not models:
		return

	imp_index = []
	for m in models:
		imp_index.append(m)

	improvement = pd.DataFrame(index=imp_index, columns=["Slow Accuracy Improvement", "CKR Improvement", "CRA Improvement"], dtype=float)

	# baseline is first model in list
	baseline = models[0]
	improvement.loc[baseline] = [0.0, 0.0, 0.0]

	for i in range(1, len(models)):
		cur = models[i]
		prev = models[i - 1]
		improvement.loc[cur, "Slow Accuracy Improvement"] = base_df.loc[cur, "Slow Accuracy F"] - base_df.loc[prev, "Slow Accuracy F"]
		improvement.loc[cur, "CKR Improvement"] = base_df.loc[cur, "CKR F"] - base_df.loc[prev, "CKR F"]
		improvement.loc[cur, "CRA Improvement"] = base_df.loc[cur, "CRA F"] - base_df.loc[prev, "CRA F"]

	improvement = improvement.reset_index().rename(columns={"index": "Model"})
	improvement = improvement.rename(columns={
		"Slow Accuracy Improvement": "Knowledge + Reasoning",
		"CKR Improvement": "Knowledge",
		"CRA Improvement": "Reasoning",	})

	plot_df = improvement.melt(id_vars="Model", value_vars=["Knowledge + Reasoning", "Knowledge", "Reasoning"], var_name="Capability Metric", value_name="Perubahan Nilai")

	plt.figure(figsize=(10, 6))
	sns.lineplot(x="Model", y="Perubahan Nilai", hue="Capability Metric", data=plot_df, marker="o")
	plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
	plt.tight_layout()
	os.makedirs(outdir, exist_ok=True)
	plt.savefig(os.path.join(outdir, "chart_capability_improvements.png"))
	plt.close()


def main(input_path: str = "data/output/combined_evaluation_metrics.json", outdir: str = "data/output/analysis") -> None:
	df = load_combined_metrics(input_path)

	plot_overthinking_and_correction(df, outdir)
	plot_anchor(df, outdir)
	plot_capability_improvements(df, outdir)


if __name__ == "__main__":
	main()
