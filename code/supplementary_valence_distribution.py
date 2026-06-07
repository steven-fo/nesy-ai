import os
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_valence_data(path: str) -> pd.DataFrame:
	return pd.read_csv(path)


def plot_valence_distribution(series: pd.Series, title: str, output_path: str) -> None:
	plt.figure(figsize=(8, 8))
	plt.pie(series.values, labels=series.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2"))
	plt.title(title)
	plt.ylabel('')
	plt.tight_layout()
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	plt.savefig(output_path, dpi=200)
	plt.close()


def main() -> dict[str, Any]:
	input_path = "data/input/valence_test.csv"
	output_dir = "data/output/analysis"
	os.makedirs(output_dir, exist_ok=True)

	df = load_valence_data(input_path)
	top_1000 = df.head(1000).copy()
	valence_distribution = top_1000["valence"].value_counts().reindex(["Supports", "Opposes", "Either"], fill_value=0)

	plot_valence_distribution(
		valence_distribution,
		title="Valence Distribution (Top 1000)",
		output_path=os.path.join(output_dir, "valence_distribution_top1000.png"),
	)


if __name__ == "__main__":
	main()