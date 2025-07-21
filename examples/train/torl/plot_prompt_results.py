import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

EXCLUDED_KEYS = {"n_success", "n_total", "success_rate"}


def infer_variable_type(series):
    """Infer if variable is continuous or discrete."""
    unique_vals = series.dropna().unique()
    if series.dtype == bool or len(unique_vals) <= 5 or all(isinstance(x, str) for x in unique_vals):
        return "categorical"
    return "continuous"


def plot_variable(df, var, output_dir):
    var_type = infer_variable_type(df[var])
    plt.figure(figsize=(6, 4))

    if var_type == "categorical":
        sns.boxplot(data=df, x=var, y="success_rate")
    else:
        sns.scatterplot(data=df, x=var, y="success_rate")
        sns.lineplot(data=df.groupby(var)["success_rate"].mean().reset_index(), x=var, y="success_rate", color='red', label='mean')

    # set ylim lower bound to 0
    plt.ylim(0, 1)

    plt.title(f"Success Rate vs {var}")
    plt.ylabel("Success Rate")
    plt.xlabel(var)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{var}_vs_success_rate.png"))
    plt.close()


def main(input_jsonl, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    records = list()

    # Load input
    with open(input_jsonl) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e} in line: {line.strip()}")

    df = pd.DataFrame(records)

    # Filter relevant columns
    candidate_vars = [c for c in df.columns if c not in EXCLUDED_KEYS]

    for var in candidate_vars:
        if df[var].nunique() > 1:
            print(f"Plotting {var}")
            plot_variable(df, var, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", help="Path to JSONL file with experiment records")
    parser.add_argument("--output_dir", default="plots", help="Directory to save plots")
    args = parser.parse_args()

    main(args.input_jsonl, args.output_dir)
