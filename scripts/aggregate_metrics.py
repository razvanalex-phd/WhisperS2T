#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent))

from tools.metrics import evaluate
from tools.text_normalizer import TextNormalizer


def bold_best_value(values, maximize=False, is_inference_time=False):
    """
    Return a list of values with the best value (min or max) in bold
    Args:
        values: List of values
        maximize: If True, the highest value is the best, otherwise the lowest
        is_inference_time: If True, zero values are treated as missing data, not the best
    """
    if not values:
        return []

    # Convert None to appropriate infinity values
    processed_values = []
    for v in values:
        if v is None:
            processed_values.append(float("inf") if not maximize else float("-inf"))
        elif is_inference_time and v == 0:  # Special handling for zero inference times
            processed_values.append(float("inf"))  # Treat as missing
        else:
            processed_values.append(v)

    # Find best index, skipping zeros for inference time
    valid_indices = []
    valid_values = []

    for i, v in enumerate(processed_values):
        is_valid = not (is_inference_time and (values[i] == 0 or values[i] is None))
        if is_valid and v != float("inf") and v != float("-inf"):
            valid_indices.append(i)
            valid_values.append(v)

    if not valid_indices:
        best_idx = -1  # No valid values
    else:
        best_value_idx = (
            np.argmax(valid_values) if maximize else np.argmin(valid_values)
        )
        best_idx = valid_indices[best_value_idx]

    result = []
    for i, v in enumerate(values):
        if v is None:
            result.append("-")
        elif v == 0 and is_inference_time:
            result.append("-")  # Show zero inference times as "-"
        elif i == best_idx:
            result.append(f"**{v:.2f}**")
        else:
            result.append(f"{v:.2f}")

    return result


def read_results_directory(base_dir, normalize=None):
    """
    Read results from all model directories
    """
    model_results = {}
    model_dirs = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]

    for model_dir in model_dirs:
        full_dir = os.path.join(base_dir, model_dir)

        # Skip if not a results directory
        if not os.path.exists(os.path.join(full_dir, "infer_time.tsv")):
            continue

        # Read inference times
        try:
            infer_times = pd.read_csv(
                os.path.join(full_dir, "infer_time.tsv"), sep="\t"
            )
            infer_times_dict = {
                row["Dataset"]: row["Time"] for _, row in infer_times.iterrows()
            }
        except:
            infer_times_dict = {}

        # Read evaluation results
        eval_results = {}

        # Check for KINCAID46_WAV.tsv
        kincaid_path = os.path.join(full_dir, "KINCAID46_WAV.tsv")
        if os.path.exists(kincaid_path):
            df = pd.read_csv(kincaid_path, sep="\t")
            references = df["raw_text"].to_list()
            hypotheses = df["pred_text"].to_list()
            if normalize:
                references = [normalize(ref) for ref in references]
                hypotheses = [normalize(hyp) for hyp in hypotheses]
            metrics = evaluate(references, hypotheses)
            eval_results["KINCAID46_WAV"] = metrics

        # Check for MultiLingualLongform.tsv
        multilingual_path = os.path.join(full_dir, "MultiLingualLongform.tsv")
        if os.path.exists(multilingual_path):
            df = pd.read_csv(multilingual_path, sep="\t")
            references = df["raw_text"].to_list()
            hypotheses = df["pred_text"].to_list()
            if normalize:
                references = [normalize(ref) for ref in references]
                hypotheses = [normalize(hyp) for hyp in hypotheses]
            metrics = evaluate(references, hypotheses)
            eval_results["MultiLingualLongform"] = metrics

        model_results[model_dir] = {
            "infer_times": infer_times_dict,
            "eval_results": eval_results,
        }

    return model_results


def generate_report(model_results):
    """
    Generate a report with tables for inference times and evaluation metrics
    """
    # Extract model names
    model_names = sorted(list(model_results.keys()))

    # --- Inference Time Table ---
    datasets_infer = set()
    for model_data in model_results.values():
        datasets_infer.update(model_data["infer_times"].keys())
    datasets_infer = sorted(list(datasets_infer))

    # Collect data column-wise (per dataset) to apply bolding correctly
    infer_time_data_by_dataset = {}
    for dataset in datasets_infer:
        values = []
        for model in model_names:
            values.append(model_results[model]["infer_times"].get(dataset, None))
        # Bold the best (lowest non-zero) inference time for this dataset
        infer_time_data_by_dataset[dataset] = bold_best_value(
            values, maximize=False, is_inference_time=True
        )

    # Build the transposed table rows
    infer_time_table = []
    headers_infer = ["Model"] + datasets_infer
    for i, model in enumerate(model_names):
        row = [model]
        for dataset in datasets_infer:
            row.append(infer_time_data_by_dataset[dataset][i])
        infer_time_table.append(row)

    # --- Evaluation Metrics Tables ---
    metrics_tables_transposed = {}
    datasets_eval = set()
    for model_data in model_results.values():
        datasets_eval.update(model_data["eval_results"].keys())
    datasets_eval = sorted(list(datasets_eval))

    for dataset in datasets_eval:
        metrics_set = set()
        for model in model_names:
            if dataset in model_results[model]["eval_results"]:
                metrics_set.update(model_results[model]["eval_results"][dataset].keys())
        metrics = sorted(list(metrics_set))

        if not metrics:  # Skip dataset if no metrics found for any model
            continue

        # Collect data column-wise (per metric) to apply bolding correctly
        eval_data_by_metric = {}
        for metric in metrics:
            values = []
            for model in model_names:
                if (
                    dataset in model_results[model]["eval_results"]
                    and metric in model_results[model]["eval_results"][dataset]
                ):
                    values.append(model_results[model]["eval_results"][dataset][metric])
                else:
                    values.append(None)

            # For WER, CER, etc. lower is better; for accuracy metrics, higher is better
            maximize = metric.startswith("acc_")
            eval_data_by_metric[metric] = bold_best_value(values, maximize=maximize)

        # Build the transposed table rows for this dataset
        metrics_table = []
        headers_metrics = ["Model"] + metrics
        for i, model in enumerate(model_names):
            row = [model]
            for metric in metrics:
                row.append(eval_data_by_metric[metric][i])
            metrics_table.append(row)

        metrics_tables_transposed[dataset] = (headers_metrics, metrics_table)

    # Format the report
    report = "# Model Performance Report\n\n"
    report += "## Inference Times (seconds)\n\n"
    report += (
        tabulate(infer_time_table, headers=headers_infer, tablefmt="pipe") + "\n\n"
    )

    for dataset, (headers, metrics_table) in metrics_tables_transposed.items():
        report += f"## Evaluation Metrics for {dataset}\n\n"
        report += tabulate(metrics_table, headers=headers, tablefmt="pipe") + "\n\n"

    return report


if __name__ == "__main__":
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
    )

    normalize = TextNormalizer()
    print(f"Reading results from {results_dir}...")
    model_results = read_results_directory(results_dir, normalize)

    if not model_results:
        print("No results found.")
        sys.exit(1)

    print(f"Generating report for {len(model_results)} models...")
    report = generate_report(model_results)

    # Write the report to a file
    report_path = os.path.join(results_dir, "metrics_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Report generated and saved to {report_path}")
