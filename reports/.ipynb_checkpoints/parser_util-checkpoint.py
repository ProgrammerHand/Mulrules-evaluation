import json
from collections import defaultdict
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def load_entries_to_df(file_path):
    entries = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if data.get("type") == "entry":
                instance_name = data.get("entry_name")
                instance = data.get("instance", {})
                features = instance.get("features", {})
                original = instance.get("original_outcome")
                predicted = instance.get("predicted_outcome")

                entry_dict = {
                    "Instance_Name": instance_name,
                    "Original_Outcome": original,
                    "Predicted_Outcome": predicted,
                    **features
                }
                entries.append(entry_dict)
    df = pd.DataFrame(entries)
    return df


def load_and_group_rules(file_path):
    grouped_rules = defaultdict(lambda: defaultdict(list))

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if data.get("type") == "rule":
                premises = data["rule"]["premises"]
                premises_str = " AND ".join(f'{p["attr"]} {p["op"]} {p["val"]}' for p in premises)
                consequence = data["rule"]["consequence"]
                consequence_str = f'{consequence["attr"]} {consequence["op"]} {consequence["val"]}'
                rule_str = f'IF {premises_str} THEN {consequence_str}'

                metrics = data.get("metrics", {})
                rule_dict = {
                    "Premises": premises,
                    "Rule_ID": None,  # You can fill this later if needed
                    "Rule": rule_str,
                    "Cov": round(metrics.get("coverage", 0), 5),
                    "Cov_class": round(metrics.get("local_class_coverage", 0), 5),
                    "Pre": round(metrics.get("precision", 0), 5),
                    "Len": metrics.get("rule_length", 0),
                    "Reject": metrics.get("rejected", 0),
                    "Elapsed_time": round(metrics.get("elapsed_time", 0), 5),
                    "Iter_Limit": data.get("at_limit", False)
                }

                instance_name = data.get("entry_name")
                explainer = data.get("explainer")
                grouped_rules[instance_name][explainer].append(rule_dict)

    return grouped_rules


def grouped_rules_to_df(grouped_rules):
    rows = []
    for instance_name, explainers_dict in grouped_rules.items():
        for explainer, rules_list in explainers_dict.items():
            for i, rule_dict in enumerate(rules_list, start=1):
                row = {
                    "Instance_Name": instance_name,
                    "Explainer": explainer,
                }
                row.update(rule_dict)
                row["Rule_ID"] = f"{explainer}{i}"
                rows.append(row)
    return pd.DataFrame(rows)


def filter_non_dominated(df, cov_col='Cov', pre_col='Pre'):
    non_dominated_indices = []
    for i, row_i in df.iterrows():
        dominated = False
        for j, row_j in df.iterrows():
            if i != j:
                if (row_j[cov_col] >= row_i[cov_col] and
                        row_j[pre_col] >= row_i[pre_col] and
                        (row_j[cov_col] > row_i[cov_col] or row_j[pre_col] > row_i[pre_col])):
                    dominated = True
                    break
        if not dominated:
            non_dominated_indices.append(i)

    return df.loc[non_dominated_indices].reset_index(drop=True)


def filter_non_dominated_3d(df, cov_col='Cov_class', pre_col='Pre', len_col='Len'):
    # convert to numpy for speed
    data = df[[cov_col, pre_col, len_col]].to_numpy()

    non_dominated_mask = []
    for i, candidate in enumerate(data):
        dominated = False
        for j, other in enumerate(data):
            if i == j:
                continue
            # check if 'other' dominates 'candidate':
            # other is better or equal in all objectives AND strictly better in at least one
            # for cov_col and pre_col (to maximize): other >= candidate
            # for len_col (to minimize): other <= candidate
            if (other[0] >= candidate[0] and
                    other[1] >= candidate[1] and
                    other[2] <= candidate[2] and
                    (other[0] > candidate[0] or other[1] > candidate[1] or other[2] < candidate[2])):
                dominated = True
                break
        non_dominated_mask.append(not dominated)

    return df.loc[non_dominated_mask]


def plot_non_dominated_rules(non_dominated_rules, instance_name):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=non_dominated_rules,
        x='Cov',
        y='Pre',
        hue='Explainer',
        style='Explainer',
        palette='tab10',
        s=100
    )

    plt.title(f"Non-dominated Rules for Instance {instance_name} (Coverage ↑, Precision ↑)")
    plt.xlabel("Coverage")
    plt.ylabel("Precision")
    plt.legend(title='Explainer', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def count_unique_attributes_from_rules(df_rules):
    attr_counter = Counter()

    for premises in df_rules['Premises']:
        seen_attrs = set()
        for cond in premises:
            attr = cond['attr']
            seen_attrs.add(attr)
        for attr in seen_attrs:
            attr_counter[attr] += 1  # Count only once per rule
    return dict(attr_counter)


def build_attr_usage_df(df_rules):
    rows = []
    for explainer, group in df_rules.groupby('Explainer'):
        attr_counts = count_unique_attributes_from_rules(group)
        for attr, count in attr_counts.items():
            rows.append({"Explainer": explainer, "Feature": attr, "Count": count})
    return pd.DataFrame(rows)


def plot_feature_usage_heatmap(df: pd.DataFrame,
                               feature_col: str,
                               explainer_col: str,
                               count_col: str,
                               all_features: list,
                               vmax: float = None,
                               figsize=(10, 8),
                               cmap='YlOrRd'):
    if df.empty:
        print("Empty dataframe: nothing to plot.")
        return
    # pivot table with counts: rows=features, columns=explainers
    heatmap_data = df.pivot_table(index=feature_col, columns=explainer_col, values=count_col, fill_value=0)
    heatmap_data = heatmap_data.reindex(all_features, fill_value=0)

    explainers_with_features = heatmap_data.columns.tolist()
    top_feature_names = heatmap_data.index.tolist()
    actual_max_count = heatmap_data.values.max()
    theoretical_max = vmax if vmax is not None else actual_max_count

    if heatmap_data is not None and theoretical_max > 0:
        plt.figure(figsize=figsize)
        plt.imshow(heatmap_data, cmap=cmap, vmax=theoretical_max, aspect='auto')
        plt.colorbar(label='Rule Count')
        plt.xticks(ticks=range(len(explainers_with_features)), labels=explainers_with_features, rotation=45, ha='right')
        plt.yticks(ticks=range(len(top_feature_names)), labels=top_feature_names)
        plt.title(f'Feature Usage Across Explainers\n(Scale: 0 to {theoretical_max}, Actual max: {actual_max_count})')
        plt.tight_layout()
        plt.show()
    else:
        print("No data to display or vmax <= 0.")


def plot_rules_comparison(all_rules, filtered_rules, instance_name):
    plt.figure(figsize=(8, 6))

    # palette for explainers across both datasets
    explainers = sorted(all_rules['Explainer'].unique())
    palette = sns.color_palette('tab10', n_colors=len(explainers))
    explainer_palette = dict(zip(explainers, palette))

    # all rules - circles
    sns.scatterplot(
        data=all_rules,
        x='Cov',
        y='Pre',
        hue='Explainer',
        palette=explainer_palette,
        s=60,
        alpha=0.5,
        marker='o'
    )

    # filtered rules - X markers with edgecolors
    sns.scatterplot(
        data=filtered_rules,
        x='Cov',
        y='Pre',
        hue='Explainer',
        palette=explainer_palette,
        s=120,
        marker='X',
        edgecolor='black'
    )

    # lines connecting filtered rules sorted by Coverage
    filtered_sorted = filtered_rules.sort_values(by='Cov')
    plt.plot(filtered_sorted['Cov'], filtered_sorted['Pre'], color='gray', linestyle='--', linewidth=1)

    plt.title(f"Rules for Instance {instance_name} (Coverage ↑, Precision ↑)")
    plt.xlabel("Coverage")
    plt.ylabel("Precision")

    handles, labels = plt.gca().get_legend_handles_labels()

    by_label = dict(zip(labels, handles))

    plt.legend(by_label.values(), by_label.keys(), title='Explainer', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.grid(True)
    plt.tight_layout()
    plt.show()


def summarize_explainer_metrics_with_global_average(explainer_dict):
    summary = {}
    all_rows = []

    for explainer, metrics_list in explainer_dict.items():
        df = pd.DataFrame(metrics_list)
        summary[explainer] = df.mean()
        all_rows.append(df)

    df_summary = pd.DataFrame(summary).T

    global_df = pd.concat(all_rows, ignore_index=True)
    df_summary.loc["Global_Average"] = global_df.mean()

    return df_summary