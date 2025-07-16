from collections import defaultdict
import pandas as pd
import seaborn as sns
from IPython.display import display, Markdown
import parser_util

def analyze_results (rules_log, entries_log, path = "./experiments_log/", min_cov = 0.1, min_cov_class = 0.1, min_pre = 0.1):
    rules_grouped = parser_util.load_and_group_rules(path + rules_log)
    df_instances = parser_util.load_entries_to_df(path + entries_log)
    df_rules = parser_util.grouped_rules_to_df(rules_grouped)

    explainers = sorted(df_rules['Explainer'].unique())
    palette = sns.color_palette('tab10', n_colors=len(explainers))
    explainer_palette = dict(zip(explainers, palette))

    instance_names = df_rules['Instance_Name'].unique()
    filtered_counts = defaultdict(list)
    non_dom1_by_explainer = defaultdict(list)
    non_dom2_by_explainer = defaultdict(list)
    filtered_counts_by_explainer = {
        'Correct Prediction': defaultdict(list),
        'Threshold Filter': defaultdict(list),
        'Non-dominated 1': defaultdict(list),
        'Non-dominated 2': defaultdict(list),
        'Unique Non-dominated 1': defaultdict(list),
        'Unique Non-dominated 2': defaultdict(list),
    }
    agg_all_dom1 = []
    agg_all_dom2 = []
    
    for i, instance_name in enumerate(instance_names, start=1):
        max_rules = df_rules[df_rules['Instance_Name'] == instance_name].groupby('Explainer').size().max()
        orig = df_instances.loc[df_instances['Instance_Name'] == instance_name, "Original_Outcome"].iloc[0]
        pred = df_instances.loc[df_instances['Instance_Name'] == instance_name, "Predicted_Outcome"].iloc[0]
        
        display(Markdown(f"## Instance {instance_name} (Original: {orig} , Predicted: {pred})"))
        instance = df_instances[df_instances['Instance_Name'] == instance_name]
        exclude_cols = ['Instance_Name', 'Original_Outcome', 'Predicted_Outcome']
        
        attributes = [col for col in instance.columns if col not in exclude_cols]
        display(instance.drop(columns=['Instance_Name', 'Original_Outcome', 'Predicted_Outcome']).T.rename(columns={df_instances[df_instances['Instance_Name'] == instance_name].index[0]: 'Value'}))
        
        display(Markdown(f"### Rules for Instance {instance_name}"))
        round_cols_names = ["Cov", "Cov_class", "Pre", "Elapsed_time", "Len"]
        display(parser_util.round_cols(df_rules[df_rules['Instance_Name'] == instance_name].drop(columns=["Premises"]).reset_index(drop=True), round_cols_names))
       
        display(Markdown(f"### Rules for Instance {instance_name}, Correct Prediction"))
        # original_outcome = df_instances.loc[df_instances['Instance_Name'] == instance_name, 'Original_Outcome'].values[0]
        predicted_outcome = df_instances.loc[df_instances['Instance_Name'] == instance_name, 'Predicted_Outcome'].values[0]
        correct_pred_rules = df_rules[
        (df_rules['Instance_Name'] == instance_name) &
        (df_rules['Rule'].str.contains(f'class = {predicted_outcome}', na=False))
        ]
        display(parser_util.round_cols(correct_pred_rules.drop(columns=["Premises"]).reset_index(drop=True), round_cols_names))

        display(Markdown(f"### Rules for Instance {instance_name}, Min_treshold (Cov {min_cov}, Cov_class {min_cov_class}, Pre {min_pre})"))
        tresholded_rules = correct_pred_rules[
        (correct_pred_rules['Cov'] >= min_cov) &
        (correct_pred_rules['Cov_class'] >= min_cov_class) &
        (correct_pred_rules['Pre'] >= min_pre)
        ]
        display(parser_util.round_cols(tresholded_rules.drop(columns=["Premises"]).reset_index(drop=True), round_cols_names))
    
        non_dominated_rules1 = parser_util.filter_non_dominated(tresholded_rules)
        non_dominated_rules1['Len'] = (non_dominated_rules1['Len'] - 0) / (non_dominated_rules1['Len'].max() - 0) # normilize len using minmax
        ideal_point = [0,0]
        if not non_dominated_rules1.empty:
            ideal_rule_id, ideal_point = parser_util.ideal_point_rule_2d(non_dominated_rules1)
            ideal_rule_idx = non_dominated_rules1[non_dominated_rules1["Rule_ID"] == ideal_rule_id].index[0]
        display(Markdown(f"### Rules for Instance {instance_name}, Non-dominated (Cov↑, Pre↑), Ideal (Cov: {ideal_point[0]}, Pre: {ideal_point[1]})"))
        df_display = non_dominated_rules1.drop(columns=["Premises"])
        round_cols_names.append("Distance_idp_eucl")
        display(parser_util.round_cols(df_display, round_cols_names).apply(parser_util.highlight_row, highlight_idx=ideal_rule_idx, axis=1))
        if not non_dominated_rules1.empty:
            parser_util.plot_rules_comparison(all_rules=df_rules[df_rules['Instance_Name'] == instance_name],
                                          filtered_rules=non_dominated_rules1, explainer_palette=explainer_palette,
                                          instance_name=instance_name, ideal_rule_idx=ideal_rule_idx, ideal_point=ideal_point)

        non_dominated_rules1_unique = parser_util.filter_duplicates_supersets(non_dominated_rules1)
        if not non_dominated_rules1_unique.empty:
            ideal_rule_idx = non_dominated_rules1_unique[non_dominated_rules1_unique["Rule_ID"] == ideal_rule_id].index[0]
        display(Markdown(f"### Rules for Instance {instance_name}, Non-dominated (Cov↑, Pre↑), Ideal (Cov: {ideal_point[0]}, Pre: {ideal_point[1]}), Unique rules (diffrent features)"))
        df_display = non_dominated_rules1_unique.drop(columns=["Premises"])
        display(parser_util.round_cols(df_display, round_cols_names).apply(parser_util.highlight_row, highlight_idx=ideal_rule_idx, axis=1))

        
        if not non_dominated_rules1_unique.empty:
            parser_util.plot_non_dominated_rules(non_dominated_rules1_unique, explainer_palette, instance_name, ideal_rule_idx, ideal_point)
            parser_util.plot_rules_comparison(all_rules=df_rules[df_rules['Instance_Name'] == instance_name],
                                          filtered_rules=non_dominated_rules1_unique, explainer_palette=explainer_palette,
                                          instance_name=instance_name, ideal_rule_idx=ideal_rule_idx, ideal_point=ideal_point)
        agg_df = parser_util.build_attr_usage_df(non_dominated_rules1_unique)
        agg_all_dom1.append(agg_df)
        parser_util.plot_feature_usage_heatmap(agg_df, feature_col="Feature", explainer_col="Explainer", count_col="Count", all_features=attributes, vmax = max_rules)
    
        non_dominated_rules2 = parser_util.filter_non_dominated_3d(tresholded_rules)
        ideal_point = [0,0,0]
        non_dominated_rules2['Len'] = (non_dominated_rules2['Len'] - 0) / (non_dominated_rules2['Len'].max() - 0) # normilize len using minmax
        if not non_dominated_rules2.empty:
            ideal_rule_id, ideal_point = parser_util.ideal_point_rule_3d(non_dominated_rules2)
            ideal_rule_idx = non_dominated_rules2[non_dominated_rules2["Rule_ID"] == ideal_rule_id].index[0]
        display(Markdown(f"### Rules for Instance {instance_name}, Non-dominated (Cov_class↑, Pre↑, Len↓), Ideal (Cov class: {ideal_point[0]}, Pre: {ideal_point[1]}, Len: {round(ideal_point[2], 5)})"))
        df_display = non_dominated_rules2.drop(columns=["Premises"]).reset_index(drop=True)
        display(parser_util.round_cols(df_display, round_cols_names).apply(parser_util.highlight_row, highlight_idx=ideal_rule_idx, axis=1))

        non_dominated_rules2_unique = parser_util.filter_duplicates_supersets(non_dominated_rules2)
        if not non_dominated_rules2_unique.empty:
            ideal_rule_idx = non_dominated_rules2[non_dominated_rules2["Rule_ID"] == ideal_rule_id].index[0]
        display(Markdown(f"### Rules for Instance {instance_name}, Non-dominated (Cov_class↑, Pre↑, Len↓), Ideal (Cov class: {ideal_point[0]}, Pre: {ideal_point[1]}, Len: {round(ideal_point[2], 5)}, Unique rules (diffrent features)"))
        df_display = non_dominated_rules2_unique.drop(columns=["Premises"])
        display(parser_util.round_cols(df_display, round_cols_names).apply(parser_util.highlight_row, highlight_idx=ideal_rule_idx, axis=1))
        
        agg_df = parser_util.build_attr_usage_df(non_dominated_rules2_unique)
        agg_all_dom2.append(agg_df)
        parser_util.plot_feature_usage_heatmap(agg_df, feature_col="Feature", explainer_col="Explainer", count_col="Count", all_features=attributes, vmax = max_rules)
    
        all_rules_count = len(df_rules[df_rules['Instance_Name'] == instance_name])
        correct_pred_count = len(correct_pred_rules)
        tresholded_count = len(tresholded_rules)
        non_dom1_count = len(non_dominated_rules1)
        non_dom2_count = len(non_dominated_rules2)
        unique_non_dom1_count = len(non_dominated_rules1_unique)
        unique_non_dom2_count = len(non_dominated_rules2_unique)
        
        # filtered counts
        filtered_counts['Correct Prediction'].append(all_rules_count - correct_pred_count)
        filtered_counts['Threshold Filter'].append(correct_pred_count - tresholded_count)
        filtered_counts['Non-dominated 1'].append(tresholded_count - non_dom1_count)
        filtered_counts['Unique Non-dominated 1'].append(non_dom1_count - unique_non_dom1_count)
        filtered_counts['Non-dominated 2'].append(tresholded_count - non_dom2_count)
        filtered_counts['Unique Non-dominated 2'].append(non_dom2_count - unique_non_dom2_count)

        def compute_explainer_diff(df_before, df_after, stage_name):
            before_counts = df_before['Explainer'].value_counts()
            after_counts = df_after['Explainer'].value_counts()
            diff = (before_counts - after_counts).fillna(before_counts).astype(int)
            for explainer, count in diff.items():
                filtered_counts_by_explainer[stage_name][explainer].append(count)

        original_rules= df_rules[df_rules['Instance_Name'] == instance_name]
        compute_explainer_diff(original_rules, correct_pred_rules, 'Correct Prediction')
        compute_explainer_diff(correct_pred_rules, tresholded_rules, 'Threshold Filter')
        compute_explainer_diff(tresholded_rules, non_dominated_rules1, 'Non-dominated 1')
        compute_explainer_diff(tresholded_rules, non_dominated_rules2, 'Non-dominated 2')
        compute_explainer_diff(non_dominated_rules1, non_dominated_rules1_unique, 'Unique Non-dominated 1')
        compute_explainer_diff(non_dominated_rules2, non_dominated_rules2_unique, 'Unique Non-dominated 2')
        # original_rules_instance = df_rules[df_rules['Instance_Name'] == instance_name]
        # orig_counts = original_rules_instance['Explainer'].value_counts()
        # correct_pred_counts = correct_pred_rules['Explainer'].value_counts()
        # filtered_per_explainer_cp = (orig_counts - correct_pred_counts).fillna(orig_counts).astype(int)
        # for explainer, count in filtered_per_explainer_cp.items():
        #     filtered_counts_by_explainer['Correct Prediction'][explainer].append(count)
        # tresholded_counts = tresholded_rules['Explainer'].value_counts()
        # filtered_per_explainer_thresh = (correct_pred_counts - tresholded_counts).fillna(correct_pred_counts).astype(int)
        # for explainer, count in filtered_per_explainer_thresh.items():
        #     filtered_counts_by_explainer['Threshold Filter'][explainer].append(count)
        # non_dom1_counts = non_dominated_rules1['Explainer'].value_counts()
        # filtered_per_explainer_non_dom1 = (tresholded_counts - non_dom1_counts).fillna(tresholded_counts).astype(int)
        # for explainer, count in filtered_per_explainer_non_dom1.items():
        #     filtered_counts_by_explainer['Non-dominated 1'][explainer].append(count)
        # non_dom2_counts = non_dominated_rules2'Explainer'].value_counts()
        # filtered_per_explainer_non_dom2 = (tresholded_counts - non_dom2_counts).fillna(tresholded_counts).astype(int)
        # for explainer, count in filtered_per_explainer_non_dom2.items():
        #     filtered_counts_by_explainer['Non-dominated 2'][explainer].append(count)
    
        
        if not non_dominated_rules1_unique.empty:
            for explainer, group in non_dominated_rules1_unique.groupby("Explainer"):
                non_dom1_by_explainer[explainer].append(
                    group[['Pre', 'Cov', 'Cov_class', 'Len', 'Reject', 'Elapsed_time', 'Distance_idp_eucl']].mean()
                )
    
        if not non_dominated_rules2_unique.empty:
            for explainer, group in non_dominated_rules2_unique.groupby("Explainer"):
                non_dom2_by_explainer[explainer].append(
                    group[['Pre', 'Cov', 'Cov_class', 'Len', 'Reject', 'Elapsed_time', 'Distance_idp_eucl']].mean()
                )
        
    
    # average filtered rules per stage
    display(Markdown("### Average Number of Filtered Rules at Each Step"))
    filtration_summary = pd.DataFrame({stage: pd.Series(values).mean() for stage, values in filtered_counts.items()}, index=["Avg Filtered"]).T
    # display(filtration_summary)

    # per-explainer average per stage
    step_avg_dfs = {}
    for step, explainer_dict in filtered_counts_by_explainer.items():
        formatted_values = {}
        for expl, counts in explainer_dict.items():
            s = pd.Series(counts)
            total = s.sum()
            count = s.count()
            mean = s.mean()
            formatted_values[expl] = f"{mean:.2f} ({int(total)}/{count})"
        step_avg_dfs[step] = pd.Series(formatted_values, name=step)  # series with index=explainer, values=avg filtered; sum(each entry,each stage)/entries_amount
    
    combined_df = pd.concat(step_avg_dfs.values(), axis=1).fillna("0.00 (0/0)")
    combined_df.index.name = 'Explainer'
    combined_df.columns = [f"Avg Filtered ({col})" for col in combined_df.columns]
    
    mapping = {
        'Correct Prediction': 'Avg Filtered (Correct Prediction)',
        'Threshold Filter': 'Avg Filtered (Threshold Filter)',
        'Non-dominated 1': 'Avg Filtered (Non-dominated 1)',
        'Unique Non-dominated 1': 'Avg Filtered (Unique Non-dominated 1)',
        'Non-dominated 2': 'Avg Filtered (Non-dominated 2)',
        'Unique Non-dominated 2': 'Avg Filtered (Unique Non-dominated 2)'
    }
    new_row = {}
    for step, col_name in mapping.items():
        values = filtered_counts.get(step, [])
        total = sum(values)
        count = len(values)
        mean = total / count if count > 0 else 0
        new_row[col_name] = f"{mean:.2f} ({total}/{count})"
    
    combined_df.loc['Overall Average'] = pd.Series(new_row)
    display(combined_df)
    
    # average metrics for non-dominated rules
    display(Markdown("### Average Metrics for Non-Dominated Unique Rules (Cov↑, Pre↑)"))
    explainer_summary1 = parser_util.summarize_explainer_metrics_with_global_average(non_dom1_by_explainer)
    display(explainer_summary1)

    display(Markdown("### Average Metrics for Non-Dominated Unique Rules (Cov_class↑, Pre↑, Len↓)"))
    explainer_summary2 = parser_util.summarize_explainer_metrics_with_global_average(non_dom2_by_explainer)
    display(explainer_summary2)

    agg_all_dom1_df = pd.concat(agg_all_dom1).groupby(['Feature', 'Explainer']).sum().reset_index()
    agg_all_dom2_df = pd.concat(agg_all_dom2).groupby(['Feature', 'Explainer']).sum().reset_index()

    all_features = [col for col in df_instances.columns if col not in ['Instance_Name', 'Original_Outcome', 'Predicted_Outcome']]
    
    display(Markdown("## Overall Heatmap – Non-dominated Rules (Cov↑, Pre↑)"))
    parser_util.plot_feature_usage_heatmap(agg_all_dom1_df, feature_col="Feature", explainer_col="Explainer", count_col="Count", all_features=all_features, vmax =max_rules * len(instance_names))
    
    display(Markdown("## Overall Heatmap – Non-dominated Rules (Cov_class↑, Pre↑, Len↓)"))
    parser_util.plot_feature_usage_heatmap(agg_all_dom2_df, feature_col="Feature", explainer_col="Explainer", count_col="Count", all_features=all_features, vmax =max_rules * len(instance_names))