from collections import defaultdict
import pandas as pd
import numpy as np
import parser_util

STEP_NAMES = ["Original", "Correct_Prediction", "Threshold", "NonDominated", "Unique"]
CHOICE_METHODS = ["distance", "random", "sum", "product"]

def counts_by_explainer(df: pd.DataFrame, explainers: list[str]) -> pd.Series:
    return pd.Series(0, index=explainers, dtype=int) if df is None or df.empty \
           else df["Explainer"].value_counts().reindex(explainers, fill_value=0).astype(int)

def accum_counts(buckets: dict, step: str, counts: pd.Series):
    for e, v in counts.items(): buckets[step][e].append(int(v))

def non_dom_and_unique(t_rules: pd.DataFrame, category: int):
    if t_rules is None or t_rules.empty: 
        return t_rules, t_rules, None, False, None
    if category == 1:
        non_dom = parser_util.filter_non_dominated(t_rules)
        ideal_rule_id, _ = parser_util.ideal_point_rule_2d(non_dom) if not non_dom.empty else (None, None)
    else:
        non_dom = parser_util.filter_non_dominated_3d(t_rules)
        if not non_dom.empty:
            max_len = non_dom["Len"].max()
            if pd.notna(max_len) and max_len > 0:
                non_dom = non_dom.copy()
                non_dom["Len"] = non_dom["Len"] / max_len
        ideal_rule_id, _ = parser_util.ideal_point_rule_3d(non_dom) if not non_dom.empty else (None, None)
    unique = parser_util.filter_duplicates_supersets(non_dom) if not non_dom.empty else non_dom
    ideal_present = bool(ideal_rule_id is not None and (unique is not None and not unique.empty)
                         and (unique["Rule_ID"] == ideal_rule_id).any())
    ideal_explainer = None
    if ideal_present:
        ideal_explainer = unique.loc[unique["Rule_ID"] == ideal_rule_id, "Explainer"].iloc[0]
    return non_dom, unique, ideal_rule_id, ideal_present, ideal_explainer

def _summarize_per_explainer(unique_cat_df: pd.DataFrame, rules_per_instance: dict) -> pd.DataFrame:
    """
    General metrics per explainer:
      Cov, Cov_class, Pre, Len, Reject, Elapsed_time, Distance_idp_eucl,
      + Avg_Rules_Per_Instance (mean of per-instance counts; zeros included).
    """
    cols = ["Cov", "Cov_class", "Pre", "Len", "Reject", "Elapsed_time", "Distance_idp_eucl"]
    if unique_cat_df is None or unique_cat_df.empty:
        base = pd.DataFrame({"Explainer": sorted(rules_per_instance.keys())})
        base["Avg_Rules_Per_Instance"] = [
            pd.Series(rules_per_instance[e]).mean() if rules_per_instance.get(e) else 0.0
            for e in base["Explainer"]
        ]
        for c in cols:
            base[c] = float("nan")
        base = base[["Explainer"] + cols + ["Avg_Rules_Per_Instance"]]
        return base

    df = unique_cat_df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    grouped = df.groupby("Explainer")[cols].mean().reset_index()
    grouped["Avg_Rules_Per_Instance"] = grouped["Explainer"].map(
        lambda e: pd.Series(rules_per_instance.get(e, [])).mean() if rules_per_instance.get(e) else 0.0
    )
    grouped = grouped[["Explainer"] + cols + ["Avg_Rules_Per_Instance"]]
    return grouped.sort_values("Explainer").reset_index(drop=True)


def _pairwise_diffs(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    For every ordered pair of explainers (A, B), compute A - B across all numeric columns.
    Sorted by A then B. Output has NO helper columns (only 'Pair' + metrics).
    """
    if summary_df is None or summary_df.empty or "Explainer" not in summary_df.columns:
        return pd.DataFrame()

    explainers = sorted(summary_df["Explainer"])
    numeric_cols = [c for c in summary_df.columns if c != "Explainer"]

    rows = []
    for a in explainers:
        A = summary_df.loc[summary_df["Explainer"] == a].iloc[0]
        for b in explainers:
            if a == b:
                continue
            B = summary_df.loc[summary_df["Explainer"] == b].iloc[0]
            row = {"Pair": f"{a} - {b}"}
            for c in numeric_cols:
                row[c] = float(A[c]) - float(B[c])
            rows.append(row)

    return pd.DataFrame(rows, columns=["Pair"] + numeric_cols).reset_index(drop=True)

def _winner_matrix(chosen_by_method: dict, explainers: list, normalize: bool = False) -> pd.DataFrame:
    """
    Build Winner Matrix (Explainer x ChoiceMethod).
    chosen_by_method: dict like {"distance": [df_rows...], "random": [...], ...}
      where each list contains 1-row DataFrames of the chosen rule for an instance.
    If normalize=True, each column is divided by the number of selections made by that method
    (i.e., number of instances where a selection was possible), yielding ratios in [0,1].
    """
    cols = sorted(chosen_by_method.keys())
    data = {}

    for m in cols:
        if not chosen_by_method[m]:
            # no selections happened for this method
            counts = pd.Series(0, index=explainers, dtype=float)
            denom = 0
        else:
            df_m = pd.concat(chosen_by_method[m], ignore_index=True)
            counts_series = df_m['Explainer'].value_counts()
            # align to all explainers
            counts = pd.Series([counts_series.get(e, 0) for e in explainers], index=explainers, dtype=float)
            denom = float(len(chosen_by_method[m]))  # one winner per instance where selection was possible

        if normalize and denom > 0:
            data[m] = counts / denom
        else:
            data[m] = counts

    out = pd.DataFrame(data, index=explainers)
    # nice ordering & types
    return out.loc[sorted(out.index)].sort_index(axis=1)

def _select_rule_by_method(unique_df: pd.DataFrame, category: int, method: str, rng: np.random.Generator):
    """
    Choose exactly ONE rule from the non-dominated set for an instance.

    Methods:
      - 'distance': smallest Distance_idp_eucl (already computed earlier)
      - 'random': random row (uniform)
      - 'sum': max sum of category metrics (Cat1: Cov+Pre; Cat2: Cov_class+Pre+(1-Len))
      - 'product': max product of category metrics (Cat1: Cov*Pre; Cat2: Cov_class*Pre*(1-Len))
    """
    if unique_df is None or unique_df.empty:
        return None  # no selection possible

    df = unique_df.copy()
    epsilon = 1e-6  # safeguard for product mode to avoid zeroing out

    if method == "distance":
        if "Distance_idp_eucl" not in df.columns:
            if category == 1:
                cov_ideal, pre_ideal = df["Cov"].max(), df["Pre"].max()
                df["Distance_idp_eucl"] = np.sqrt((df["Cov"] - cov_ideal)**2 + (df["Pre"] - pre_ideal)**2)
            else:
                covc_ideal, pre_ideal, len_ideal = df["Cov_class"].max(), df["Pre"].max(), df["Len"].min()
                df["Distance_idp_eucl"] = np.sqrt(
                    (df["Cov_class"] - covc_ideal)**2
                    + (df["Pre"] - pre_ideal)**2
                    + (df["Len"] - len_ideal)**2
                )
        idx = df["Distance_idp_eucl"].idxmin()
        return df.loc[idx]

    if method == "random":
        idx = rng.integers(0, len(df))
        return df.iloc[int(idx)]

    if category == 1:
        if method == "sum":
            score = df["Cov"] + df["Pre"]
        elif method == "product":
            score = df["Cov"] * df["Pre"]
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        # Cat2: use gain = 1 - Len, but clamp with epsilon to avoid zero in product mode
        len_gain = 1.0 - df["Len"].clip(lower=0.0, upper=1.0)
        len_gain = len_gain.clip(lower=epsilon)
        if method == "sum":
            score = df["Cov_class"] + df["Pre"] + len_gain
        elif method == "product":
            score = df["Cov_class"] * df["Pre"] * len_gain
        else:
            raise ValueError(f"Unknown method: {method}")

    idx = score.idxmax()
    return df.loc[idx]

def _averages_table(step_buckets):
    """
    Convert step_buckets (step -> explainer -> list[counts]) to a DataFrame:
    rows = explainers, columns = steps, values = mean counts over instances.
    """
    explainers = sorted(next(iter(step_buckets.values())).keys())
    steps = list(step_buckets.keys())
    data = {}
    for s in steps:
        data[s] = [pd.Series(step_buckets[s][e]).mean() if step_buckets[s][e] else 0.0 for e in explainers]
    df = pd.DataFrame(data, index=explainers)
    df.index.name = "Explainer"
    return df

def _attribute_usage_score_for_instance(df_rules_inst, attribute_universe, epsilon=1e-12):
    """
    Normalized entropy of attribute usage per explainer, with
    p_i = count_i / total_rules (unchanged), but normalization by
    H_max = L * ln(N / L), where L = sum_i p_i = average rule length.
    Guarantees score in [0, 1].
    """
    N = max(len(attribute_universe), 1)
    scores = {}

    for e, grp in df_rules_inst.groupby("Explainer"):
        total_rules = len(grp)
        if total_rules == 0:
            scores[e] = 0.0
            continue

        # Count attribute presence (at most once per rule)
        counts = {attr: 0 for attr in attribute_universe}
        for _, row in grp.iterrows():
            attrs_in_rule = {p['attr'] for p in row['Premises']}
            for attr in attrs_in_rule:
                if attr in counts:
                    counts[attr] += 1

        # p_i = count_i / total_rules (keep your original definition)
        ps = [counts[attr] / total_rules for attr in attribute_universe if counts[attr] > 0]

        if not ps:
            scores[e] = 0.0
            continue

        # Entropy numerator: H = -Σ p ln p  (zeros contribute 0, already excluded)
        H = -sum(p * np.log(max(p, epsilon)) for p in ps)

        # Average rule length L = Σ p_i  (you can use sum(ps); zeros don't change the sum)
        L = sum(ps)

        # H_max = L * ln(N / L); guard edge cases
        if L <= 0:
            scores[e] = 0.0
            continue

        ratio = N / max(L, epsilon)            # avoid divide-by-zero
        H_max = L * np.log(max(ratio, 1.0))    # ln(1)=0 when L>=N; keeps H_max >= 0

        if H_max <= 0:
            # Happens when L == N (every rule uses all attributes): entropy is 0 anyway
            scores[e] = 0.0
            continue

        score = H / H_max

        # Numerical safety clamp (optional but harmless)
        scores[e] = float(min(max(score, 0.0), 1.0))

    return scores

def _attribute_usage_score_for_instance_old(df_rules_inst, attribute_universe, epsilon=1e-12):
    N = max(len(attribute_universe), 1)
    scores = {}
    for e, grp in df_rules_inst.groupby("Explainer"):
        total_rules = len(grp)
        if total_rules == 0:
            scores[e] = 0.0
            continue

        counts = {attr: 0 for attr in attribute_universe}
        for _, row in grp.iterrows():
            attrs_in_rule = {p['attr'] for p in row['Premises']}
            for attr in attrs_in_rule:
                if attr in counts:
                    counts[attr] += 1

        ps = [counts[attr] / total_rules for attr in attribute_universe if counts[attr] > 0]
        if not ps:
            scores[e] = 0.0
            continue

        num = -sum(p * np.log(max(p, epsilon)) for p in ps)  # -Σ p ln p
        den = np.log(max(N, 2))
        scores[e] = float(num / den) if den > 0 else 0.0
    return scores


def _avg_attr_usage_over_instances(attr_usage_per_instance):
    """
    attr_usage_per_instance: list of dicts {explainer -> score} (one per instance).
    Returns DataFrame with average per explainer.
    """
    if not attr_usage_per_instance:
        return pd.DataFrame(columns=["Explainer", "Attr_Usage_Score"])
    # union of explainers
    all_expl = sorted({e for d in attr_usage_per_instance for e in d.keys()})
    rows = []
    for e in all_expl:
        vals = [d.get(e, 0.0) for d in attr_usage_per_instance]
        rows.append({"Explainer": e, "Attr_Usage_Score": float(np.mean(vals)) if vals else 0.0})
    return pd.DataFrame(rows).sort_values("Explainer").reset_index(drop=True)


from itertools import combinations

def _jaccard_for_instance(df_rules_inst, mode="attributes"):
    """
    Build explainer rule-sets for one instance and compute pairwise Jaccard.
    mode:
      - 'attributes' -> each rule becomes a frozenset of attribute NAMES only
      - 'premises'   -> each rule becomes a frozenset of (attr, op, val_norm) tuples
      - 'rule_string'-> uses the 'Rule' column (human-readable string)
    Returns: dict {(A,B): jaccard}
    """
    def rule_to_token(rule_row):
        if mode == "rule_string":
            return str(rule_row["Rule"])
        elif mode == "premises":
            toks = []
            for p in rule_row["Premises"]:
                # normalize value to string; if numeric-like, keep the already rounded form (your loader rounds)
                v = p["val"]
                toks.append((p["attr"], p["op"], str(v)))
            return frozenset(sorted(toks))
        else:  # 'attributes' (default)
            attrs = {p["attr"] for p in rule_row["Premises"]}
            return frozenset(sorted(attrs))

    by_expl = {}
    for e, grp in df_rules_inst.groupby("Explainer"):
        S = {rule_to_token(r) for _, r in grp.iterrows()}
        by_expl[e] = S

    pairs = {}
    expl_list = sorted(by_expl.keys())
    for a, b in combinations(expl_list, 2):
        A, B = by_expl[a], by_expl[b]
        if not A and not B:
            j = 0  # both empty -> define as 1.0 (can change to 0.0 if you prefer)
        else:
            union = len(A | B)
            inter = len(A & B)
            j = inter / union if union > 0 else 0.0
        pairs[(a, b)] = j
    return pairs



def _avg_jaccard_over_instances(jacc_list):
    """
    jacc_list: list of dicts {(A,B): jaccard} per instance.
    Returns DataFrame with average Jaccard for each pair (A,B), and
    also a square matrix DataFrame (Explainer x Explainer) for convenience.
    """
    from collections import defaultdict
    agg = defaultdict(list)
    expl_set = set()
    for d in jacc_list:
        for (a, b), v in d.items():
            agg[(a, b)].append(v)
            expl_set.update([a, b])
    rows = []
    for (a, b), vals in agg.items():
        rows.append({"Pair": f"{a} - {b}", "Avg_Jaccard": float(np.mean(vals)) if vals else 0.0})
    pair_df = pd.DataFrame(rows).sort_values("Pair").reset_index(drop=True)

    # square matrix
    expl = sorted(expl_set)
    mat = pd.DataFrame(0.0, index=expl, columns=expl)
    for (a, b), vals in agg.items():
        mat.loc[a, b] = float(np.mean(vals))
        mat.loc[b, a] = mat.loc[a, b]
    np.fill_diagonal(mat.values, 1.0)
    mat.index.name = "Explainer"
    return pair_df, mat

def _attribute_universe(df_instances):
    # take feature keys from the first entry; all entries share the same schema
    exclude = ['Instance_Name', 'Original_Outcome', 'Predicted_Outcome']
    return [c for c in df_instances.columns if c not in exclude]

def _filtered_averages_from_step_table(step_avg_df: pd.DataFrame) -> pd.DataFrame:
    """
    step_avg_df: rows = explainers, columns = ['Original','Correct_Prediction','Threshold','NonDominated','Unique']
    Returns a new DataFrame with average FILTERED counts per step and a Total column:
      - Filtered_at_Correct_Prediction = Original - Correct_Prediction
      - Filtered_at_Threshold          = Correct_Prediction - Threshold
      - Filtered_at_NonDominated       = Threshold - NonDominated
      - Filtered_at_Unique             = NonDominated - Unique
      - Total_Filtered                 = row-wise sum of the four above
    """
    required = ["Original", "Correct_Prediction", "Threshold", "NonDominated", "Unique"]
    missing = [c for c in required if c not in step_avg_df.columns]
    if missing:
        raise ValueError(f"Missing required step columns: {missing}")

    df = step_avg_df.copy()
    out = pd.DataFrame(index=df.index)
    out.index.name = "Explainer"

    out["Filtered_at_Correct_Prediction"] = df["Original"] - df["Correct_Prediction"]
    out["Filtered_at_Threshold"]          = df["Correct_Prediction"] - df["Threshold"]
    out["Filtered_at_NonDominated"]       = df["Threshold"] - df["NonDominated"]
    out["Filtered_at_Unique"]             = df["NonDominated"] - df["Unique"]

    # Row sum as the last column
    out["Total_Filtered"] = (
        out["Filtered_at_Correct_Prediction"]
        + out["Filtered_at_Threshold"]
        + out["Filtered_at_NonDominated"]
        + out["Filtered_at_Unique"]
    )

    # Keep a tidy order
    return out[
        [
            "Filtered_at_Correct_Prediction",
            "Filtered_at_Threshold",
            "Filtered_at_NonDominated",
            "Filtered_at_Unique",
            "Total_Filtered",
        ]
    ]

def combine_with_shared(rules_step_cat1: pd.DataFrame, rules_step_cat2: pd.DataFrame,
                        want_filtered: bool) -> pd.DataFrame:
    expl = sorted(set(rules_step_cat1.index) | set(rules_step_cat2.index))
    c1, c2 = rules_step_cat1.reindex(expl).fillna(0.0), rules_step_cat2.reindex(expl).fillna(0.0)
    if want_filtered:
        cp  = c1["Original"] - c1["Correct_Prediction"]
        thr = c1["Correct_Prediction"] - c1["Threshold"]
        cat1_nd = c1["Threshold"] - c1["NonDominated"];  cat1_u = c1["NonDominated"] - c1["Unique"]
        cat2_nd = c2["Threshold"] - c2["NonDominated"];  cat2_u = c2["NonDominated"] - c2["Unique"]
        out = pd.DataFrame({
            "Filtered_at_Correct_Prediction": cp,
            "Filtered_at_Threshold":          thr,
            "Cat1_Filtered_at_NonDominated":  cat1_nd,
            "Cat1_Filtered_at_Unique":        cat1_u,
            "Cat1_Total_Filtered":            cp + thr + cat1_nd + cat1_u,
            "Cat2_Filtered_at_NonDominated":  cat2_nd,
            "Cat2_Filtered_at_Unique":        cat2_u,
            "Cat2_Total_Filtered":            cp + thr + cat2_nd + cat2_u,
        }, index=expl)
        cols = ["Filtered_at_Correct_Prediction","Filtered_at_Threshold",
                "Cat1_Filtered_at_NonDominated","Cat1_Filtered_at_Unique","Cat1_Total_Filtered",
                "Cat2_Filtered_at_NonDominated","Cat2_Filtered_at_Unique","Cat2_Total_Filtered"]
        out = out[cols]
    else:
        out = pd.DataFrame({
            "Original":           c1["Original"],
            "Correct_Prediction": c1["Correct_Prediction"],
            "Threshold":          c1["Threshold"],
            "Cat1_NonDominated":  c1["NonDominated"],
            "Cat1_Unique":        c1["Unique"],
            "Cat2_NonDominated":  c2["NonDominated"],
            "Cat2_Unique":        c2["Unique"],
        }, index=expl)
    out.index.name = "Explainer"
    # add “Sum” row
    out = pd.concat([out, out.sum(axis=0).to_frame().T.rename(index={0:"Sum"})])
    return out

def _init_step_buckets(explainers):
    return ({s: {e: [] for e in explainers} for s in STEP_NAMES},
            {s: {e: [] for e in explainers} for s in STEP_NAMES})

def _attr_usage_table(df_rules_subset: pd.DataFrame,
                      all_rules_df: pd.DataFrame,
                      use_avg: bool = True) -> pd.DataFrame:
    """
    Build Feature–Explainer usage table via parser_util.build_attr_usage_df(df),
    then (optionally) convert to AvgCountPerInstance by dividing by the
    number of instances that had at least one rule for that explainer.
    Returns a DataFrame with columns: Feature, Explainer, Count (plottable).
    """
    if df_rules_subset is None or df_rules_subset.empty:
        return pd.DataFrame(columns=["Feature", "Explainer", "Count"])
    # raw counts per Feature×Explainer
    usage = parser_util.build_attr_usage_df(df_rules_subset)
    if not use_avg:
        return usage.rename(columns={"Count": "Count"}).copy()
    # average per instance for that explainer
    inst_per_expl = (all_rules_df.groupby("Explainer")["Instance_Name"]
                     .nunique().rename("n_inst_with_rules"))
    usage = usage.merge(inst_per_expl, on="Explainer", how="left")
    usage["Count"] = usage["Count"] / usage["n_inst_with_rules"].clip(lower=1)
    return usage[["Feature", "Explainer", "Count"]]

def _jaccard_for_instance_attribute_union(df_rules_inst: pd.DataFrame, both_empty="one"):
    """
    Jaccard similarity per explainer pair for ONE instance, computed on
    the union of attributes used by each explainer across ALL its rules.
    
    both_empty: what to return when both explainers used no attributes
        - "zero" -> 0.0 (current default in your code)
        - "one"  -> 1.0
        - "nan"  -> np.nan  (and then use nanmean when averaging)
    """
    # Build one attribute set per explainer (union across its rules)
    by_expl = {}
    for expl, grp in df_rules_inst.groupby("Explainer"):
        attrs = set()
        # grp["Premises"] is a list of dicts: [{"attr": ..., "op": ..., "val": ...}, ...]
        for premises in grp["Premises"]:
            if isinstance(premises, (list, tuple)):
                for p in premises:
                    a = p.get("attr")
                    if a is not None:
                        attrs.add(a)
        by_expl[expl] = attrs

    # Compute Jaccard for all pairs
    pairs = {}
    expl_list = sorted(by_expl.keys())
    for a, b in combinations(expl_list, 2):
        A, B = by_expl[a], by_expl[b]
        if not A and not B:
            j = 0.0 if both_empty == "zero" else (1.0 if both_empty == "one" else np.nan)
        else:
            u = len(A | B)
            i = len(A & B)
            j = (i / u) if u > 0 else (np.nan if both_empty == "nan" else 0.0)
        pairs[(a, b)] = j
    return pairs

def analyze_results(
    rules_log: str,
    entries_log: str,
    path: str = "./experiments_log/",
    min_cov: float = 0.1,
    min_cov_class: float = 0.1,
    min_pre: float = 0.1,
    random_seed: int = 42,
    use_avg_attr_per_instance = True
):
    # Load
    rules_grouped = parser_util.load_and_group_rules(path + rules_log)
    df_instances = parser_util.load_entries_to_df(path + entries_log)
    df_rules = parser_util.grouped_rules_to_df(rules_grouped)

    instance_names = df_rules['Instance_Name'].unique()
    explainers = sorted(df_rules['Explainer'].unique())
    # buckets to track per-explainer counts per filtration step for each category
    cat1_step_buckets, cat2_step_buckets = _init_step_buckets(explainers)

    # overall filtered-at-step trackers (counts removed at each step, per instance)
    filtered_counts_overall = {
        "Correct Prediction": [],
        "Threshold Filter": [],
        "Non-dominated 1": [],
        "Unique Non-dominated 1": [],
        "Non-dominated 2": [],
        "Unique Non-dominated 2": [],
    }
    
    rng = np.random.default_rng(random_seed)

    # Accumulators for both categories (final unique sets)
    cat1_frames, cat2_frames = [], []
    attr_usage_instance_scores = []   # list of dicts {explainer -> score} per instance
    jaccard_instance_list = []        # list of dicts {(A,B): jaccard} per instance
    attribute_univ = _attribute_universe(df_instances)
    
    # Keep your ideal-found printouts
    cat1_ideal_instances = defaultdict(list)
    cat2_ideal_instances = defaultdict(list)

    # Per-instance counts for Avg_Rules_Per_Instance
    cat1_rules_per_instance = defaultdict(list)
    cat2_rules_per_instance = defaultdict(list)

    # NEW: chosen rule holders per method per category
    methods = CHOICE_METHODS
    chosen_cat1 = {m: [] for m in methods}  # list of selected rows (as DataFrames) across instances
    chosen_cat2 = {m: [] for m in methods}
    
    for instance_name in instance_names:
        # ---------- Original rules & attribute usage / Jaccard ----------
        original_rules = df_rules[df_rules["Instance_Name"] == instance_name]
        counts_orig = counts_by_explainer(original_rules, explainers)
    
        # Attribute usage (global universe) + Jaccard (attributes mode)
        attr_usage_instance_scores.append(
            _attribute_usage_score_for_instance(original_rules, attribute_univ)
        )
        jaccard_instance_list.append(
            _jaccard_for_instance_attribute_union(original_rules)
        )
    
        # ---------- Correct prediction subset ----------
        predicted_outcome = df_instances.loc[
            df_instances["Instance_Name"] == instance_name, "Predicted_Outcome"
        ].iloc[0]
    
        correct_pred_rules = df_rules[
            (df_rules["Instance_Name"] == instance_name)
            & (df_rules["Rule"].str.contains(f"class = {predicted_outcome}", na=False))
        ]
        counts_cp = counts_by_explainer(correct_pred_rules, explainers)
    
        # ---------- Thresholded once ----------
        t_rules = correct_pred_rules[
            (correct_pred_rules["Cov"] >= min_cov)
            & (correct_pred_rules["Cov_class"] >= min_cov_class)
            & (correct_pred_rules["Pre"] >= min_pre)
        ].copy()
        counts_thr = counts_by_explainer(t_rules, explainers)
    
        # ---------- Category 1 artifacts ----------
        non_dom1, unique1, _, ideal1_present, ideal1_explainer = non_dom_and_unique(t_rules, category=1)
        counts_nd1 = counts_by_explainer(non_dom1, explainers)
        counts_u1  = counts_by_explainer(unique1, explainers)
    
        # update per-instance counts for Avg_Rules_Per_Instance
        for e in explainers:
            cat1_rules_per_instance[e].append(int(counts_u1[e]))
    
        if unique1 is not None and not unique1.empty:
            cat1_frames.append(unique1)
        if non_dom1 is not None and not non_dom1.empty:
            for m in methods:
                row = _select_rule_by_method(non_dom1, category=1, method=m, rng=rng)
                if row is not None:
                    chosen_cat1[m].append(row.to_frame().T)
        if ideal1_present and ideal1_explainer is not None:
            cat1_ideal_instances[ideal1_explainer].append(instance_name)
    
        # ---------- Category 2 artifacts ----------
        non_dom2, unique2, _, ideal2_present, ideal2_explainer = non_dom_and_unique(t_rules, category=2)
        counts_nd2 = counts_by_explainer(non_dom2, explainers)
        counts_u2  = counts_by_explainer(unique2, explainers)
    
        for e in explainers:
            cat2_rules_per_instance[e].append(int(counts_u2[e]))
    
        if unique2 is not None and not unique2.empty:
            cat2_frames.append(unique2)
        if non_dom2 is not None and not non_dom2.empty:
            for m in methods:
                row = _select_rule_by_method(non_dom2, category=2, method=m, rng=rng)
                if row is not None:
                    chosen_cat2[m].append(row.to_frame().T)
        if ideal2_present and ideal2_explainer is not None:
            cat2_ideal_instances[ideal2_explainer].append(instance_name)
    
        # ---------- Accumulate step buckets (Cat1 & Cat2 share first 3 steps) ----------
        accum_counts(cat1_step_buckets, "Original",           counts_orig)
        accum_counts(cat1_step_buckets, "Correct_Prediction", counts_cp)
        accum_counts(cat1_step_buckets, "Threshold",          counts_thr)
        accum_counts(cat1_step_buckets, "NonDominated",       counts_nd1)
        accum_counts(cat1_step_buckets, "Unique",             counts_u1)
    
        accum_counts(cat2_step_buckets, "Original",           counts_orig)
        accum_counts(cat2_step_buckets, "Correct_Prediction", counts_cp)
        accum_counts(cat2_step_buckets, "Threshold",          counts_thr)
        accum_counts(cat2_step_buckets, "NonDominated",       counts_nd2)
        accum_counts(cat2_step_buckets, "Unique",             counts_u2)
    
        # ---------- Overall filtered (instance-level) ----------
        all_rules_count       = int(len(original_rules))
        correct_pred_count    = int(len(correct_pred_rules))
        thresholded_count     = int(len(t_rules))
        non_dom1_count        = int(len(non_dom1)) if non_dom1 is not None else 0
        unique_non_dom1_count = int(len(unique1))  if unique1  is not None else 0
        non_dom2_count        = int(len(non_dom2)) if non_dom2 is not None else 0
        unique_non_dom2_count = int(len(unique2))  if unique2  is not None else 0
    
        filtered_counts_overall["Correct Prediction"].append(all_rules_count - correct_pred_count)
        filtered_counts_overall["Threshold Filter"].append(correct_pred_count - thresholded_count)
        filtered_counts_overall["Non-dominated 1"].append(thresholded_count - non_dom1_count)
        filtered_counts_overall["Unique Non-dominated 1"].append(non_dom1_count - unique_non_dom1_count)
        filtered_counts_overall["Non-dominated 2"].append(thresholded_count - non_dom2_count)
        filtered_counts_overall["Unique Non-dominated 2"].append(non_dom2_count - unique_non_dom2_count)
        
    # Build per-explainer summaries over the entire unique sets (as before)
    dom1_df_all = pd.concat(cat1_frames, ignore_index=True) if cat1_frames else pd.DataFrame()
    dom2_df_all = pd.concat(cat2_frames, ignore_index=True) if cat2_frames else pd.DataFrame()

    # Original (pre-filtration): use the full df_rules
    attr_usage_original_tbl = _attr_usage_table(
        df_rules, all_rules_df=df_rules, use_avg=use_avg_attr_per_instance
    )
    # Cat1 Unique
    attr_usage_cat1_tbl = _attr_usage_table(
        dom1_df_all, all_rules_df=df_rules, use_avg=use_avg_attr_per_instance
    )
    # Cat2 Unique
    attr_usage_cat2_tbl = _attr_usage_table(
        dom2_df_all, all_rules_df=df_rules, use_avg=use_avg_attr_per_instance
    )
    
    summary_dom1 = _summarize_per_explainer(dom1_df_all, cat1_rules_per_instance)
    summary_dom2 = _summarize_per_explainer(dom2_df_all, cat2_rules_per_instance)
    
    # Pairwise diffs (ordered pairs)
    diff_dom1 = _pairwise_diffs(summary_dom1) if not summary_dom1.empty else pd.DataFrame()
    diff_dom2 = _pairwise_diffs(summary_dom2) if not summary_dom2.empty else pd.DataFrame()
    
    # Winner matrices (counts and ratios) per category
    winner_matrix_cat1_counts = _winner_matrix(chosen_cat1, explainers, normalize=False)
    winner_matrix_cat1_ratio  = _winner_matrix(chosen_cat1, explainers, normalize=True)
        
    winner_matrix_cat2_counts = _winner_matrix(chosen_cat2, explainers, normalize=False)
    winner_matrix_cat2_ratio  = _winner_matrix(chosen_cat2, explainers, normalize=True)
    
    
    # ---- NEW: averaged metrics per explainer for each choice method (per category) ----
    def _avg_metrics_over_chosen(chosen_rows_list):
        if not chosen_rows_list:
            return pd.DataFrame(columns=["Explainer", "Cov", "Cov_class", "Pre", "Len", "Reject", "Elapsed_time", "Distance_idp_eucl"])
        df = pd.concat(chosen_rows_list, ignore_index=True)
        cols = ["Cov", "Cov_class", "Pre", "Len", "Reject", "Elapsed_time", "Distance_idp_eucl"]
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        out = df.groupby("Explainer")[cols].mean().reset_index().sort_values("Explainer")
        return out
    
    choice_tables_cat1 = {m: _avg_metrics_over_chosen(chosen_cat1[m]) for m in methods}
    choice_tables_cat2 = {m: _avg_metrics_over_chosen(chosen_cat2[m]) for m in methods}
    
        # # Print instance lists where ideal rule found (kept)
        # print("\nInstances with ideal rule found (Category 1: Cov↑, Pre↑):")
        # for expl, instances in cat1_ideal_instances.items():
        #     print(f"{expl} : {', '.join(map(str, instances))}")
    
        # print("\nInstances with ideal rule found (Category 2: Cov_class↑, Pre↑, Len↓):")
        # for expl, instances in cat2_ideal_instances.items():
        #     print(f"{expl} : {', '.join(map(str, instances))}")
    
         # NEW: average number of rules per step (rows = explainers, columns = steps)
    rules_per_step_cat1 = _averages_table(cat1_step_buckets)  # Original, Correct_Prediction, Threshold, NonDominated, Unique
    rules_per_step_cat2 = _averages_table(cat2_step_buckets)
    
    rules_per_step_combined   = combine_with_shared(rules_per_step_cat1, rules_per_step_cat2, want_filtered=False)
        
    filtered_per_step_combined = combine_with_shared(rules_per_step_cat1, rules_per_step_cat2, want_filtered=True)
    
        
    # NEW: average number of filtered rules per step (overall)
    avg_filtered_per_step = pd.DataFrame({
        "Step": list(filtered_counts_overall.keys()),
        "Avg_Filtered": [pd.Series(v).mean() if len(v) > 0 else 0.0 for v in filtered_counts_overall.values()]
    })
    
    # average attribute-usage per explainer over instances
    attr_usage_avg = _avg_attr_usage_over_instances(attr_usage_instance_scores)
    
    # average Jaccard over instances (pair list + square matrix)
    jaccard_pairs_avg, jaccard_matrix_avg = _avg_jaccard_over_instances(jaccard_instance_list)
    
    return {
        "attribute_universe": attribute_univ,
        "summary_dom1": summary_dom1,
        "diff_dom1": diff_dom1,
        "summary_dom2": summary_dom2,
        "diff_dom2": diff_dom2,
        "choice_tables_cat1": choice_tables_cat1,
        "choice_tables_cat2": choice_tables_cat2,
        "winner_matrix_cat1_counts": winner_matrix_cat1_counts,
        "winner_matrix_cat1_ratio": winner_matrix_cat1_ratio,
        "winner_matrix_cat2_counts": winner_matrix_cat2_counts,
        "winner_matrix_cat2_ratio": winner_matrix_cat2_ratio,
        "rules_per_step_combined": rules_per_step_combined,
        "filtered_per_step_combined": filtered_per_step_combined,
        "avg_filtered_per_step": avg_filtered_per_step,
        "attr_usage_avg": attr_usage_avg,                 # Explainer, Attr_Usage_Score (0..1)
        "jaccard_pairs_avg": jaccard_pairs_avg,           # rows: "A - B", Avg_Jaccard
        "jaccard_matrix_avg": jaccard_matrix_avg,         # square matrix DataFrame
        "attr_usage_original_tbl": attr_usage_original_tbl,
        "attr_usage_cat1_tbl": attr_usage_cat1_tbl, 
        "attr_usage_cat2_tbl": attr_usage_cat2_tbl, 
    }

