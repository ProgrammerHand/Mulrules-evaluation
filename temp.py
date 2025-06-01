# import dataset
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
# import pandas as pd
# import numpy as np
# from lux.lux import LUX
# from alibi.utils import gen_category_map
# from alibi.explainers import AnchorTabular
#
#
# class SimpleNN(nn.Module):
#     def __init__(self,input_size):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, 128)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 1)
#         self.sigmoid = nn.Sigmoid()
#
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         x = self.sigmoid(x)
#         return x
#
#     def fit(self,X,y):
#       # Convert data to tensors
#       X_train_tensor = torch.tensor(X.todense(), dtype=torch.float32)
#       y_train_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
#
#       # Loss and optimizer
#       criterion = nn.BCELoss()
#       optimizer = optim.Adam(self.parameters(), lr=0.001)
#
#       # Training loop
#       for epoch in range(100):
#           optimizer.zero_grad()
#           outputs = self(X_train_tensor)
#           loss = criterion(outputs, y_train_tensor)
#           loss.backward()
#           optimizer.step()
#
#           if (epoch + 1) % 10 == 0:
#               print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')
#
#     def predict_proba(self, X):
#         # Ensure input is dense (convert if sparse)
#         if hasattr(X, "toarray"):  # This checks if the input is a sparse matrix (like from OneHotEncoder)
#             X = X.toarray()  # Convert sparse matrix to dense
#
#         # Convert to tensor if necessary
#         X_tensor = torch.tensor(X, dtype=torch.float32)
#
#         # Perform the forward pass to get predictions
#         with torch.no_grad():
#             outputs = self(X_tensor)
#
#         # Convert to probabilities (binary classification)
#         probabilities = outputs.numpy()  # Convert to numpy array
#         return np.column_stack([1 - probabilities, probabilities])  # For binary classification
#
#     def predict(self, X):
#         # Ensure input is dense (convert if sparse)
#         if hasattr(X, "toarray"):  # This checks if the input is a sparse matrix (like from OneHotEncoder)
#             X = X.toarray()  # Convert sparse matrix to dense
#
#         # Convert to tensor if necessary
#         X_tensor = torch.tensor(X, dtype=torch.float32)
#
#         # Perform the forward pass to get predictions
#         with torch.no_grad():
#             outputs = self(X_tensor)
#
#         # Classify based on the output probability (threshold of 0.5)
#         predictions = (outputs >= 0.5).float()  # Binary classification: 0 or 1
#         return predictions.numpy().ravel()   # Convert to numpy array
#
# class CategoricalWrapper:
#     def __init__(self, model_creator,  model_params=None, ohe_encoder=None, categorical_indicator=None, features=None, categories='auto', normalize=False):
#         from sklearn.compose import ColumnTransformer
#         from sklearn.preprocessing import OneHotEncoder,StandardScaler
#
#         # OneHotEncoder for categorical columns
#         if ohe_encoder is None:
#             self.ohe_encoder = OneHotEncoder(categories=categories)
#         else:
#             self.ohe_encoder = ohe_encoder
#
#         # Store parameters
#         self.features = features
#         self.categories = categories
#         self.categorical_indicator = categorical_indicator
#         self.model_params = model_params
#         self.model_creator = model_creator
#
#         # Add StandardScaler for non-categorical features if normalize=True
#         transformers = [
#             ("categorical", self.ohe_encoder, [f for f, c in zip(features, categorical_indicator) if c])
#         ]
#
#         # If normalize is True, add StandardScaler for non-categorical columns
#         if normalize:
#             non_categorical_columns = [f for f, c in zip(features, categorical_indicator) if not c]
#             transformers.append(("scaler", StandardScaler(), non_categorical_columns))
#
#         # Create the ColumnTransformer
#         self.ct = ColumnTransformer(
#             transformers,
#             remainder='passthrough'
#         )
#
#         self.model_params = model_params
#         self.model_creator = model_creator
#
#
#     def fit(self, X, y):
#         X_tr = self.ct.fit_transform(X)
#
#         if self.model_params is None:
#             model_params = {}
#         elif self.model_params=='input_size':
#             model_params = {'input_size':X_tr.shape[1]}
#
#         # Create the model by passing parameters to the model_creator lambda
#         self.model = self.model_creator(**model_params)
#
#         self.model.fit(X_tr, y)
#         return self
#
#
#     def predict(self, X):
#         if type(X) is np.ndarray and self.features is not None:
#             X = pd.DataFrame(X, columns=self.features)
#         return self.model.predict(self.ct.transform(X))
#
#     def predict_proba(self, X):
#         if type(X) is np.ndarray and self.features is not None:
#             X = pd.DataFrame(X, columns=self.features)
#
#         X = self.ct.transform(X)
#         if hasattr(self.model, 'predict_proba'):
#             return self.model.predict_proba(X)
#         elif hasattr(self.model, 'decision_function'):
#             # Sigmoid transformation for decision_function output
#             decision_scores = self.model.decision_function(X)
#             probabilities = 1 / (1 + np.exp(-decision_scores))
#             return np.column_stack([1 - probabilities, probabilities])
#         else:
#             return np.array([self.model.predict(X)==c for c in self.model.classes_]).T
#
#     def score(self,X,y):
#         return self.model.score(self.ct.transform(X),y)
#
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
#
# column_names = [
#     "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
#     "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
#     "hours_per_week", "native_country", "income"
# ]
#
# df = pd.read_csv(url, header=None, names=column_names, na_values=" ?", skipinitialspace=True)
# # Display basic information about the dataset
# print("Dataset Shape:", df.shape)
# print("\nSample Data:")
# print(df.head())
#
# # Check for missing values
# print("\nMissing Values:")
# print(df.isnull().sum())
#
# # Preprocess the dataset (e.g., encoding categorical variables, handling missing values)
# df = df.dropna()
# categorical_columns = df.select_dtypes(include=['object']).columns
# categorical_columns = categorical_columns[categorical_columns != 'income']
# features = df.columns[:-1]
# categorical_indicator = [col in categorical_columns for col in features]
#
# le = LabelEncoder()
# for col in categorical_columns:
#     df[col] = le.fit_transform(df[col])
# df['income'] = le.fit_transform(df['income'])
#
# # Split the data into features and target
# target = 'income'
# X = df.drop(columns=[target])
# y = df[target]
#
# # Split into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model_type = 'deep_learning'
#
# model_creators = {
#     'random_forest': lambda: RandomForestClassifier(),
#     'svm': lambda: SVC(probability=True),
#     'logistic_regression': lambda: LogisticRegression(),
#     'mlp': lambda: MLPClassifier(),
#     'deep_learning': lambda input_size: SimpleNN(input_size=input_size)  # Lambda with parameter
# }
#
# if model_type == 'xgb':
#     # Use XGBoost with categorical support enabled
#     model = xgb.XGBClassifier(enable_categorical=True)
#     model.fit(X_train, y_train)
# elif model_type == 'random_forest':
#     model = CategoricalWrapper(model_creators[model_type], categorical_indicator=categorical_indicator, features=features)
#     model.fit(X_train, y_train)
# elif model_type == 'svm':
#     model = CategoricalWrapper(model_creators[model_type], categorical_indicator=categorical_indicator, features=features)
#     model.fit(X_train, y_train)
# elif model_type == 'logistic_regression':
#     model = CategoricalWrapper(model_creators[model_type], categorical_indicator=categorical_indicator, features=features)
#     model.fit(X_train, y_train)
# elif model_type == 'mlp':
#     model = CategoricalWrapper(model_creators[model_type], categorical_indicator=categorical_indicator, features=features)
#     model.fit(X_train, y_train)
# elif model_type == 'deep_learning':
#     # Define a simple neural network with PyTorch
#
#     # Wrap the trained model in CategoricalWrapper
#     model = CategoricalWrapper(
#         model_creator=model_creators[model_type],
#         model_params='input_size',
#         features=X_train.columns,
#         normalize=True,
#         categorical_indicator=categorical_indicator
#     )
#     model.fit(X_train, y_train)
# else:
#     print("Invalid model type selected.")
#
# explain_instance = X_train.sample(1).values
# categorical_columns = X_train.select_dtypes(include=['object']).columns
# features = df.columns[:-1]
# categorical_indicator = [col in categorical_columns for col in features]
# lux = LUX(predict_proba = model.predict_proba,
#     neighborhood_size=50,max_depth=2,
#     node_size_limit = 1,
#     grow_confidence_threshold = 0 )
# lux.fit(X_train, y_train,
# instance_to_explain=explain_instance,
# # categorical indicator
# categorical=categorical_indicator)
# print(lux.justify(explain_instance))
#
# category_map = gen_category_map(X)
# explainer = AnchorTabular(model.predict, features, categorical_names=category_map, seed=1)
# explainer.fit(X_train)
#
# explanation = explainer.explain(explain_instance)
# print('Anchor: IF %s' % (' AND '.join(explanation.anchor) + f' THEN {explainer.predictor(explain_instance.reshape(1, -1))[0]}'))

from lore_sa.dataset import TabularDataset
from lore_sa.neighgen import GeneticGenerator
from lore_sa.encoder_decoder import ColumnTransformerEnc
from lore_sa.lore import Lore
from lore_sa.surrogate import DecisionTreeSurrogate
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from lore_sa.bbox import sklearn_classifier_bbox

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from xailib.data_loaders.dataframe_loader import prepare_dataframe

from xailib.explainers.lime_explainer import LimeXAITabularExplainer
from xailib.explainers.lore_explainer import LoreTabularExplainer
from xailib.explainers.shap_explainer_tab import ShapXAITabularExplainer

from xailib.models.sklearn_classifier_wrapper import sklearn_classifier_wrapper

# source_file = 'data/german.csv'
# class_field = 'class'
# # Load and transform dataset
# df = pd.read_csv(source_file, skipinitialspace=True, na_values='?', keep_default_na=True)
# df, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map = prepare_dataframe(df, class_field)
# test_size = 0.3
# random_state = 42
# X_train, X_test, Y_train, Y_test = train_test_split(df[feature_names], df[class_field],
#                                                         test_size=test_size,
#                                                         random_state=random_state,
#                                                         stratify=df[class_field])
#
# bb = RandomForestClassifier(n_estimators=20, random_state=random_state)
# bb.fit(X_train.values, Y_train.values)
# bbox = sklearn_classifier_wrapper(bb)
#
# explainer = LoreTabularExplainer(bbox)
# config = {'neigh_type':'rndgen', 'size':1000, 'ocr':0.1, 'ngen':10}
# df = pd.read_csv(source_file, skipinitialspace=True, na_values='?', keep_default_na=True)
# explainer.fit(df, class_field, config)
# inst = X_train.iloc[147].values
# exp = explainer.explain(inst)
# print(exp.exp.rule)


# df = pd.read_csv('data/adult.csv')
#
# num_cols = [0,2,4,10,11,12]  # Indices of numerical columns
# cat_cols = [1,3,5,6,7,8,9,13]  # Indices of categorical columns
#
# # Extract unique categories from the entire dataset
# unique_categories = [df.iloc[:, col].astype(str).unique().tolist() for col in cat_cols]
# numeric_col_names = df.columns[num_cols]
# categorical_col_names = df.columns[cat_cols]
#
# preprocessor = ColumnTransformer(
#     transformers=[
#         # ('num', StandardScaler(), num_cols),
#         ('cat', OrdinalEncoder(categories=unique_categories, handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
#         # ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
#     ]
# )
# model = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=100, random_state=42))
# # model = RandomForestClassifier(n_estimators=100, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(df.loc[:, 'age':'native.country'].values, df['class'].values,
#                                             test_size=0.3, random_state=42, stratify=df['class'].values)
# model.fit(X_train, y_train)
#
# bbox = sklearn_classifier_bbox.sklearnBBox(model)
#
#
# dataset = TabularDataset.from_csv('data/adult.csv', numeric_col_names, categorical_col_names, class_name='class', dropna = False)
# # dataset.df.dropna(inplace=True)
# # dataset.df.drop(['fnlwgt', 'education.num'], axis=1, inplace=True)
# dataset.update_descriptor()
#
# enc = ColumnTransformerEnc(dataset.descriptor)
# generator = GeneticGenerator(bbox, dataset, enc)
# surrogate = DecisionTreeSurrogate()
#
# tabularLore = Lore(bbox, dataset, enc, generator, surrogate)
#
# instance = X_test[0]
# explanation = tabularLore.explain(instance)
# print("IF ", end='')
# conditions = [f"{part['attr']} {part['op']} {part['val']}" for part in explanation['rule']['premises']]
# print(" AND ".join(conditions), end=' ')
# print(f"THEN {explanation['rule']['consequence']['val']}")

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


def count_attributes_from_rules(df_rules):
    attr_counter = Counter()

    for premises in df_rules['Premises']:
        for cond in premises:
            attr = cond['attr']
            attr_counter[attr] += 1

    return dict(attr_counter)


def build_attr_usage_df(df_rules):
    rows = []
    for explainer, group in df_rules.groupby('Explainer'):
        attr_counts = count_attributes_from_rules(group)
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
    # pivot table with counts: rows=features, columns=explainers
    heatmap_data = df.pivot_table(index=feature_col, columns=explainer_col, values=count_col, fill_value=0)
    print(df)
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


from IPython.display import display, Markdown

pd.set_option('display.max_colwidth', None)

path = "./experiments_log/"
rules_log = "adult_random_forest_rules_11-22_31-05-2025.txt"
entries_log = "adult_random_forest_entries_11-22_31-05-2025.txt"
rules_grouped = load_and_group_rules(path + rules_log)
df_instances = load_entries_to_df(path + entries_log)
df_rules = grouped_rules_to_df(rules_grouped)
instance_names = df_rules['Instance_Name'].unique()

for i, instance_name in enumerate(instance_names, start=1):
    max_rules = df_rules[df_rules['Instance_Name'] == instance_name].groupby('Explainer').size().max()
    orig = df_instances.loc[df_instances['Instance_Name'] == instance_name, "Original_Outcome"].iloc[0]
    pred = df_instances.loc[df_instances['Instance_Name'] == instance_name, "Predicted_Outcome"].iloc[0]
    display(Markdown(f"## Instance {instance_name} (Original: {orig} , Predicted: {pred})"))
    instance = df_instances[df_instances['Instance_Name'] == instance_name]
    exclude_cols = ['Instance_Name', 'Original_Outcome', 'Predicted_Outcome']
    attributes = [col for col in instance.columns if col not in exclude_cols]
    display(instance.drop(columns=['Instance_Name', 'Original_Outcome', 'Predicted_Outcome']).T.rename(
        columns={df_instances[df_instances['Instance_Name'] == instance_name].index[0]: 'Value'}))

    display(Markdown(f"### Rules for Instance {instance_name}"))
    display(df_rules[df_rules['Instance_Name'] == instance_name].drop(columns=["Premises"]).reset_index(drop=True))

    display(Markdown(f"### Rules for Instance {instance_name}, Correct Prediction"))
    original_outcome = df_instances.loc[df_instances['Instance_Name'] == instance_name, 'Original_Outcome'].values[0]
    correct_pred_rules = df_rules[
        (df_rules['Instance_Name'] == instance_name) &
        (df_rules['Rule'].str.contains(f'class = {original_outcome}', na=False))
        ]
    display(correct_pred_rules.drop(columns=["Premises"]).reset_index(drop=True))

    min_cov = 0.01
    min_cov_class = 0.01
    min_pre = 0.01
    display(Markdown(
        f"### Rules for Instance {instance_name}, Min_treshold (Cov {min_cov}, Cov_class {min_cov_class}, Pre {min_pre})"))
    tresholded_rules = correct_pred_rules[
        (correct_pred_rules['Cov'] >= min_cov) &
        (correct_pred_rules['Cov_class'] >= min_cov_class) &
        (correct_pred_rules['Pre'] >= min_pre)
        ]
    display(tresholded_rules.drop(columns=["Premises"]).reset_index(drop=True))

    display(Markdown(f"### Rules for Instance {instance_name}, Non-dominated (Cov↑, Pre↑)"))
    non_dominated_rules1 = filter_non_dominated(tresholded_rules)
    display(non_dominated_rules1.drop(columns=["Premises"]).reset_index(drop=True))
    plot_non_dominated_rules(non_dominated_rules1, instance_name)
    plot_rules_comparison(all_rules=df_rules[df_rules['Instance_Name'] == instance_name],
                          filtered_rules=non_dominated_rules1,
                          instance_name=instance_name)
    agg_df = build_attr_usage_df(non_dominated_rules1)
    plot_feature_usage_heatmap(agg_df, feature_col="Feature", explainer_col="Explainer", count_col="Count",
                               all_features=attributes, vmax=max_rules)

    display(Markdown(f"### Rules for Instance {instance_name}, Non-dominated (Cov_class↑, Pre↑, Len↓)"))
    non_dominated_rules2 = filter_non_dominated_3d(tresholded_rules)
    display(non_dominated_rules2.drop(columns=["Premises"]).reset_index(drop=True))
    agg_df = build_attr_usage_df(non_dominated_rules2)
    plot_feature_usage_heatmap(agg_df, feature_col="Feature", explainer_col="Explainer", count_col="Count",
                               all_features=attributes, vmax=max_rules)
