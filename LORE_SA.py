import dataset_manager_legacy
import Anchor
import LORE
import LUX
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from xailib.models.sklearn_classifier_wrapper import sklearn_classifier_wrapper
from lore_sa.bbox import sklearn_classifier_bbox
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from alibi.utils import gen_category_map
import numpy as np
from lore_sa.bbox import sklearn_classifier_bbox
from lore_sa.dataset import TabularDataset
from lore_sa.neighgen import GeneticGenerator
from lore_sa.encoder_decoder import ColumnTransformerEnc
from lore_sa.lore import Lore
from lore_sa.surrogate import DecisionTreeSurrogate
from Rule_wrapper import rule_wrapper
from collections import Counter
import time
from sklearn.pipeline import make_pipeline

class lore_sa_object:

    def __init__(self, name, numeric_col_names, categorical_col_names, X_test, y_test, raw, train_df, target_name = 'class'):
        self.dataset = TabularDataset(data=raw, class_name="class", categorial_columns=categorical_col_names, numerical_columns=numeric_col_names)
        self.dataset.df = train_df
        # self.dataset.update_descriptor(categorical_col_names)
        self.X_test = X_test
        self.y_test = y_test
        self.train_df = train_df
        self.tabularLore = None
        self.inst = None

    def init_explainer(self, bbox, iter_limit=15):
        enc = ColumnTransformerEnc(self.dataset.descriptor)
        generator = GeneticGenerator(bbox, self.dataset, enc)
        surrogate = DecisionTreeSurrogate()
        self.tabularLoreExplainer = Lore(bbox, self.dataset, enc, generator, surrogate)
        self.iter_limit = iter_limit
        return f"Initializing LORE_SA Explainer with params: iter_limit = {self.iter_limit}"

    def get_instance(self, idx):
        if type(self.X_test) == np.ndarray:
            self.inst = self.X_test[idx]
        else:
            self.inst = self.X_test.to_numpy()[idx]

    def normalize_rule(self, rule):
        return {
            'premises': sorted((p['attr'], p['op'], str(p['val'])) for p in rule['premises']),
            'consequence': (rule['consequence']['attr'], rule['consequence']['op'], str(rule['consequence']['val']))
        }

    def explain(self, amount):
        explanations = []
        rejected_count = Counter({i: 0 for i in range(amount)})
        i = 0
        start_rule_time = time.time()
        # while len(explanations) < amount:
        for n in range(self.iter_limit):
            print(n)
            explanation = self.tabularLoreExplainer.explain(self.inst)
            if len(explanations) > 0:
                if amount - len(explanations) < self.iter_limit - n:
                    flag = False
                    for rule in explanations:
                        if rule.matches_raw_rule(explanation["rule"]["premises"], explanation["rule"]["consequence"]):
                            flag = True
                            rejected_count[i] += 1
                            break
                    if not flag:
                        elapsed = time.time() - start_rule_time
                        explanations.append(rule_wrapper.from_rule(explanation["rule"]["premises"], explanation["rule"]["consequence"], "LORE_SA", rejected_count[i], elapsed, False))
                        i += 1
                        start_rule_time = time.time()
                else:
                    elapsed = time.time() - start_rule_time
                    explanations.append(
                        rule_wrapper.from_rule(explanation["rule"]["premises"], explanation["rule"]["consequence"],
                                               "LORE_SA", rejected_count[i], elapsed, True))
                    i += 1
                    start_rule_time = time.time()

            else:
                elapsed = time.time() - start_rule_time
                explanations.append(rule_wrapper.from_rule(explanation["rule"]["premises"], explanation["rule"]["consequence"], "LORE_SA", rejected_count[i], elapsed, False))
                i += 1
                start_rule_time = time.time()

            if len(explanations) >= amount:
                break
            # if self.normalize_rule(explanation["rule"]) not in [self.normalize_rule(entry["rule"]) for entry in explanations]:#
            #     explanations.append(explanation)
        return explanations
        # return self.explainer.explain(self.inst)

    def print_explanation(self, explanation):
        conditions = " AND ".join(
            [f"{part['attr']} {part['op']} {part['val']}" for part in explanation['rule']['premises']])
        result_string = f"LORE_sa: IF {conditions} THEN {explanation['rule']['consequence']['val']} Pre, Cov : {self.calculate_precision_coverage(explanation)}"
        return result_string

    def apply_condition(self, condition):
        if condition["op"] == '=':
            return self.raw[condition["attr"]] == condition["val"]
        elif condition["op"] == '!=':
            return self.raw[condition["attr"]] != condition["val"]
        elif condition["op"] == '<=':
            return self.raw[condition["attr"]] <= condition["val"]
        elif condition["op"] == '<':
            return self.raw[condition["attr"]] < condition["val"]
        elif condition["op"] == '>=':
            return self.raw[condition["attr"]] >= condition["val"]
        elif condition["op"] == '>':
            return self.raw[condition["attr"]] > condition["val"]

    def calculate_precision_coverage(self, explanation):

        condition_mask = pd.Series([True] * len(self.raw))  # start with all True (all rows selected)

        for condition in explanation["rule"]["premises"]:
            condition_mask &= self.apply_condition(condition)

        # filter rows that satisfy the rule
        filtered_data = self.raw[condition_mask]

        # Coverage: Proportion of rows that satisfy the rule
        coverage = len(filtered_data) / len(self.raw) if len(self.raw) > 0 else 0

        # Precision: Proportion of rows that satisfy the rule and match the predicted class
        correct_predictions = filtered_data[filtered_data[explanation['rule']['consequence']['attr']] == explanation['rule']['consequence']['val']]
        precision = len(correct_predictions) / len(filtered_data) if len(filtered_data) > 0 else 0

        return precision, coverage



# def read_file(dataset_names: str, directory: str = './data/') -> pd.DataFrame:
#     _, _, stats_filenames = os.walk(directory).__next__()
#     for stat_filename in stats_filenames:
#         if dataset_names not in stat_filename:
#             continue
#         else:
#             with open(f'{directory}' + stat_filename, 'r') as file:
#                 print(f"Reading {stat_filename} from {directory}")
#                 return pd.read_csv(file)
#                 # return pd.read_csv(file, skipinitialspace=True, na_values='?', keep_default_na=True)
#     print("File not found")
#
#
# categorical_cols = [1,3,5,6,7,8,9,13]
# numeric_cols = [0,2,4,10,11,12]
#
# raw = read_file("adult")
# data = raw.iloc[:,:-1]
# numeric_col_names = data.columns[numeric_cols]
# categorical_col_names = data.columns[categorical_cols]
# amount = 10
# idx = 0
# unique_categories = [data.iloc[:, col].astype(str).unique().tolist() for col in categorical_cols]
#
# num_imputer = SimpleImputer(strategy='mean')
# cat_imputer = SimpleImputer(fill_value="?", strategy='constant')
# # data[numeric_col_names] = num_imputer.fit_transform(data[numeric_col_names])
# # data[categorical_col_names] = cat_imputer.fit_transform(data[categorical_col_names])
#
# target_names = raw.iloc[:, -1].squeeze().unique()
# target_map = {name: idx for idx, name in enumerate(target_names)}
# target = raw.iloc[:, -1].squeeze().map(target_map)
# X_train, X_test, y_train, y_test = train_test_split(data, raw['class'],
#                                             test_size=0.3, random_state=42, stratify=raw['class'])
# preprocessor = ColumnTransformer(
#     transformers=[
#         # ('num', StandardScaler(), num_cols),
#         ('cat', OrdinalEncoder(categories=unique_categories, handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
#     ]
# )
#
# clf = make_pipeline(preprocessor, RandomForestClassifier(max_depth=20, n_estimators=50, random_state=42))
# clf.fit(X_train, y_train)
# bbox = sklearn_classifier_bbox.sklearnBBox(clf)
#
#
# dataset = TabularDataset.from_csv('data/adult.csv', numeric_col_names, categorical_col_names, class_name='class', dropna = False)
#
# dataset.update_descriptor()
#
# enc = ColumnTransformerEnc(dataset.descriptor)
# generator = GeneticGenerator(bbox, dataset, enc)
# surrogate = DecisionTreeSurrogate()
#
# tabularLore = Lore(bbox, dataset, enc, generator, surrogate)
#
# # instance = X_test.iloc[0].to_numpy()
# instance = X_test.to_numpy()[0]
# explanation = tabularLore.explain(instance)
# print("IF ", end='')
# conditions = [f"{part['attr']} {part['op']} {part['val']}" for part in explanation['rule']['premises']]
# print(" AND ".join(conditions), end=' ')
# print(f"THEN {explanation['rule']['consequence']['val']}")