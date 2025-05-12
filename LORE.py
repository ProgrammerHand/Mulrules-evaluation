import dataset_manager_legacy
import pandas as pd
import numpy as np
from lore_explainer.datamanager import prepare_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xailib.explainers.lime_explainer import LimeXAITabularExplainer
from xailib.explainers.lore_explainer import LoreTabularExplainer
from xailib.models.sklearn_classifier_wrapper import sklearn_classifier_wrapper
from sklearn.metrics import accuracy_score

# Load and preprocess dataset
# dataset = dataset.dataset("adult")
# dataset.onehot_encode()
# dataset.split_dataset(use_ohe=True, random_state=0)
# dataset.init_preprocessor()
#
# clf = RandomForestClassifier(n_estimators=20, random_state=0)
# clf.fit(dataset.X_train, dataset.y_train)
# print('Train accuracy: ', accuracy_score(dataset.y_train, clf.predict(dataset.X_train)))
# print('Test accuracy: ', accuracy_score(dataset.y_test, clf.predict(dataset.X_test)))
#
# # Wrap model for explanation
# bbox = sklearn_classifier_wrapper(clf)
# idx = 0
# inst = dataset.X_test[idx]
#
# print(f"Preditction: {dataset.target_names[bbox.predict(inst.reshape(1, -1))]}, Correct: {dataset.target_names[dataset.y_test[dataset.y_test.index[idx]]]}")
# # Configure and run LORE explainer
# config = {"neigh_type": "geneticp", "size": 1000, "ocr": 0.1, "ngen": 10}
# print(f"Initializing LORE Explainer with config: {config}")
#
# explainer = LoreTabularExplainer(bbox)
# explainer.fit(dataset.raw_ohe, dataset.raw.columns[-1], config)
# for i in range(2):
#     exp = explainer.explain(inst)
#     print(f"Explanation:\n{exp.exp}")

class lore_object_old:
    def __init__(self, X_train, y_train, X_test, y_test, raw, config = {"neigh_type": "geneticp", "size": 1000, "ocr": 0.1, "ngen": 10}):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.raw = raw
        self.config = config

    def init_explainer(self, bbox, numeric_columns_input = None):
        self.explainer = LoreTabularExplainer(bbox)
        self.explainer.fit(self.raw, self.raw.columns[-1], self.config, numeric_columns_input)
        return f"Initializing LORE_xailib Explainer with config: {self.config}"

    def get_instance(self, idx):
        if type(self.X_test) == np.ndarray:
            self.inst = self.X_test[idx]
        else:
            self.inst = self.X_test.to_numpy()[idx]

    def explain(self, amount):
        explanations = []
        while len(explanations) < amount: # TODO speed up rule comparsion
            explanation = self.explainer.explain(self.inst)
            explanation.exp.rule.premises = sorted(explanation.exp.rule.premises, key=lambda x: x.att)
            if explanation.exp.rule.premises not in [entry.exp.rule.premises for entry in explanations]:
                explanations.append(explanation)
        return explanations
        # return self.explainer.explain(self.inst)

    def print_explanation(self, explanation):
        # print("LORE: IF", end=" ")
        # conditions = " AND ".join(str(part) for part in explanation.exp.rule.premises)
        # print(f"{conditions} THEN {explanation.exp.rule.cons} Cov, Pre : {self.calculate_coverage_precision(explanation)}")
        conditions = " AND ".join(str(part) for part in explanation.exp.rule.premises)
        result_string = f"LORE_xailib: IF {conditions} THEN {explanation.exp.rule.cons} Pre, Cov : {self.calculate_precision_coverage(explanation)}"
        return result_string

    def apply_condition(self, condition):# TODO check ops
        if condition.is_continuous:
            if condition.op == '<=':
                return self.raw[condition.att] <= condition.thr
            elif condition.op == '<':
                return self.raw[condition.att] < condition.thr
            elif condition.op == '>=':
                return self.raw[condition.att] >= condition.thr
            elif condition.op == '>':
                return self.raw[condition.att] > condition.thr
        else:
            temp = condition.att.split("=")
            if condition.op == '>':
                return self.raw[temp[0]] == temp[1]
            elif condition.op == '<=':
                return self.raw[temp[0]] != temp[1]

    def calculate_precision_coverage(self, explanation):

        condition_mask = pd.Series([True] * len(self.raw))  # Start with all True (all rows selected)

        for condition in explanation.exp.rule.premises:
            condition_mask &= self.apply_condition(condition)

        # Filter rows that satisfy the rule
        filtered_data = self.raw[condition_mask]

        # Coverage: Proportion of rows that satisfy the rule
        coverage = len(filtered_data) / len(self.raw) if len(self.raw) > 0 else 0

        # Precision: Proportion of rows that satisfy the rule and match the predicted class
        correct_predictions = filtered_data[filtered_data[explanation.exp.rule.class_name] == explanation.exp.rule.cons]
        precision = len(correct_predictions) / len(filtered_data) if len(filtered_data) > 0 else 0

        return precision, coverage


# Train a classification model
# X_train, X_test, Y_train, Y_test = train_test_split(
#     df[feature_names].values, df[class_field].values, test_size=0.3, random_state=0, stratify=df[class_field].values
# )