import dataset_manager_legacy
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from lux.lux import LUX
import numpy as np
from sklearn.metrics import accuracy_score

# dataset = dataset.dataset("adult_cleaned")
# dataset.split_dataset()
# dataset.init_preprocessor()
# fraction=0.01
#
#
# clf = RandomForestClassifier(max_depth=20, n_estimators=50, random_state=42)
# clf.fit(dataset.X_train, dataset.y_train)
# print('Train accuracy: ', accuracy_score(dataset.y_train, clf.predict(dataset.X_train)))
# print('Test accuracy: ', accuracy_score(dataset.y_test, clf.predict(dataset.X_test)))
#
# predict_fn = lambda x: clf.predict(dataset.preprocessor.transform(x))
# idx = 0
# inst = dataset.X_test[idx].reshape(1, -1)
#
# print(f"Preditction: {dataset.target_names[clf.predict(inst)]}, Correct: {dataset.target_names[dataset.y_test[dataset.y_test.index[idx]]]}")
#
# lux = LUX(predict_proba = clf.predict_proba, neighborhood_size=int(len(dataset.X_train)*fraction), max_depth=5,  node_size_limit = 2, grow_confidence_threshold = 0 )
# lux.fit(dataset.data, dataset.target, instance_to_explain=inst, class_names=list(dataset.target_map.values()))
# for i in range(2):
#     print(lux.justify(inst))

class lux_object:
    def __init__(self, X_train, y_train, X_test, y_test, reverse_target_map, fraction=0.05):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.reverse_target_map = reverse_target_map
        self.fraction = fraction

    def init_explainer(self, predict_proba, max_depth=10, node_size_limit=1, grow_confidence_threshold=0, min_samples = 5):
        self.explainer = LUX(predict_proba=predict_proba, neighborhood_size=int(len(self.X_train) * self.fraction), max_depth=max_depth,
                  node_size_limit=node_size_limit, grow_confidence_threshold=grow_confidence_threshold, min_samples = min_samples)
        return f"Initializing LUX with params neighborhood_size = {int(len(self.X_train) * self.fraction)}, max_depth = {max_depth}, min_samples = {min_samples}, node_size_limit = {node_size_limit}, grow_confidence_threshold = {grow_confidence_threshold}"

    def get_instance(self, idx):
        if type(self.X_test) == np.ndarray:
            self.inst = self.X_test[idx].reshape(1, -1)
        else:
            self.inst = self.X_test.to_numpy()[idx].reshape(1, -1)

    def normalize_rule(self, rule_dict):
        conditions = [f"{feature} {condition[0]}" for feature, condition in rule_dict.items()]
        return tuple(sorted(conditions))

    def explain(self, data, target, categorical_cols_names, amount = 1):
        explanations = []
        while len(explanations) < amount:
            self.explainer.fit(data, target, instance_to_explain=self.inst, categorical=categorical_cols_names)
            explanation = self.explainer.justify(self.inst, to_dict=True)
            if self.normalize_rule(explanation[0][0]["rule"]) not in [self.normalize_rule(entry[0][0]["rule"]) for entry in explanations]:
                explanations.append(explanation)
        return explanations

    def print_explanation(self, explanation):
        # print(f"LUX: {explanation[0]}")
        rule_dict = explanation[0][0]['rule']
        conditions = [f"{feature} {condition[0]}" for feature, condition in sorted(rule_dict.items())]
        rule_string = "LUX: IF " + " AND ".join(conditions)
        predicted_class = self.reverse_target_map.get(int(explanation[0][0]['prediction']), explanation[0][0]['prediction'])
        rule_string += f" THEN class = {predicted_class}  # confidence = {explanation[0][0]['confidence']}"
        return rule_string
