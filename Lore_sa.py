import dataset
import pandas as pd
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

class lore_object:
    def __init__(self, X_train, y_train, X_test, y_test, raw_ohe, config = {"neigh_type": "geneticp", "size": 1000, "ocr": 0.1, "ngen": 10}):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.raw_ohe = raw_ohe
        self.config = config

    def init_explainer(self, bbox):
        self.explainer = LoreTabularExplainer(bbox, )
        self.explainer.fit(self.raw_ohe, self.raw_ohe.columns[-1], self.config)
        print(f"Initializing LORE Explainer with config: {self.config}")

    def get_instance(self, idx):
        self.inst = self.X_test[idx]

    def explain(self):
        return self.explainer.explain(self.inst)

    def print_explanation(self, explanation):
        print("LORE: IF", end=" ")
        conditions = " AND ".join(str(part) for part in explanation.exp.rule.premises)
        print(f"{conditions} THEN {explanation.exp.rule.cons}")




# Train a classification model
# X_train, X_test, Y_train, Y_test = train_test_split(
#     df[feature_names].values, df[class_field].values, test_size=0.3, random_state=0, stratify=df[class_field].values
# )