import dataset_manager
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from lux.lux import LUX
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
    def __init__(self, X_train, y_train, X_test, y_test, fraction=0.05):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.fraction = fraction

    def init_explainter(self, predict_proba, max_depth=5, node_size_limit=2, grow_confidence_threshold=0 ):
        self.explainer = LUX(predict_proba=predict_proba, neighborhood_size=int(len(self.X_train) * self.fraction), max_depth=max_depth,
                  node_size_limit=node_size_limit, grow_confidence_threshold=grow_confidence_threshold)
        return f"Initializing LUX with params neighborhood_size = {int(len(self.X_train) * self.fraction)}, max_depth = {max_depth}, node_size_limit = {node_size_limit}, grow_confidence_threshold {grow_confidence_threshold}"

    def get_instance(self, idx):
        self.inst = self.X_test[idx].reshape(1, -1)

    def explain(self, data, target, target_map, numeric_vars):
        self.explainer.fit(data, target, instance_to_explain=self.inst, class_names=list(target_map.values()), categorical=[col not in numeric_vars for col in data.columns])
        return self.explainer.justify(self.inst)

    def print_expalanation(self, explanation):
        # print(f"LUX: {explanation[0]}")
        return explanation[0]
