import numpy as np

from Rule_wrapper import rule_wrapper
from alibi.explainers import AnchorTabular
from collections import Counter
import time
from logger import log_with_custom_tag
import dataset_manager_legacy
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

# dataset = dataset.dataset("adult")
# dataset.label_encoding()
# dataset.split_dataset(use_labeled = True)
# dataset.init_preprocessor()
#
# clf = RandomForestClassifier(max_depth=20, n_estimators=50, random_state=42)
# clf.fit(dataset.preprocessor.transform(dataset.X_train), dataset.y_train)
#
# predict_fn = lambda x: clf.predict(dataset.preprocessor.transform(x))
# print('Train accuracy: ', accuracy_score(dataset.y_train, predict_fn(dataset.X_train)))
# print('Test accuracy: ', accuracy_score(dataset.y_test, predict_fn(dataset.X_test)))
#
# explainer = AnchorTabular(predict_fn, dataset.feature_names, categorical_names=dataset.category_map, seed=1)
# explainer.fit(dataset.X_train)
#
# idx = 0
# inst = dataset.X_test[idx]
#
# print(f"Prediction: {dataset.target_names[explainer.predictor(inst.reshape(1, -1))[0]]}, Correct: {dataset.target_names[dataset.y_test[dataset.y_test.index[idx]]]}")
# explanation = explainer.explain(inst)
# print('Anchor: IF %s' % (' AND '.join(explanation.anchor) + f' THEN {dataset.target_names[explainer.predictor(inst.reshape(1, -1))[0]]}'))
# print('Precision: %.2f' % explanation.precision)
# print('Coverage: %.2f' % explanation.coverage)

class anchor_object:
    def __init__(self,X_train, y_train, X_test, y_test, feature_names, category_map, target_names, continuous_col_name, numeric_cols_names):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.category_map = category_map
        self.target_names = target_names
        self.continuous_col_names = continuous_col_name
        self.numeric_cols_names = numeric_cols_names
        self.treshold = None
        self.beam_size = None

    def init_explainer(self, predict_fn, iter_limit=15, treshold = 0.9, beam_size = 1, ohe=False, seed=None):
        self.explainer = AnchorTabular(predict_fn, self.feature_names, categorical_names=self.category_map, ohe=ohe, seed=seed) # seed to control random
        self.explainer.fit(self.X_train)
        self.treshold = treshold
        self.beam_size = beam_size
        self.iter_limit = iter_limit
        return f"Initializing Anchor Explainer with params: precision_treshold = {treshold}, beam_size = {beam_size}, feature_names = {self.feature_names}, categorical_names = {self.category_map}, seed = {seed}, iter_limit = {iter_limit}"

    def get_instance(self, idx):
        self.inst = self.X_test[idx]

    def explain(self, amount, verbose=False):
        explanations = []
        rejected_count = Counter({i: 0 for i in range(amount)})
        i = 0
        start_time = time.time()
        # while len(explanations) < amount:
        start_rule_time = time.time()
        predicted_class = self.target_names[self.explainer.predictor(self.inst.reshape(1, -1))[0]]
        for n in range (self.iter_limit):
            print(n)
            explanation = self.explainer.explain(self.inst, threshold=self.treshold, beam_size = self.beam_size, verbose=verbose)
            # explanation.anchor = sorted(explanation.anchor)
            if len(explanations) > 0:
                if amount - len(explanations) < self.iter_limit - n:
                    flag = False
                    for rule in explanations:
                        if rule.matches_raw_rule(explanation.anchor, f"class = {predicted_class}", self.numeric_cols_names):
                            flag = True
                            rejected_count[i] += 1
                            break
                    if not flag:
                        elapsed = time.time() - start_rule_time
                        explanations.append(
                                rule_wrapper.from_rule(explanation.anchor, f"class = {predicted_class}", "ANCHOR", rejected_count[i], elapsed, False, self.numeric_cols_names))
                        i += 1
                        start_rule_time = time.time()
                        # print(f"Pre, Cov: {explanation.precision},{explanation.coverage}")
                else:
                    elapsed = time.time() - start_rule_time
                    explanations.append(
                        rule_wrapper.from_rule(explanation.anchor, f"class = {predicted_class}", "ANCHOR", rejected_count[i], elapsed, True,
                                               self.continuous_col_names))
                    i += 1
                    start_rule_time = time.time()
            else:
                elapsed = time.time() - start_rule_time
                explanations.append(rule_wrapper.from_rule(explanation.anchor, f"class = {predicted_class}", "ANCHOR", rejected_count[i], elapsed, False, self.numeric_cols_names))
                i += 1
                start_rule_time = time.time()
            if len(explanations) >= amount:
                break
                # print(f"Pre, Cov: {explanation.precision},{explanation.coverage}")
            # if explanation.anchor not in [entry.anchor for entry in explanations]:
            #     explanations.append(explanation)
        return explanations
        # return self.explainer.explain(self.inst)

    def print_explanation(self, explanation):
        anchor_conditions = ' AND '.join(explanation.anchor)
        # predicted_class = self.target_names[self.explainer.predictor(self.inst.reshape(1, -1))[0]]
        predicted_class = self.explainer.predictor(self.inst.reshape(1, -1))[0]
        if type(predicted_class) == np.ndarray:
            predicted_class = int(predicted_class[0])
        result_string = f"Anchor: IF {anchor_conditions} THEN {self.target_names[predicted_class]} Pre, Cov : ({explanation.precision}, {explanation.coverage})"
        return result_string
        # print('Anchor: IF %s' % (' AND '.join(
        #     explanation.anchor) + f' THEN {self.target_names[self.explainer.predictor(self.inst.reshape(1, -1))[0]]}' + f" Pre {explanation.precision}" + f" Cov {explanation.coverage}"))