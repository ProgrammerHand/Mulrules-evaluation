import dataset_manager
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from alibi.explainers import AnchorTabular
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
    def __init__(self,X_train, y_train, X_test, y_test, feature_names, category_map, target_names):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.category_map = category_map
        self.target_names = target_names

    def init_explainer(self, predict_fn, ohe=False, seed=1):
        self.explainer = AnchorTabular(predict_fn, self.feature_names, categorical_names=self.category_map, ohe=ohe, seed=seed) # seed to control random
        self.explainer.fit(self.X_train)
        return f"Initializing Anchor Explainer with params: feature_names = {self.feature_names}, categorical_names {self.category_map}, seed = {seed}"

    def get_instance(self, idx):
        self.inst = self.X_test[idx]

    def explain(self, amount, threshold=0.95, beam_size=1, verbose=False):
        explanations = []
        while len(explanations) < amount:
            explanation = self.explainer.explain(self.inst, threshold=threshold, beam_size = beam_size, verbose=verbose)
            if explanation.anchor not in [entry.anchor for entry in explanations]:
                explanations.append(explanation)
        return explanations
        # return self.explainer.explain(self.inst)

    def print_expalanation(self, explanation):
        anchor_conditions = ' AND '.join(explanation.anchor)
        predicted_class = self.target_names[self.explainer.predictor(self.inst.reshape(1, -1))[0]]
        result_string = f"Anchor: IF {anchor_conditions} THEN {predicted_class} Pre, Cov : ({explanation.precision}, {explanation.coverage})"
        return result_string
        # print('Anchor: IF %s' % (' AND '.join(
        #     explanation.anchor) + f' THEN {self.target_names[self.explainer.predictor(self.inst.reshape(1, -1))[0]]}' + f" Pre {explanation.precision}" + f" Cov {explanation.coverage}"))