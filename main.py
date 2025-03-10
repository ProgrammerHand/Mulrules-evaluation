import dataset
import Anchor
import Lore_sa
import LUX
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from alibi.explainers import AnchorTabular
from sklearn.metrics import accuracy_score
from xailib.models.sklearn_classifier_wrapper import sklearn_classifier_wrapper

dataset = dataset.dataset("adult")
# dataset.label_encoding()
dataset.onehot_encode()
dataset.split_dataset(use_labeled = False, random_state=42)
dataset.init_preprocessor()
amount = 10
idx = 0

clf = RandomForestClassifier(max_depth=20, n_estimators=50, random_state=42)
clf.fit(dataset.preprocessor.transform(dataset.X_train), dataset.y_train)
predict_fn = lambda x: clf.predict(dataset.preprocessor.transform(x))
print('Train accuracy: ', accuracy_score(dataset.y_train, predict_fn(dataset.X_train)))
print('Test accuracy: ', accuracy_score(dataset.y_test, predict_fn(dataset.X_test)))

anchor_explainer = Anchor.anchor_object(dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test, dataset.feature_names, dataset.category_map)
anchor_explainer.init_explainer(predict_fn)
anchor_explainer.get_instance(idx)

explanations = [[], [], []]
explanation = anchor_explainer.explain()
explanations[0].append(explanation)
for elem in explanations[0]:
    anchor_explainer.print_expalanation(elem)

# while len(explanations[0]) < amount:
#     explanation = anchor_explainer.explain()
#     if explanation.anchor not in [entry.anchor for entry in explanations[0]]:
#         explanations[0].append(explanation)
# for elem in explanations[0]:
#     anchor_explainer.print_expalanation(elem)


dataset.split_dataset(use_ohe=True, random_state=0)

clf = RandomForestClassifier(n_estimators=20, random_state=42)
clf.fit(dataset.X_train, dataset.y_train)
bbox = sklearn_classifier_wrapper(clf)
print('Train accuracy: ', accuracy_score(dataset.y_train, clf.predict(dataset.X_train)))
print('Test accuracy: ', accuracy_score(dataset.y_test, clf.predict(dataset.X_test)))

lore_explainer = Lore_sa.lore_object(dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test, dataset.raw_ohe)
lore_explainer.init_explainer(bbox)
lore_explainer.get_instance(idx)

while len(explanations[2]) < amount:
    explanation = lore_explainer.explain()
    if explanation.exp.rule not in [entry.exp.rule for entry in explanations[1]]:
        explanations[2].append(explanation)
for elem in explanations[2]:
    lore_explainer.print_expalanation(elem)

# lux_explainer = LUX.lux_object(dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test)
# lux_explainer.get_instance(idx)
# lux_explainer.init_explainter(clf.predict_proba)
#
# while len(explanations[1]) < amount:
#     explanation = lux_explainer.explain(dataset.data_ohe, dataset.target, dataset.target_map)
#     if explanation not in [entry for entry in explanations[1]]:
#         explanations[1].append(explanation)
# for elem in explanations[1]:
#     lux_explainer.print_expalanation(elem)

