import dataset_manager_legacy
import Anchor
import LORE
import LUX
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from xailib.models.sklearn_classifier_wrapper import sklearn_classifier_wrapper

dataset = dataset_manager_legacy.dataset_object("adult")
dataset.label_encoding()
dataset.onehot_encode()
dataset.split_dataset(use_labeled = True, use_ohe=False, random_state=42)
dataset.init_preprocessor()
amount = 10
idx = 0

clf = RandomForestClassifier(max_depth=20, n_estimators=50, random_state=42)
clf.fit(dataset.preprocessor.transform(dataset.X_train), dataset.y_train)
predict_fn = lambda x: clf.predict(dataset.preprocessor.transform(x))
bbox = sklearn_classifier_wrapper(clf)
print('Train accuracy: ', accuracy_score(dataset.y_train, predict_fn(dataset.X_train)))
print('Test accuracy: ', accuracy_score(dataset.y_test, predict_fn(dataset.X_test)))

anchor_explainer = Anchor.anchor_object(dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test, dataset.feature_names, dataset.category_map, dataset.target_names)
anchor_explainer.init_explainer(predict_fn, ohe=False)
anchor_explainer.get_instance(idx)


explanations = anchor_explainer.explain(1)
for elem in explanations:
    anchor_explainer.print_explanation(elem)
print("test")
# while len(explanations[0]) < amount:
#     explanation = anchor_explainer.explain()
#     if explanation.anchor not in [entry.anchor for entry in explanations[0]]:
#         explanations[0].append(explanation)
# for elem in explanations[0]:
#     anchor_explainer.print_explanation(elem)


# dataset.split_dataset(use_ohe=True, random_state=0)

lore_explainer = LORE.lore_object_old(dataset.preprocessor.transform(dataset.X_train), dataset.y_train, dataset.preprocessor.transform(dataset.X_test), dataset.y_test, dataset.raw, config = {"neigh_type": "geneticp", "size": 1000, "ocr": 0.1, "ngen": 10})
lore_explainer.init_explainer(bbox)
lore_explainer.get_instance(idx)

explanations = lore_explainer.explain(1)
for elem in explanations:
    lore_explainer.print_explanation(elem)

# while len(explanations[2]) < amount:
#     explanation = lore_explainer.explain()
#     if explanation.exp.rule not in [entry.exp.rule for entry in explanations[1]]:
#         explanations[2].append(explanation)
# for elem in explanations[2]:
#     lore_explainer.print_explanation(elem)

# clf1 = RandomForestClassifier(max_depth=20, n_estimators=50)
# clf1.fit(dataset.X_train, dataset.y_train)
# print('Train accuracy: ', accuracy_score(dataset.y_train, predict_fn(dataset.X_train)))
# print('Test accuracy: ', accuracy_score(dataset.y_test, predict_fn(dataset.X_test)))

# def wrapped_predict_proba(x):
#     if  len(dataset.preprocessor.get_feature_names_out()) == x.shape[1]:
#         return clf.predict_proba(x)
#     else:
#         return clf.predict_proba(dataset.preprocessor.transform(x))

# lux_explainer = LUX.lux_object(dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test)
lux_explainer = LUX.lux_object(dataset.preprocessor.transform(dataset.X_train), dataset.y_train, dataset.preprocessor.transform(dataset.X_test), dataset.y_test)
lux_explainer.get_instance(idx)
# lux_explainer.init_explainer(wrapped_predict_proba)
lux_explainer.init_explainer(clf.predict_proba)

explanations = []
explanation = lux_explainer.explain(dataset.data_ohe, dataset.target, dataset.target_map, dataset.numerical_features)
explanations.append(explanation)
explanation = lux_explainer.explain(dataset.data_labeled, dataset.target, dataset.target_map, dataset.numerical_features)
explanations.append(explanation)
for elem in explanations:
    lux_explainer.print_explanation(elem)

# while len(explanations[1]) < amount:
#     explanation = lux_explainer.explain(dataset.data_ohe, dataset.target, dataset.target_map)
#     if explanation not in [entry for entry in explanations[1]]:
#         explanations[1].append(explanation)
# for elem in explanations[1]:
#     lux_explainer.print_explanation(elem)

