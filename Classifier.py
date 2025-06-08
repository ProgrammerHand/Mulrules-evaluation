import deap
from xailib.models.bbox import AbstractBBox
import numpy as np

class sklearn_classifier_wrapper_custom(AbstractBBox):
    def __init__(self, classifier, custom_scaler=None, transformer=None):
        super().__init__()
        self.bbox = classifier
        if custom_scaler:
            self.custom_scaler = custom_scaler
        if transformer:
            self.transformer = transformer

    def model(self):
        return self.bbox

    def predict(self, X):
        if isinstance(X, deap.creator.individual):
            data = np.array([[float(x) for x in np.ravel(X.copy())]], dtype=object)
        elif X.shape[0] > 1:
            data = np.array([[float(x) for x in row] for row in X], dtype=object)
        else:
            data = X.copy()
            # np.array([[float(x) for x in X.copy()]], dtype=object)
        data = self.custom_scaler.transform(data) if hasattr(self, 'custom_scaler') else data
        data = self.transformer.transform(data) if hasattr(self, 'transformer') else data
        return self.bbox.predict(data)

    def predict_proba(self, X):
        # print(type(X))
        # print(type(X[0][0]))
        # print(X.dtype)
        # if isinstance(X, deap.creator.individual):
        if type(X) != np.array:
            data = np.array([[float(x) for x in np.ravel(X.copy())]], dtype=object)
        elif X.shape[0] > 1:
            data = np.array([[float(x) for x in row] for row in X], dtype=object)
        else:
            data = X.copy()
        data = self.custom_scaler.transform(data) if hasattr(self, 'custom_scaler') else data
        data = self.transformer.transform(data) if hasattr(self, 'transformer') else data
        return self.bbox.predict_proba(data)

def create_classifier(experiment_name, classifier_name, classifiers_names, classifier_parametrs):
    if classifier_name in classifiers_names and classifier_name in classifier_parametrs:
        # Instantiate the classifier using the lambda function
        classifier = classifiers_names[classifier_name]()
        if classifier_name == "simpleNN":
            return classifier(**classifier_parametrs[classifier_name][experiment_name])
        # Set parameters using the classifier's parameters from classifier_parametrs
        classifier.set_params(**classifier_parametrs[classifier_name][experiment_name])
        return classifier
    else:
        raise ValueError(f"Classifier '{classifier_name}' not found or missing parameters.")

def get_predict_functions(dataset, clf, custom_scaler):
    """
    Returns predict_fn, predict_probab_fn, and predict_fn_anchor
    that handle optional presence of continuous and categorical columns.
    """
    if dataset.continuous_cols:
        if dataset.categorical_cols:
            predict_fn = lambda x: clf.predict(dataset.onehot_encoder.transform(custom_scaler.transform(x)))
            predict_probab_fn = lambda x: clf.predict_proba(dataset.onehot_encoder.transform(
                    custom_scaler.transform(dataset.label_decode_features(x, dataset.categorical_cols, dataset.categorical_col_names))
                )
            )
            predict_fn_anchor = lambda x: clf.predict(dataset.onehot_encoder.transform(
                    custom_scaler.transform(dataset.label_decode_features(x, dataset.categorical_cols, dataset.categorical_col_names))
                )
            )
        else:
            predict_fn = lambda x: clf.predict(custom_scaler.transform(x))
            predict_probab_fn = lambda x: clf.predict_proba(custom_scaler.transform(x))
            predict_fn_anchor = lambda x: clf.predict(custom_scaler.transform(x))
    else:
        if dataset.categorical_cols:
            predict_fn = lambda x: clf.predict(dataset.onehot_encoder.transform(x))
            predict_probab_fn = lambda x: clf.predict_proba(dataset.onehot_encoder.transform(
                    dataset.label_decode_features(x, dataset.categorical_cols)))

            predict_fn_anchor = lambda x: clf.predict(
                dataset.onehot_encoder.transform(
                    dataset.label_decode_features(x, dataset.categorical_cols)))
        else:
            predict_fn = lambda x: clf.predict(x)
            predict_probab_fn = lambda x: clf.predict_proba(x)
            predict_fn_anchor = lambda x: clf.predict(x)

    return predict_fn, predict_probab_fn, predict_fn_anchor

def get_balanced_correct_indexes(pred_funct, X_test, y_test, n, instance_2e):

    # remove choosen indexes
    X_test_dropped = X_test.drop(instance_2e)
    y_test_dropped = y_test.drop(instance_2e)
    positional_indexes_instance_2e = X_test.index.get_indexer(instance_2e).tolist()
    # predictions
    y_pred = pred_funct(X_test_dropped)

    # correct predictions
    correct_mask = y_pred == y_test_dropped
    correct_indices = np.where(correct_mask)[0]


    # X_correct = X_test[correct_mask]
    # y_correct = y_test[correct_mask]

    # how many samples per class
    classes = np.unique(y_test_dropped)
    n_per_class = n // len(classes)

    selected_indices = []

    for cls in classes:
        # indices where the true class is cls and prediction is correct
        # cls_indices = np.where((y_correct == cls))[0]
        cls_correct_indices = np.where((y_test == cls) & correct_mask)[0]

        if len(cls_correct_indices) < n_per_class:
            print(f"Not enough correct samples for class {cls}. Requested {n_per_class}, but only {len(cls_correct_indices)} available.")
            # sampled = np.random.choice(cls_indices, size=len(cls_indices), replace=False)
            sampled = np.random.choice(cls_correct_indices, size=len(cls_correct_indices), replace=False)
        else:
            sampled = np.random.choice(cls_correct_indices, size=n_per_class, replace=False)
            # sampled = np.random.choice(cls_indices, size=n_per_class, replace=False)

        # map back to original X_test index
        selected_indices.extend(sampled.tolist())

    return positional_indexes_instance_2e + selected_indices