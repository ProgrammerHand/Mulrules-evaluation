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

def create_classifier(classifier_name, classifiers_names, classifier_parametrs):
    if classifier_name in classifiers_names and classifier_name in classifier_parametrs:
        # Instantiate the classifier using the lambda function
        classifier = classifiers_names[classifier_name]()
        if classifier_name == "simpleNN":
            return classifier(**classifier_parametrs[classifier_name])
        # Set parameters using the classifier's parameters from classifier_parametrs
        classifier.set_params(**classifier_parametrs[classifier_name])
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
                    custom_scaler.transform(dataset.label_decode_features(x, dataset.categorical_cols))
                )
            )
            predict_fn_anchor = lambda x: clf.predict(dataset.onehot_encoder.transform(
                    custom_scaler.transform(dataset.label_decode_features(x, dataset.categorical_cols))
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