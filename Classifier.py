import os, torch, deap
from datetime import datetime
from xailib.models.bbox import AbstractBBox
from joblib import dump, load
from pathlib import Path
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
        data = self.custom_scaler.transform(data) if hasattr(self, 'custom_scaler') else data
        data = self.transformer.transform(data) if hasattr(self, 'transformer') else data
        return self.bbox.predict(data)

    def predict_proba(self, X):
        if type(X) != np.array:
            data = np.array([[float(x) for x in np.ravel(X.copy())]], dtype=object)
        elif X.shape[0] > 1:
            data = np.array([[float(x) for x in row] for row in X], dtype=object)
        else:
            data = X.copy()
        data = self.custom_scaler.transform(data) if hasattr(self, 'custom_scaler') else data
        data = self.transformer.transform(data) if hasattr(self, 'transformer') else data
        return self.bbox.predict_proba(data)

def create_classifier(experiment_name, classifier_name, classifiers_names, classifier_parametrs, device):
    if classifier_name in classifiers_names and classifier_name in classifier_parametrs:
        classifier = classifiers_names[classifier_name]()
        params = classifier_parametrs[classifier_name][experiment_name]
        if classifier_name == "simpleNN":
            model = classifier(**params).to(device)

            model_path = Path(f"models/{experiment_name}_{classifier_name}_{datetime.now().strftime('%m-%Y')}.pt")
            if model_path.is_file():
                ckpt = torch.load(model_path, map_location=device, weights_only=False)
                state = ckpt.get("state_dict", ckpt)
                model.load_state_dict(state, strict=True)
                if "decision_threshold" in ckpt:
                    model.decision_threshold = float(ckpt["decision_threshold"])
                model.eval()
                return model, True
            return model, False

        classifier.set_params(**params)
        model_path = Path(f"models/{experiment_name}_{classifier_name}_{datetime.now().strftime('%m-%Y')}.joblib")
        if model_path.is_file():
            classifier = load(model_path)
            return classifier, True
        return classifier, False
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
        # indices where true class is cls and prediction is correct
        # cls_indices = np.where((y_correct == cls))[0]
        cls_correct_indices = np.where((y_test == cls) & correct_mask)[0]

        if len(cls_correct_indices) < n_per_class:
            print(f"Not enough correct samples for class {cls}. Requested {n_per_class}, but only {len(cls_correct_indices)} available.")
            sampled = np.random.choice(cls_correct_indices, size=len(cls_correct_indices), replace=False)
        else:
            sampled = np.random.choice(cls_correct_indices, size=n_per_class, replace=False)

        # map back to original X_test index
        selected_indices.extend(sampled.tolist())

    return positional_indexes_instance_2e + selected_indices

def save_model(model, experiment_name, classifier_name, decision_threshold = None):
    if classifier_name == "simpleNN":
        extension = ".pt"
    else:
        extension = ".joblib"
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{experiment_name}_{classifier_name}_{datetime.now().strftime('%m-%Y')}"+extension
    if classifier_name == "simpleNN":
        torch.save(
            {
                "state_dict": model.state_dict(),
                "decision_threshold": float(getattr(model, "decision_threshold", decision_threshold)),
            },
            model_path,
        )
    else:
        dump(model, model_path)
    return model_path