import dataset_manager
import Anchor
import LORE
import LUX
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xailib.models.sklearn_classifier_wrapper import sklearn_classifier_wrapper
import os
import logging
import random


def setup_logger(log_file, name):
    logger = logging.getLogger(name)  # Create a named logger for each dataset
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='w')  # 'w' to overwrite, 'a' to append
    file_handler.setFormatter(logging.Formatter('%(message)s'))  # Log only message content

    logger.addHandler(file_handler)
    return logger

# experiment_datasets = ["adult", "german"]
# experiment_datasets = ["titanic_nan_prepr", "boston_nan_prepr"]
experiment_datasets = ["boston_nan_prepr"]
rules_amount = 5
entries_amount = 3
log_folder_name = "experiments_log"
if not os.path.exists("experiments_log"):
    os.makedirs("experiments_log")

for name in experiment_datasets:
    log_file = os.path.join(log_folder_name, name + ".txt")
    logger = setup_logger(log_file, name)
    dataset = dataset_manager.dataset_object(name)
    dataset.label_encoding()
    dataset.onehot_encode()
    logger.info(dataset.split_dataset(use_labeled=True, use_ohe=False, random_state=42))
    dataset.init_preprocessor()


    idxs2e = set()
    while len(idxs2e) < entries_amount:
        idxs2e.add(random.randint(0, len(dataset.X_test)))

    clf = RandomForestClassifier(max_depth=20, n_estimators=50, random_state=42)
    logger.info("Classifier: RandomForestClassifier\n Params: max_depth=20, n_estimators=50, random_state=42 ")
    clf.fit(dataset.preprocessor.transform(dataset.X_train), dataset.y_train)
    predict_fn = lambda x: clf.predict(dataset.preprocessor.transform(x))
    bbox = sklearn_classifier_wrapper(clf)
    logger.info('Train accuracy: %f', accuracy_score(dataset.y_train, predict_fn(dataset.X_train)))
    logger.info('Test accuracy: %f', accuracy_score(dataset.y_test, predict_fn(dataset.X_test)))
    # Anchor
    anchor_explainer = Anchor.anchor_object(dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test,
                                            dataset.feature_names, dataset.category_map, dataset.target_names)
    logger.info(anchor_explainer.init_explainer(predict_fn, ohe=False))


    # Lore
    lore_explainer = LORE.lore_object(dataset.preprocessor.transform(dataset.X_train), dataset.y_train,
                                      dataset.preprocessor.transform(dataset.X_test), dataset.y_test, dataset.raw,
                                      config={"neigh_type": "geneticp", "size": 1000, "ocr": 0.1, "ngen": 10})
    logger.info(lore_explainer.init_explainer(bbox))


    # Lux
    lux_explainer = LUX.lux_object(dataset.preprocessor.transform(dataset.X_train), dataset.y_train,
                                   dataset.preprocessor.transform(dataset.X_test), dataset.y_test)

    logger.info(lux_explainer.init_explainter(clf.predict_proba))

    for idx in idxs2e:
        anchor_explainer.get_instance(idx)
        lore_explainer.get_instance(idx)
        lux_explainer.get_instance(idx)
        #explain
        logger.info(
            f"Explaining instance: {dataset.data_labeled.index[(dataset.data_labeled == dataset.X_test[idx]).all(axis=1)].tolist()[0]} outcome: {dataset.target_names[dataset.y_test.iloc[idx]]}")
        print(f"Explaining instance: {dataset.data_labeled.index[(dataset.data_labeled == dataset.X_test[idx]).all(axis=1)].tolist()[0]} outcome: {dataset.target_names[dataset.y_test.iloc[idx]]}")
        explanations = anchor_explainer.explain(rules_amount, beam_size=5, verbose=False)
        for rule in explanations:
            logger.info(anchor_explainer.print_expalanation(rule))
        explanations = lore_explainer.explain(rules_amount)
        for rule in explanations:
            logger.info(lore_explainer.print_explanation(rule))
        explanation = lux_explainer.explain(dataset.data_ohe, dataset.target, dataset.target_map,
                                            dataset.numerical_features)
        logger.info(lux_explainer.print_expalanation(explanation))
