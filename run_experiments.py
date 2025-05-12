import os
import random
from typing import Dict, List

from sklearn.ensemble import RandomForestClassifier
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score, classification_report
from lore_sa.bbox import sklearn_classifier_bbox

import dataset_manager
import Anchor, LUX, LORE, LORE_2
from SimpleNN import SimpleNN
from logger import log_with_custom_tag, setup_logger
from Classifier import create_classifier, get_predict_functions, sklearn_classifier_wrapper_custom
from Scaler import CustomScaler

experiment_name = [
    # "adult",
    # "adult_numeric",
    "german",
    "german_numeric",
    # "fico_heloc",
    # "fico_heloc_numeric",
    # "titanic",
    # "nursery"
]
dataset_names: Dict[str, str] = {
    "adult": "adult",
    "adult_numeric": "adult",
    "german": "german",
    "german_numeric": "german",
    "fico_heloc": "fico",
    "fico_heloc_numeric": "fico",
    "titanic": "titanic",
    "nursery": "nursery"
}
drop_cols_names_datasets: Dict[str, List[str] | None] = {
    "adult" : None,
    "adult_numeric": ["workclass", "education", "marital.status", "occupation" ,"relationship", "race", "sex", "native.country"],
    "german": None,
    "german_numeric": ["checking_status","credit_history","purpose","savings_status","employment","personal_status","other_parties","property_magnitude","other_payment_plans","housing","job","own_telephone","foreign_worker"],
    "fico_heloc": None,
    "fico_heloc_numeric": ["MaxDelq2PublicRecLast12M", "MaxDelqEver"],
    "titanic": ["PassengerId", "Name"],
    "nursery": None,
    }

categorical_cols_names_datasets: Dict[str, List[str] | None] = {
    "adult": ["workclass", "education", "marital.status", "occupation" ,"relationship", "race", "sex", "native.country"],
    "adult_numeric": None,
    "german": ["checking_status","credit_history","purpose","savings_status","employment","personal_status","other_parties","property_magnitude","other_payment_plans","housing","job","own_telephone","foreign_worker"],
    "german_numeric": None,
    "fico_heloc": ["MaxDelq2PublicRecLast12M", "MaxDelqEver"],
    "fico_heloc_numeric": None,
    "titanic": ["Pclass", "Sex", "Embarked", "Cabin", "Ticket"],
    "nursery": ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health"],
}

numeric_cols_names_datasets: Dict[str, List[str]] = {
    "adult": ["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"],
    "adult_numeric":  ["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"],
    "german": ["duration", "credit_amount", "installment_commitment", "residence_since", "age", "existing_credits", "num_dependents"],
    "german_numeric": ["duration", "credit_amount", "installment_commitment", "residence_since", "age", "existing_credits", "num_dependents"],
    "fico_heloc": ["ExternalRiskEstimate", "MSinceOldestTradeOpen", "MSinceMostRecentTradeOpen", "AverageMInFile", "NumSatisfactoryTrades", "NumTrades60Ever2DerogPubRec",
    "NumTrades90Ever2DerogPubRec", "PercentTradesNeverDelq", "MSinceMostRecentDelq", "NumTotalTrades", "NumTradesOpeninLast12M", "PercentInstallTrades",
    "MSinceMostRecentInqexcl7days", "NumInqLast6M", "NumInqLast6Mexcl7days", "NetFractionRevolvingBurden", "NetFractionInstallBurden", "NumRevolvingTradesWBalance",
    "NumInstallTradesWBalance", "NumBank2NatlTradesWHighUtilization", "PercentTradesWBalance"],
    "fico_heloc_numeric": ["ExternalRiskEstimate", "MSinceOldestTradeOpen", "MSinceMostRecentTradeOpen", "AverageMInFile", "NumSatisfactoryTrades", "NumTrades60Ever2DerogPubRec",
    "NumTrades90Ever2DerogPubRec", "PercentTradesNeverDelq", "MSinceMostRecentDelq", "NumTotalTrades", "NumTradesOpeninLast12M", "PercentInstallTrades",
    "MSinceMostRecentInqexcl7days", "NumInqLast6M", "NumInqLast6Mexcl7days", "NetFractionRevolvingBurden", "NetFractionInstallBurden", "NumRevolvingTradesWBalance",
    "NumInstallTradesWBalance", "NumBank2NatlTradesWHighUtilization", "PercentTradesWBalance"],
    "titanic": ["Age", "SibSp", "Parch", "Fare"],
    "nursery": None,
}

target_name: Dict[str, List[str]] = {
    "adult": ["class"],
    "adult_numeric": ["class"],
    "german": ["class"],
    "german_numeric": ["class"],
    "fico_heloc": ["RiskPerformance"],
    "fico_heloc_numeric": ["RiskPerformance"],
    "titanic": ["Survived"],
    "nursery": ["final_evaluation"],
}

continuous_cols_names_datasets: Dict[str, List[str]] = {
    "adult": ["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"],
    "adult_numeric": ["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"],
    "german": ["duration", "credit_amount", "age"],
    "german_numeric": ["duration", "credit_amount", "age"],
    "fico_heloc": ["ExternalRiskEstimate", "MSinceOldestTradeOpen", "MSinceMostRecentTradeOpen", "AverageMInFile",
                   "NumSatisfactoryTrades", "NumTrades60Ever2DerogPubRec",
                   "NumTrades90Ever2DerogPubRec", "PercentTradesNeverDelq", "MSinceMostRecentDelq", "NumTotalTrades",
                   "NumTradesOpeninLast12M", "PercentInstallTrades",
                   "MSinceMostRecentInqexcl7days", "NumInqLast6M", "NumInqLast6Mexcl7days",
                   "NetFractionRevolvingBurden", "NetFractionInstallBurden", "NumRevolvingTradesWBalance",
                   "NumInstallTradesWBalance", "NumBank2NatlTradesWHighUtilization", "PercentTradesWBalance"],
    "fico_heloc_numeric": ["ExternalRiskEstimate", "MSinceOldestTradeOpen", "MSinceMostRecentTradeOpen", "AverageMInFile",
                   "NumSatisfactoryTrades", "NumTrades60Ever2DerogPubRec",
                   "NumTrades90Ever2DerogPubRec", "PercentTradesNeverDelq", "MSinceMostRecentDelq", "NumTotalTrades",
                   "NumTradesOpeninLast12M", "PercentInstallTrades",
                   "MSinceMostRecentInqexcl7days", "NumInqLast6M", "NumInqLast6Mexcl7days",
                   "NetFractionRevolvingBurden", "NetFractionInstallBurden", "NumRevolvingTradesWBalance",
                   "NumInstallTradesWBalance", "NumBank2NatlTradesWHighUtilization", "PercentTradesWBalance"],
    "titanic": ["Age", "Fare"],
    "nursery": None,
}

random_state = 42
classifiers_names = {
    # "random_forest": lambda: RandomForestClassifier(),
    # "simpleNN": lambda: SimpleNN,
    "tab_pfn": lambda: TabPFNClassifier()
}

classifier_parametrs = {
    # "random_forest": {"max_depth": 5, "n_estimators": 10, "random_state": random_state},
    # "random_forest": {"max_depth": 20, "n_estimators": 50, "random_state": random_state},
    # "simpleNN": {"input_size": 10},
    "tab_pfn": {"n_estimators": 4, "ignore_pretraining_limits": False, "random_state": random_state}
}

rules_amount = 2
entries_amount = 3

log_folder_name = "experiments_log"
if not os.path.exists("experiments_log"):
    os.makedirs("experiments_log")

for classifier_name in classifiers_names:
    for name in experiment_name:
        log_file = os.path.join(log_folder_name, name + "_" + classifier_name + ".txt")
        logger = setup_logger(log_file, name)
        dataset = dataset_manager.dataset_object()
        log_with_custom_tag(logger, dataset.read_file(dataset_name=dataset_names[name], drop_cols_datasets=drop_cols_names_datasets.get(name, None)))
        log_with_custom_tag(logger, dataset.get_cols_name(categorical_cols_names_datasets[name], numeric_cols_names_datasets[name], continuous_cols_names_datasets[name]))
        log_with_custom_tag(logger, dataset.impute_missing())
        if categorical_cols_names_datasets[name]:
            dataset.init_encoders()
        log_with_custom_tag(logger, dataset.target_ordinal_encode())
        log_with_custom_tag(logger, dataset.split_dataset())

        idxs2e = set()
        while len(idxs2e) < entries_amount:
            idxs2e.add(random.randint(0, len(dataset.X_test)))

        if classifier_name == "simpleNN":
            if categorical_cols_names_datasets[name]:
                classifier_parametrs["simpleNN"]["input_size"] = dataset.onehot_encoder.transform(dataset.X_train).shape[1]
            else:
                # For datasets without categorical columns
                classifier_parametrs["simpleNN"]["input_size"] = dataset.X_train.shape[1]

        if continuous_cols_names_datasets[name]:
            custom_scaler = CustomScaler(
                scalers_dict=dataset.standard_scalers,
                continuous_col_names=dataset.continuous_col_names,
                continuous_cols = dataset.continuous_cols
            )
        else:
            custom_scaler = None

        clf = create_classifier(classifier_name, classifiers_names, classifier_parametrs)
        log_with_custom_tag(logger, f"Classifier: {classifier_name} Parameters: {classifier_parametrs[classifier_name]}")

        if continuous_cols_names_datasets[name]:
            clf.fit(
                dataset.onehot_encoder.transform(custom_scaler.transform(dataset.X_train)) if categorical_cols_names_datasets[
                    name] else custom_scaler.transform(dataset.X_train).to_numpy(),
                dataset.y_train)
        else:
            clf.fit(
                dataset.onehot_encoder.transform(dataset.X_train) if categorical_cols_names_datasets[
                    name] else dataset.X_train.to_numpy(),
                dataset.y_train)

        # clf.fit(dataset.onehot_encoder.transform(custom_scaler.transform(dataset.X_train)), dataset.y_train)
        # clf.fit(dataset.onehot_encoder.transform(dataset.X_train), dataset.y_train)

        predict_fn, predict_probab_fn, predict_fn_anchor = get_predict_functions(dataset, clf, custom_scaler)
        # predict_fn = lambda x: clf.predict(dataset.onehot_encoder.transform(custom_scaler.transform(x)))
        # predict_fn = lambda x: clf.predict(dataset.onehot_encoder.transform(x))
        # predict_probab_fn = lambda x: clf.predict_proba(dataset.onehot_encoder.transform(custom_scaler.transform(dataset.label_decode_features(x, dataset.categorical_cols))))
        # predict_probab_fn = lambda x: clf.predict_proba(dataset.onehot_encoder.transform(
        #     dataset.label_decode_features(x, dataset.categorical_cols)))
        # predict_fn_anchor = lambda x: clf.predict(dataset.onehot_encoder.transform(custom_scaler.transform(dataset.label_decode_features(x, dataset.categorical_cols))))

        bbox_lore = sklearn_classifier_bbox.sklearnBBox(
            clf,
            map=dataset.target_names,
            transformer=dataset.onehot_encoder if dataset.categorical_cols else None,
            custom_scaler=custom_scaler if dataset.continuous_cols else None
        )

        # bbox_lore = sklearn_classifier_bbox.sklearnBBox(clf, map=dataset.target_names, transformer=dataset.onehot_encoder, custom_scaler=custom_scaler)

        bbox_lore_old = sklearn_classifier_wrapper_custom(
            clf,
            # transformer=dataset.onehot_encoder if dataset.categorical_cols else None,
            # custom_scaler=custom_scaler if dataset.continuous_cols else None
        )

        # bbox_lore_old = sklearn_classifier_wrapper_custom(clf, transformer = dataset.onehot_encoder, custom_scaler = custom_scaler)

        log_with_custom_tag(logger, f"Train accuracy: {accuracy_score(dataset.y_train, predict_fn(dataset.X_train))}")
        log_with_custom_tag(logger, f"Test accuracy: {accuracy_score(dataset.y_test, predict_fn(dataset.X_test))}")

        log_with_custom_tag(logger, '\nClassification Report (Train):\n' + classification_report(dataset.y_train, predict_fn(dataset.X_train),
                                                                                 target_names=dataset.target_names))
        log_with_custom_tag(logger, '\nClassification Report (Test):\n' + classification_report(dataset.y_test, predict_fn(dataset.X_test),
                                                                                target_names=dataset.target_names))

        if categorical_cols_names_datasets[name]:
            # Anchor
            X_train_labeled = dataset.label_encode_features(dataset.X_train, dataset.categorical_cols)
            X_test_labeled = dataset.label_encode_features(dataset.X_test, dataset.categorical_cols)
            anchor_explainer = Anchor.anchor_object(X_train_labeled.to_numpy(), dataset.y_train, X_test_labeled.to_numpy(),
                                               dataset.y_test,
                                               dataset.feature_names, dataset.categorical_map, dataset.target_names)

            # Lore old
            if continuous_cols_names_datasets[name]:
                lore_explainer_old = LORE.lore_object_old(
                            dataset.onehot_encoder.transform(custom_scaler.transform(dataset.X_train)), dataset.y_train,
                            dataset.onehot_encoder.transform(custom_scaler.transform(dataset.X_test)), dataset.y_test,
                            custom_scaler.transform(dataset.raw),
                            config={"neigh_type": "geneticp", "size": 1000, "ocr": 0.1,
                                    "ngen": 10})
            else:
                lore_explainer_old = LORE.lore_object_old(
                            dataset.onehot_encoder.transform(dataset.X_train), dataset.y_train,
                            dataset.onehot_encoder.transform(dataset.X_test), dataset.y_test,
                            dataset.raw,
                            config={"neigh_type": "geneticp", "size": 1000, "ocr": 0.1,
                                    "ngen": 10})
            # Lux
            lux_explainer = LUX.lux_object(X_train_labeled, dataset.y_train,
                                               X_test_labeled, dataset.y_test, dataset.reverse_target_map)
        else:
            # Anchor
            anchor_explainer = Anchor.anchor_object(dataset.X_train.to_numpy(), dataset.y_train,
                                                    dataset.X_test.to_numpy(),
                                                    dataset.y_test,
                                                    dataset.feature_names, dataset.categorical_map,
                                                    dataset.target_names)
            # Lore old
            if continuous_cols_names_datasets[name]:
                lore_explainer_old = LORE.lore_object_old(
                    custom_scaler.transform(dataset.X_train), dataset.y_train,
                    custom_scaler.transform(dataset.X_test), dataset.y_test,
                    custom_scaler.transform(dataset.raw),
                    config={"neigh_type": "geneticp", "size": 1000, "ocr": 0.1,
                            "ngen": 10})
            else:
                lore_explainer_old = LORE.lore_object_old(
                    dataset.X_train, dataset.y_train,
                    dataset.X_test, dataset.y_test,
                    dataset.raw,
                    config={"neigh_type": "geneticp", "size": 1000, "ocr": 0.1,
                            "ngen": 10})
            # Lux
            lux_explainer = LUX.lux_object(dataset.X_train, dataset.y_train,
                                               dataset.X_test, dataset.y_test, dataset.reverse_target_map)

        log_with_custom_tag(logger, anchor_explainer.init_explainer(predict_fn_anchor, ohe=False))
        log_with_custom_tag(logger, lore_explainer_old.init_explainer(bbox_lore_old, numeric_cols_names_datasets[name]))
        log_with_custom_tag(logger, lux_explainer.init_explainer(predict_probab_fn))

        # Lore
        lore_explainer = LORE_2.lore_object(str(dataset_names[name]+".csv"), dataset.numeric_col_names, dataset.categorical_col_names, dataset.X_test, dataset.y_test, dataset.raw, target_name=target_name[name])
        lore_explainer.init_explainer(bbox_lore)


        # lore_explainer_old = LORE.lore_object_old(dataset.X_train, dataset.y_train,
        #                                       dataset.X_test, dataset.y_test,
        #                                       dataset.raw, config = {"neigh_type": "geneticp", "size": 1000, "ocr": 0.1, "ngen": 10})

        for idx in idxs2e:
            anchor_explainer.get_instance(idx)
            lore_explainer_old.get_instance(idx)
            lore_explainer.get_instance(idx)
            lux_explainer.get_instance(idx)
            #explain
            log_with_custom_tag(logger, f"Explaining instance: {dataset.X_test.iloc[idx].name} outcome: {dataset.target_names[dataset.y_test.iloc[idx]]}", 'ENTRY')
            # print(f"Explaining instance: {dataset.data_labeled.index[(dataset.data_labeled == dataset.X_test[idx]).all(axis=1)].tolist()[0]} outcome: {dataset.target_names[dataset.y_test.iloc[idx]]}")
            # Anchor
            explanations = anchor_explainer.explain(rules_amount, beam_size=8, verbose=False)
            for rule in explanations:
                log_with_custom_tag(logger, anchor_explainer.print_explanation(rule), 'RULE')
            # Lore old
            explanations = lore_explainer_old.explain(rules_amount)
            for rule in explanations:
                log_with_custom_tag(logger, lore_explainer_old.print_explanation(rule), 'RULE')
            # Lore
            explanations = lore_explainer.explain(rules_amount)
            for rule in explanations:
                log_with_custom_tag(logger, lore_explainer.print_explanation(rule), 'RULE')
            if categorical_cols_names_datasets[name]:
                explanations = lux_explainer.explain(X_train_labeled, dataset.y_train,
                                                     [col in dataset.categorical_col_names for col in
                                                      dataset.data.columns], 1)
            else:
                explanations = lux_explainer.explain(dataset.X_train, dataset.y_train,
                                                     [not col in dataset.numeric_col_names for col in dataset.data.columns], 1)

            for rule in explanations:
                log_with_custom_tag(logger, lux_explainer.print_explanation(rule), 'RULE')