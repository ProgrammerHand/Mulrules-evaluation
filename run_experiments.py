import os
import random
from typing import Dict, List
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score, classification_report
from lore_sa.bbox import sklearn_classifier_bbox
import numpy as np
from collections import defaultdict

import dataset_manager
import Anchor, LUX, LORE_SA, LORE_wrapper, EXPLAN_wrapper
from SimpleNN import SimpleNN
from logger import log_with_custom_tag, log_info, setup_logger, log_entry, log_rule
from Classifier import create_classifier, get_predict_functions, sklearn_classifier_wrapper_custom, get_balanced_correct_indexes
from Scaler import CustomScaler
import torch

def print_class_distribution(name, y_series):
    counts = y_series.value_counts()
    percentages = y_series.value_counts(normalize=True) * 100
    distribution = pd.DataFrame({
        'Count': counts,
        'Percentage': percentages.round(2)
    })
    return f"\n{name} class distribution:\n{distribution}"

def nans_count_report(df, name):
    nans = df.isna().sum()
    return f"\n{name} NaN counts per column:\n{nans[nans > 0]}"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # GPU
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

experiment_name = [
    "adult",
    # "adult_numeric",
    "german",
    # "german_numeric",
    "fico_heloc",
    # "fico_heloc_numeric",
    # "titanic",
    # "nursery"
]
dataset_names: Dict[str, str] = {
    "adult": "adult",
    "adult_numeric": "adult",
    "german": "german_rename_vals",
    "german_numeric": "german",
    "fico_heloc": "fico",
    "fico_heloc_numeric": "fico",
    "titanic": "titanic",
    "nursery": "nursery"
}

instance_2e: Dict[str, List[int]] = {
    "adult": [5988, 18073, 652, 2711, 12758, 5759, 20, 1808, 12191, 3374],
    "german": [844, 480, 11, 917, 523, 296, 301, 915, 941, 216],
    "fico_heloc": [1758, 2674, 8352, 9583, 8530, 9264, 781, 5633, 6983, 779]
}

drop_cols_names_datasets: Dict[str, List[str] | None] = {
    "adult" : ["fnlwgt"],
    "adult_numeric": ["workclass", "education", "marital.status", "occupation" ,"relationship", "race", "sex", "native.country"],
    "german": [],
    "german_numeric": ["checking_status","credit_history","purpose","savings_status","employment","personal_status","other_parties","property_magnitude","other_payment_plans","housing","job","own_telephone","foreign_worker"],
    "fico_heloc": [],
    "fico_heloc_numeric": ["MaxDelq2PublicRecLast12M", "MaxDelqEver"],
    "titanic": ["PassengerId", "Name"],
    "nursery": [],
    }

categorical_cols_names_datasets: Dict[str, List[str] | None] = {
    "adult": ["workclass", "education", "education.num", "marital.status", "occupation" ,"relationship", "race", "sex", "native.country"],
    "adult_numeric": [],
    "german": ["checking_status","credit_history","purpose","savings_status","employment","personal_status","other_parties","property_magnitude","other_payment_plans","housing","job","own_telephone","foreign_worker"],
    "german_numeric": [],
    "fico_heloc": ["MaxDelq2PublicRecLast12M", "MaxDelqEver"],
    "fico_heloc_numeric": [],
    "titanic": ["Pclass", "Sex", "Embarked", "Cabin", "Ticket"],
    "nursery": ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health"],
}

numeric_cols_names_datasets: Dict[str, List[str]] = {
    "adult": ["age", "capital.gain", "capital.loss", "hours.per.week"],
    "adult_numeric":  ["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"],
    "german": ["duration", "credit_amount", "installment_commitment", "residence_since", "age", "existing_credits", "num_dependents"],
    "german_numeric": ["duration", "credit_amount", "installment_commitment", "residence_since", "age", "existing_credits", "num_dependents"],
    "fico_heloc": [
        "ExternalRiskEstimate", "MSinceOldestTradeOpen", "MSinceMostRecentTradeOpen", "AverageMInFile",
        "NumSatisfactoryTrades", "NumTrades60Ever2DerogPubRec", "NumTrades90Ever2DerogPubRec",
        "PercentTradesNeverDelq", "MSinceMostRecentDelq",
        "NumTotalTrades", "NumTradesOpeninLast12M", "PercentInstallTrades", "MSinceMostRecentInqexcl7days",
        "NumInqLast6M", "NumInqLast6Mexcl7days", "NetFractionRevolvingBurden", "NetFractionInstallBurden",
        "NumRevolvingTradesWBalance", "NumInstallTradesWBalance", "NumBank2NatlTradesWHighUtilization",
        "PercentTradesWBalance"
    ],
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
    "fico_heloc": ["class"],
    "fico_heloc_numeric": ["class"],
    "titanic": ["Survived"],
    "nursery": ["final_evaluation"],
}

continuous_cols_names_datasets: Dict[str, List[str]] = {
    "adult": ["age", "capital.gain", "capital.loss", "hours.per.week"],
    "adult_numeric": ["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"],
    "german": ["duration", "credit_amount", "age"],
    "german_numeric": ["duration", "credit_amount", "age"],
    "fico_heloc": [
        "ExternalRiskEstimate", "MSinceOldestTradeOpen", "MSinceMostRecentTradeOpen", "AverageMInFile",
        "PercentTradesNeverDelq", "MSinceMostRecentDelq", "PercentInstallTrades",
        "MSinceMostRecentInqexcl7days", "NetFractionRevolvingBurden", "NetFractionInstallBurden",
        "PercentTradesWBalance"
    ],
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
    "simpleNN": lambda: SimpleNN,
    # "tab_pfn": lambda: TabPFNClassifier()
}

classifier_parametrs = {
    "random_forest": {
        "adult": {'max_depth': 20, 'n_estimators': 50, 'random_state': random_state},
        "german": {"max_depth": 8, "n_estimators": 50, "random_state": random_state},# "class_weight": 'balanced'},
        "fico_heloc": {'max_depth': 20, 'n_estimators': 50, 'random_state': random_state}
        # "adult": {'max_depth': 15, 'n_estimators': 100, "min_samples_split":10, "min_samples_leaf": 4, 'random_state': random_state}
        # "adult": {'max_depth': 20, 'n_estimators': 50, 'random_state': random_state, 'min_samples_split': 20, 'min_samples_leaf': 2},
        # "adult_numeric": {'max_depth': 18, 'n_estimators': 81, 'random_state': 42, 'min_samples_split': 3, 'min_samples_leaf': 1}
        },
    "simpleNN": {
        "adult": {"input_size": 10},
        "german": {"input_size": 10},
        "fico_heloc": {"input_size": 10},
    }
    # "tab_pfn": {"n_estimators": 4, "ignore_pretraining_limits": False, "random_state": random_state}
}

rules_amount = 5
entries_amount = 0

log_folder_name = "experiments_log"
if not os.path.exists("experiments_log"):
    os.makedirs("experiments_log")

combined_inst2e = defaultdict(list)

for classifier_name in classifiers_names:
    for name in experiment_name:
        log_file_info = os.path.join(log_folder_name, name + "_" + classifier_name + "_info.txt")
        log_file_entries = os.path.join(log_folder_name, name + "_" + classifier_name + "_entries.txt")
        log_file_rules = os.path.join(log_folder_name, name + "_" + classifier_name + "_rules.txt")
        logger_info = setup_logger(log_file_info, name + "_info")
        logger_entries = setup_logger(log_file_entries, name + "_entries")
        logger_rules = setup_logger(log_file_rules, name + "_rules")
        # log_file = os.path.join(log_folder_name, name + "_" + classifier_name + ".txt")
        # logger = setup_logger(log_file, name)
        dataset = dataset_manager.dataset_object()
        log_info(logger_info, dataset.read_file(dataset_name=dataset_names[name], drop_cols_datasets=drop_cols_names_datasets.get(name, None)))
        log_info(logger_info,f"Dropped columns:{drop_cols_names_datasets.get(name, None)}")
        log_info(logger_info, nans_count_report(dataset.raw, "Raw"))
        log_info(logger_info, dataset.get_cols_name(categorical_cols_names_datasets[name],
                                                    numeric_cols_names_datasets[name],
                                                    continuous_cols_names_datasets[name]))
        log_info(logger_info, dataset.impute_missing())
        log_info(logger_info, nans_count_report(dataset.raw, "Raw"))

        if categorical_cols_names_datasets[name]:
            dataset.init_encoders()
        log_info(logger_info, dataset.target_ordinal_encode())
        log_info(logger_info, dataset.split_dataset(random_state=random_state))

        if continuous_cols_names_datasets[name]:
            custom_scaler = CustomScaler(
                scalers_dict=dataset.standard_scalers,
                continuous_col_names=dataset.continuous_col_names,
                continuous_cols = dataset.continuous_cols
            )
        else:
            custom_scaler = None
        if name == "german":
            log_info(logger_info, dataset.smote_oversample(custom_scaler, random_state=random_state))

        df_test = dataset.X_test.copy()
        df_test['class'] = dataset.y_test.map(dataset.reverse_target_map)
        df_train = dataset.X_train.copy()
        df_train['class'] = dataset.y_train.map(dataset.reverse_target_map)


        if classifier_name == "simpleNN":
            if categorical_cols_names_datasets[name]:
                classifier_parametrs["simpleNN"][name]["input_size"] = dataset.onehot_encoder.transform(dataset.X_train).shape[1]
            else:
                # for datasets without categorical columns
                classifier_parametrs["simpleNN"][name]["input_size"] = dataset.X_train.shape[1]

        clf = create_classifier(name, classifier_name, classifiers_names, classifier_parametrs)
        log_info(logger_info, f"Classifier: {classifier_name} Parameters: {classifier_parametrs[classifier_name][name]}")

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
        log_info(logger_info,
                            print_class_distribution("Test", dataset.target))
        log_info(logger_info,
                            print_class_distribution("Train", dataset.y_train.map(dataset.reverse_target_map)))
        log_info(logger_info, print_class_distribution("Test", dataset.y_test.map(dataset.reverse_target_map)))
        # tempX = dataset.X_test.reset_index(drop=True)
        # temp = dataset.y_test.reset_index(drop=True)

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

        log_info(logger_info, f"Train accuracy: {accuracy_score(dataset.y_train, predict_fn(dataset.X_train))}")
        log_info(logger_info, f"Test accuracy: {accuracy_score(dataset.y_test, predict_fn(dataset.X_test))}")

        log_info(logger_info, '\nClassification Report (Train):\n' + classification_report(dataset.y_train, predict_fn(dataset.X_train),
                                                                                 target_names=dataset.target_names))
        log_info(logger_info, '\nClassification Report (Test):\n' + classification_report(dataset.y_test, predict_fn(dataset.X_test),
                                                                                target_names=dataset.target_names))

        if categorical_cols_names_datasets[name]:
            # Anchor
            X_train_labeled = dataset.label_encode_features(dataset.X_train, dataset.categorical_cols, dataset.categorical_col_names)
            X_test_labeled = dataset.label_encode_features(dataset.X_test, dataset.categorical_cols, dataset.categorical_col_names)
            anchor_explainer = Anchor.anchor_object(X_train_labeled.to_numpy(), dataset.y_train,
                                                    X_test_labeled.to_numpy(), dataset.y_test, dataset.feature_names,
                                                    dataset.categorical_map, dataset.target_names,
                                                    dataset.continuous_col_names, dataset.numeric_col_names)

            # Lore old
            # if continuous_cols_names_datasets[name]:
            #     lore_explainer_old = LORE.lore_object_old(
            #                 dataset.onehot_encoder.transform(custom_scaler.transform(dataset.X_train)), dataset.y_train,
            #                 dataset.onehot_encoder.transform(custom_scaler.transform(dataset.X_test)), dataset.y_test,
            #                 custom_scaler.transform(dataset.raw),
            #                 config={"neigh_type": "geneticp", "size": 1000, "ocr": 0.1,
            #                         "ngen": 10})
            # else:
            #     lore_explainer_old = LORE.lore_object_old(
            #                 dataset.onehot_encoder.transform(dataset.X_train), dataset.y_train,
            #                 dataset.onehot_encoder.transform(dataset.X_test), dataset.y_test,
            #                 dataset.raw,
            #                 config={"neigh_type": "geneticp", "size": 1000, "ocr": 0.1,
            #                         "ngen": 10})

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
            # if continuous_cols_names_datasets[name]:
            #     lore_explainer_old = LORE.lore_object_old(
            #         custom_scaler.transform(dataset.X_train), dataset.y_train,
            #         custom_scaler.transform(dataset.X_test), dataset.y_test,
            #         custom_scaler.transform(dataset.raw),
            #         config={"neigh_type": "geneticp", "size": 1000, "ocr": 0.1,
            #                 "ngen": 10})
            # else:
            #     lore_explainer_old = LORE.lore_object_old(
            #         dataset.X_train, dataset.y_train,
            #         dataset.X_test, dataset.y_test,
            #         dataset.raw,
            #         config={"neigh_type": "geneticp", "size": 1000, "ocr": 0.1,
            #                 "ngen": 10})
            # Lux
            lux_explainer = LUX.lux_object(dataset.X_train, dataset.y_train,
                                               dataset.X_test, dataset.y_test, dataset.reverse_target_map)

        log_info(logger_info, anchor_explainer.init_explainer(predict_fn_anchor, ohe=False, beam_size=8))
        # log_with_custom_tag(logger, lore_explainer_old.init_explainer(bbox_lore_old, numeric_cols_names_datasets[name]))
        # log_with_custom_tag(logger, lux_explainer.init_explainer(predict_probab_fn))

        # Lore
        lore_explainer = LORE_wrapper.lore_object(str(dataset_names[name]), dataset.raw, dataset.label_encode_features(dataset.data, dataset.categorical_cols, dataset.categorical_col_names), dataset.label_encoders, dataset.label_encode_features(dataset.X_train, dataset.categorical_cols, dataset.categorical_col_names), dataset.continuous_col_names, dataset.numeric_col_names, dataset.categorical_col_names, dataset.target, target_name=target_name[name][0])
        log_info(logger_info, lore_explainer.init_explainer(dataset.categorical_cols, dataset.target_encoder))

        # Lore_sa
        lore_sa_explainer = LORE_SA.lore_sa_object(str(dataset_names[name] + ".csv"), dataset.numeric_col_names, dataset.categorical_col_names, dataset.X_test, dataset.y_test, dataset.raw, train_df= df_train, target_name=target_name[name])
        lore_sa_explainer.init_explainer(bbox_lore)

        # EXPLAN
        explan_explainer = EXPLAN_wrapper.explan_object(str(dataset_names[name]), df_train,
                                                        dataset.label_encode_features(dataset.X_train, dataset.categorical_cols,
                                                                                dataset.categorical_col_names),
                                                        dataset.label_encoders, dataset.label_encode_features(dataset.X_test,
                                                                                                        dataset.categorical_cols,
                                                                                                        dataset.categorical_col_names),
                                                        dataset.continuous_col_names, dataset.numeric_col_names, dataset.categorical_col_names,
                                                        dataset.y_train, target_name=target_name[name][0])
        # log_with_custom_tag(logger, explan_explainer.init_explainer(dataset.categorical_cols, dataset.target_encoder, N_samples = 7000, tau = 100))
        log_info(logger_info, explan_explainer.init_explainer(dataset.categorical_cols, dataset.target_encoder))


        # lore_explainer_old = LORE.lore_object_old(dataset.X_train, dataset.y_train,
        #                                       dataset.X_test, dataset.y_test,
        #                                       dataset.raw, config = {"neigh_type": "geneticp", "size": 1000, "ocr": 0.1, "ngen": 10})
        if name not in combined_inst2e:
            combined_inst2e[name] = get_balanced_correct_indexes(pred_funct=predict_fn, X_test=dataset.X_test, y_test=dataset.y_test,
                                              n=entries_amount, instance_2e=instance_2e[name])

        for idx in combined_inst2e[name]:
            anchor_explainer.get_instance(idx)
            # lore_explainer_old.get_instance(idx)
            lore_sa_explainer.get_instance(idx)
            # lux_explainer.get_instance(idx)
            #explain
            instance = dataset.X_test.iloc[idx]
            instance_from_encoded = dataset.label_encode_features(dataset.X_test, dataset.categorical_cols,
                                                                  dataset.categorical_col_names).iloc[idx]
            outcome = dataset.target_names[dataset.y_test.iloc[idx]]
            # log_with_custom_tag(logger,
            #                     f"Explaining instance:\n {instance} original_outcome: {outcome}, predicted_outcome {dataset.reverse_target_map[predict_fn(np.array(instance).reshape(1, -1))[0]]}",
            #                     'ENTRY')
            log_entry(logger_entries, instance, instance.name, outcome, dataset.reverse_target_map[predict_fn(np.array(instance).reshape(1, -1))[0]])
            explanations = {}
            # print(f"Explaining instance: {dataset.data_labeled.index[(dataset.data_labeled == dataset.X_test[idx]).all(axis=1)].tolist()[0]} outcome: {dataset.target_names[dataset.y_test.iloc[idx]]}")
            # ANCHOR
            print("ANCHOR")
            explanations["ANCHOR"] = anchor_explainer.explain(rules_amount, verbose=False)
            # for rule in explanations:
            #     log_with_custom_tag(logger, anchor_explainer.print_explanation(rule), 'RULE')
            # Lore old
            # explanations = lore_explainer_old.explain(rules_amount)
            # for rule in explanations:
            #     log_with_custom_tag(logger, lore_explainer_old.print_explanation(rule), 'RULE')
            # LORE
            # explanations, infos = lore_explainer.explain(rules_amount, idx, predict_fn_anchor)
            print("LORE")
            explanations["LORE"] = lore_explainer.explain(rules_amount, instance_from_encoded, predict_fn_anchor)
            # for rule, info in zip(explanations, infos):
            #     log_with_custom_tag(logger, lore_explainer.print_explanation(rule, info), 'RULE')
            # LORE_SA
            print("LORE_SA")
            explanations["LORE_SA"] = lore_sa_explainer.explain(rules_amount)
            # for rule in explanations:
            #     log_with_custom_tag(logger, lore_sa_explainer.print_explanation(rule), 'RULE')
            # EXPLAN
            print("EXPLAN")
            explanations["EXPLAN"] = explan_explainer.explain(rules_amount, idx, predict_fn_anchor)
            # explanations, infos = explan_explainer.explain(rules_amount, idx, predict_fn_anchor)
            # for rule, info in zip(explanations, infos):
            #     log_with_custom_tag(logger, explan_explainer.print_explanation(rule, info), 'RULE')
            for explainer in explanations.values():
                for rule in explainer:
                    print(rule.get_rule())
                    print(rule.raw)
                    # log_rule(log_file_rules, rule.get_rule() + " " + rule.evaluate_on(df_train), "RULE")
                    log_rule(logger_rules, rule, instance.name, rule.evaluate_on(df_train))

            # #LUX
            # if categorical_cols_names_datasets[name]:
            #     explanations = lux_explainer.explain(X_train_labeled, dataset.y_train,
            #                                          [col in dataset.categorical_col_names for col in
            #                                           dataset.data.columns], 1)
            # else:
            #     explanations = lux_explainer.explain(dataset.X_train, dataset.y_train,
            #                                          [not col in dataset.numeric_col_names for col in dataset.data.columns], 1)
            #
            # for rule in explanations:
            #     log_with_custom_tag(logger, lux_explainer.print_explanation(rule), 'RULE')