import os
import random
from typing import Dict, List
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve
from lore_updated.bbox import sklearn_classifier_bbox
import numpy as np
from collections import defaultdict

import dataset_manager
import Anchor_wrapper, LORE_SA_wrapper, LORE_wrapper, EXPLAN_wrapper#, LUX
from logger import log_info, setup_logger, log_entry, log_rule
from Classifier import create_classifier, get_predict_functions, sklearn_classifier_wrapper_custom, \
    get_balanced_correct_indexes, save_model
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

def set_best_f1_threshold(model, X_val, y_val):
    # probs on validation
    p = model.predict_proba(X_val)[:, 1]
    prec, rec, thr = precision_recall_curve(y_val, p)

    # F1 for each threshold (skip the last prec/rec point)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    i = np.argmax(f1)
    best_thr = float(thr[i])
    best_f1 = float(f1[i])

    model.decision_threshold = best_thr
    print(f"[Threshold tuning] best F1={best_f1:.4f} at threshold={best_thr:.4f}")
    return best_thr, best_f1

experiment_name = [
    "adult",
    # "adult_numeric",
    # "german",
    # "german_numeric",
    # "fico_heloc",
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
    "adult": [],
    "german": [],
    "fico_heloc": []
}

# instance_2e: Dict[str, List[int]] = {
#     "adult": [6694, 31792, 24956, 19575, 22095, 13782, 24645, 15081, 13126, 30027, 13849, 21427, 4747, 28678,
#               24020, 13410, 18105, 26589, 21720, 14118, 17232, 21990, 3288, 4010, 26551, 4528, 21544, 15654,
#               28742, 4135, 333, 30980, 14060, 14291, 25929, 16095, 30193, 17563, 26738, 23499, 23094, 16836,
#               20719, 31321, 27105, 13764, 4718, 23173, 1240, 29960, 2413, 10680, 15043, 498, 6092, 2479, 3006,
#               3021, 3694, 2446, 590, 22390, 2141, 465, 374, 1772, 29183, 2368, 1991, 1645, 2061, 3086, 26573,
#               31205, 2172, 17726, 608, 2297, 506, 5494, 31536, 9863, 1553, 2224, 31440, 30982, 18670, 24772, 10994,
#               3193, 17755, 1752, 6738, 2821, 25008, 663, 4328, 14162, 20005, 22119]
#     ,
#     "german": [918, 29, 368, 651, 624, 471, 320, 814, 235, 374, 588, 528, 170, 835, 862, 998, 357, 257, 759, 788,
#                596, 983, 4, 337, 321, 914, 59, 634, 924, 931, 475, 87, 812, 378, 10, 878, 853, 809, 131, 548, 922,
#                444, 491, 212, 884, 631, 192, 736, 301, 714, 391, 26, 584, 377, 844, 297, 798, 870, 24, 60, 699, 280,
#                694, 810, 506, 128, 875, 153, 133, 546, 554, 514, 738, 681, 590, 269, 244, 524, 635, 220, 829, 48, 394,
#                659, 511, 50, 348, 897, 231, 365, 715, 33, 989, 151, 134, 859, 673, 768, 806, 709]
#     ,
#     "fico_heloc": [1117, 2480, 5583, 5124, 5934, 1284, 3205, 3063, 2164, 9931, 9102, 7393, 272, 8173, 5455, 7416, 1511,
#                    8801, 2833, 8613, 5900, 7110, 1553, 2013, 516, 6515, 6944, 7752, 6144, 2597, 2239, 8046, 4508, 5280,
#                    563, 1661, 1765, 2689, 9009, 9079, 9479, 2303, 678, 8071, 5025, 1837, 9432, 8541, 6641, 4127, 9359,
#                    2359, 9279, 691, 3714, 8078, 8652, 7877, 2098, 2370, 4044, 3967, 8625, 9325, 6614, 7736, 2203, 5870,
#                    419, 9751, 10322, 8021, 1078, 5546, 4737, 6279, 4549, 3304, 1052, 9912, 9034, 9661, 10333, 2485, 3086,
#                    7957, 9345, 6718, 6896, 5082, 7423, 987, 2975, 3921, 2877, 3504, 3291, 3957, 1526, 1507]
# }

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
    "random_forest": lambda: RandomForestClassifier(),
    # "simpleNN": lambda: SimpleNN,
}

classifier_parametrs = {
    "random_forest": {
        "adult": {'max_depth': 20, 'n_estimators': 50, 'random_state': random_state},
        "german": {"max_depth": 8, "n_estimators": 50, "random_state": random_state},# "class_weight": 'balanced'},
        "fico_heloc": {'max_depth': 20, 'n_estimators': 50, 'random_state': random_state}
        },
    "simpleNN": {
        "adult": {"input_size": 10},
        "german": {"input_size": 10},
        "fico_heloc": {"input_size": 10},
    }
}
explainers = ["ANCHOR", "LORE", "LORE_SA", "EXPLAN"]

rules_amount = 5
entries_amount = 2

log_folder_name = "experiments_log"
if not os.path.exists("experiments_log"):
    os.makedirs("experiments_log")

combined_inst2e = defaultdict(list)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

for classifier_name in classifiers_names:
    for name in experiment_name:

        #log files names
        log_file_info = os.path.join(log_folder_name, name + "_" + classifier_name + "_info.txt")
        log_file_entries = os.path.join(log_folder_name, name + "_" + classifier_name + "_entries.txt")
        log_file_rules = os.path.join(log_folder_name, name + "_" + classifier_name + "_rules.txt")
        log_file_train = os.path.join(log_folder_name,
                                      f"{name}_{classifier_name}_train_df_{datetime.now().strftime('%M-%H_%d-%m-%Y')}.csv")
        log_file_test = os.path.join(log_folder_name,
                                     f"{name}_{classifier_name}_test_df_{datetime.now().strftime('%M-%H_%d-%m-%Y')}.csv")

        # loggers
        logger_info = setup_logger(log_file_info, name + "_info")
        logger_entries = setup_logger(log_file_entries, name + "_entries")
        logger_rules = setup_logger(log_file_rules, name + "_rules")

        # dataset manager
        dataset = dataset_manager.dataset_object()

        # read data and drop cols
        log_info(logger_info, dataset.read_file(dataset_name=dataset_names[name], drop_cols_datasets=drop_cols_names_datasets.get(name, None)))
        log_info(logger_info, f"Dropped columns:{drop_cols_names_datasets.get(name, None)}")

        # count nans amount in raw data
        log_info(logger_info, nans_count_report(dataset.raw, "Raw"))

        # get cols types
        log_info(logger_info, dataset.get_cols_name(categorical_cols_names_datasets[name],
                                                    numeric_cols_names_datasets[name],
                                                    continuous_cols_names_datasets[name]))

        # impute missing vals
        log_info(logger_info, dataset.impute_missing())

        # count nans amount in raw data
        log_info(logger_info, nans_count_report(dataset.raw, "Raw"))

        if categorical_cols_names_datasets[name]:
            # count nans amount in raw data
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
        if name == "german": #or name == "adult":
            log_info(logger_info, dataset.smote_oversample(custom_scaler, random_state=random_state))

        df_test = dataset.X_test.copy()
        df_test['class'] = dataset.y_test.map(dataset.reverse_target_map)
        df_train = dataset.X_train.copy()
        df_train['class'] = dataset.y_train.map(dataset.reverse_target_map)
        df_test.to_csv(log_file_test)
        df_train.to_csv(log_file_train)

        if classifier_name == "simpleNN":
            if categorical_cols_names_datasets[name]:
                classifier_parametrs["simpleNN"][name]["input_size"] = dataset.onehot_encoder.transform(dataset.X_train).shape[1]
            else:
                # for datasets without categorical columns
                classifier_parametrs["simpleNN"][name]["input_size"] = dataset.X_train.shape[1]

        clf, clf_loaded = create_classifier(name, classifier_name, classifiers_names, classifier_parametrs, device)

        if not clf_loaded:
            log_info(logger_info,
                     f"Classifier: {classifier_name} Parameters: {classifier_parametrs[classifier_name][name]}")
            if continuous_cols_names_datasets[name]:
                X_train_proc = custom_scaler.transform(dataset.X_train)
                X_val_proc = custom_scaler.transform(dataset.X_val)

            if categorical_cols_names_datasets[name]:
                X_train_proc = dataset.onehot_encoder.transform(X_train_proc)
                X_val_proc = dataset.onehot_encoder.transform(X_val_proc)

            X_train_proc = X_train_proc.to_numpy() if hasattr(X_train_proc, "to_numpy") else X_train_proc
            X_val_proc = X_val_proc.to_numpy() if hasattr(X_val_proc, "to_numpy") else X_val_proc

            clf.fit(X_train_proc, dataset.y_train)
            if classifier_name == "simpleNN":
                best_thr, best_f1 = set_best_f1_threshold(clf, X_val_proc, dataset.y_val)
                log_info(logger_info, f"Best f1 threshold: {best_thr}")
                log_info(logger_info, save_model(clf, name, classifier_name, best_thr))
            else:
                log_info(logger_info, save_model(clf, name, classifier_name))
        else:
            log_info(logger_info,
                     f"Loaded classifier: {classifier_name} for this month")


        predict_fn, predict_probab_fn, predict_fn_anchor = get_predict_functions(dataset, clf, custom_scaler)
        log_info(logger_info,
                            print_class_distribution("WholeData", dataset.target))
        log_info(logger_info,
                            print_class_distribution("Train", dataset.y_train.map(dataset.reverse_target_map)))
        log_info(logger_info, print_class_distribution("Test", dataset.y_test.map(dataset.reverse_target_map)))

        bbox_lore = sklearn_classifier_bbox.sklearnBBox(
            clf,
            map=dataset.target_names,
            transformer=dataset.onehot_encoder if dataset.categorical_cols else None,
            custom_scaler=custom_scaler if dataset.continuous_cols else None
        )

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
        else:
            # Anchor
            anchor_explainer = Anchor.anchor_object(dataset.X_train.to_numpy(), dataset.y_train,
                                                    dataset.X_test.to_numpy(),
                                                    dataset.y_test,
                                                    dataset.feature_names, dataset.categorical_map,
                                                    dataset.target_names)

        log_info(logger_info, anchor_explainer.init_explainer(predict_fn_anchor, ohe=False, beam_size=10, treshold=0.9, tau = 0.10, delta=0.05 ))

        # Lore
        lore_explainer = LORE_wrapper.lore_object(str(dataset_names[name]), dataset.raw,
                                                  dataset.label_encode_features(dataset.data, dataset.categorical_cols,
                                                                                dataset.categorical_col_names),
                                                  dataset.label_encoders,
                                                  dataset.label_encode_features(dataset.X_train,
                                                                                dataset.categorical_cols,
                                                                                dataset.categorical_col_names),
                                                  dataset.continuous_col_names, dataset.numeric_col_names,
                                                  dataset.categorical_col_names,
                                                  dataset.target, target_name=target_name[name][0])
        log_info(logger_info, lore_explainer.init_explainer(dataset.categorical_cols, dataset.target_encoder, iter_limit = 10))

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


        if name not in combined_inst2e:
            combined_inst2e[name] = get_balanced_correct_indexes(pred_funct=predict_fn, X_test=dataset.X_test, y_test=dataset.y_test,
                                              n=entries_amount, instance_2e=instance_2e[name])
        explanations = {
            "ANCHOR": [],
            "LORE": [],
            "LORE_SA": [],
            "EXPLAN": []
        }

        # ANCHOR
        for idx in combined_inst2e[name]:
            explainer_name = "ANCHOR"
            instance = dataset.X_test.iloc[idx]
            outcome = dataset.target_names[dataset.y_test.iloc[idx]]
            log_entry(logger_entries, instance, instance.name, outcome,
                      dataset.reverse_target_map[predict_fn(np.array(instance).reshape(1, -1))[0]])
            print(f"{explainer_name} Explaining instance: {instance.name} outcome: {outcome}")
            anchor_explainer.get_instance(idx)
            explanations[explainer_name].append(
                anchor_explainer.explain(rules_amount, instance_name=instance.name, verbose=False)
            )
        # LORE
        for idx in combined_inst2e[name]:
            explainer_name = "LORE"
            instance = dataset.X_test.iloc[idx]
            instance_from_encoded = dataset.label_encode_features(dataset.X_test, dataset.categorical_cols,
                                                                  dataset.categorical_col_names).iloc[idx]
            outcome = dataset.target_names[dataset.y_test.iloc[idx]]
            print(f"{explainer_name} Explaining instance: {instance.name} outcome: {outcome}")
            explanations[explainer_name].append(
                lore_explainer.explain(rules_amount, instance_from_encoded, predict_fn_anchor, instance_name=instance.name)
            )

        # LORE_SA
        for idx in combined_inst2e[name]:
            explainer_name = "LORE_SA"
            lore_sa_explainer.get_instance(idx)
            instance = dataset.X_test.iloc[idx]
            outcome = dataset.target_names[dataset.y_test.iloc[idx]]
            print(f"{explainer_name} Explaining instance: {instance.name} outcome: {outcome}")
            explanations[explainer_name].append(
                lore_sa_explainer.explain(rules_amount, instance_name=instance.name)
            )
        # EXPLAN
        for idx in combined_inst2e[name]:
            explainer_name = "EXPLAN"
            instance = dataset.X_test.iloc[idx]
            outcome = dataset.target_names[dataset.y_test.iloc[idx]]
            print(f"{explainer_name} Explaining instance: {instance.name} outcome: {outcome}")
            explanations[explainer_name].append(
                explan_explainer.explain(rules_amount, idx, predict_fn_anchor, instance_name=instance.name)
            )
        for explainer_results in explanations.values():  # list of explanations
            for explanation in explainer_results:
                for rule in explanation:
                    print(rule.get_rule())
                    print(rule.raw)
                    log_rule(logger_rules, rule, rule.instance_name, rule.evaluate_on(df_train))