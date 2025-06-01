from typing import List, Dict

from lore_sa.dataset import TabularDataset
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from lore_sa.bbox import sklearn_classifier_bbox
from itertools import product
import dataset_manager
import optuna
from Scaler import CustomScaler
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_val_score

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
    "adult_numeric": [],
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
dataset_name = "german"

from sklearn.metrics import precision_score, make_scorer

def precision_class_1(y_true, y_pred):
    return precision_score(y_true, y_pred, pos_label=1, average='binary')  # change `1` if needed
def objective(trial):
    # hyperparameters
    # max_depth = trial.suggest_int("max_depth", 3, 30)
    max_depth = trial.suggest_categorical("max_depth", [9])
    # max_depth = trial.suggest_int("max_depth", 20, 30)
    # n_estimators = trial.suggest_int("n_estimators", 10, 200)
    n_estimators = trial.suggest_categorical("n_estimators", [49])
    # n_estimators = trial.suggest_int("n_estimators", 30, 80)
    random_state = trial.suggest_categorical("random_state", [42])
    min_samples_split = trial.suggest_int('min_samples_split', 2, 100)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 100)

    # dataset loading & preprocessing
    dataset = dataset_manager.dataset_object()
    dataset.read_file(dataset_name=dataset_names[dataset_name], drop_cols_datasets=drop_cols_names_datasets.get(dataset_name, None))
    dataset.get_cols_name(categorical_cols_names_datasets[dataset_name], numeric_cols_names_datasets[dataset_name], continuous_cols_names_datasets[dataset_name])
    dataset.impute_missing()
    if categorical_cols_names_datasets[dataset_name]:
        dataset.init_encoders()
    dataset.target_ordinal_encode()
    dataset.split_dataset()

    if continuous_cols_names_datasets[dataset_name]:
        custom_scaler = CustomScaler(
            scalers_dict=dataset.standard_scalers,
            continuous_col_names=dataset.continuous_col_names,
            continuous_cols=dataset.continuous_cols
        )
    else:
        custom_scaler = None

    # preparing data
    X_train = dataset.X_train
    X_test = dataset.X_test
    y_train = dataset.y_train
    if continuous_cols_names_datasets[dataset_name]:
        X_train = custom_scaler.transform(X_train)
        X_test = custom_scaler.transform(X_test)
    if categorical_cols_names_datasets[dataset_name]:
        X_train = dataset.onehot_encoder.transform(X_train)
        X_test = dataset.onehot_encoder.transform(X_test)

    # fit & evaluate
    clf = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=random_state,
        min_samples_split=min_samples_split,
        min_samples_leaf = min_samples_leaf,
        class_weight= 'balanced'
    )
    clf.fit(X_train, dataset.y_train)
    preds = clf.predict(X_test)

    # calculating Macro F1-score
    # f1 = f1_score(dataset.y_test, preds, average='macro')
    # precision_macro = precision_score(dataset.y_test, preds, average='macro')
    # scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='precision_macro')

    precision_scorer = make_scorer(precision_class_1)

    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring=precision_scorer)

    return scores.mean()

study = optuna.create_study(direction="maximize")  # maximize F1-score
study.optimize(objective, n_trials=500)

print("Best parameters:", study.best_params)
print("Best macro F1-score:", study.best_value)


# df = pd.read_csv('data/credit_risk.csv')
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), [0,8,9,10]),
#         ('cat', OrdinalEncoder(), [1,2,3,4,5,6,7,11])
#     ]
# )
# model = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=100, random_state=42))
#
# X_train, X_test, y_train, y_test = train_test_split(df.loc[:, 'age':'native-country'].values, df['class'].values,
#                                             test_size=0.3, random_state=42, stratify=df['class'].values)
# model.fit(X_train, y_train)
#
# bbox = sklearn_classifier_bbox.sklearnBBox(model)
