import dataset_manager
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

def read_file(dataset_name: str, directory: str = './data/') -> pd.DataFrame:
    _, _, stats_filenames = os.walk(directory).__next__()
    for stat_filename in stats_filenames:
        if dataset_name not in stat_filename:
            continue
        else:
            with open(f'{directory}' + stat_filename, 'r') as file:
                print(f"Reading {stat_filename} from {directory}")
                return pd.read_csv(file, na_values="?")
    print("File not found")

name = "german_rename_vals"
random_state = 42
dropped_cols = None

test_size = 0.3
dataset = read_file(name)


# print(dataset["checking_status"].unique())
# dataset['checking_status'] = dataset['checking_status'].replace({
#     '<0': 'less 0',
#     '0<=X<200': 'between 0 and 200',
#     '>=200': 'greater 200',
#     'no checking': 'no checking'
# })
# print(dataset["checking_status"].unique())
# print(dataset["savings_status"].unique())
# dataset['savings_status'] = dataset['savings_status'].replace({
#     '<100': 'less 100',
#     '500<=X<1000': 'between 500 and 1000',
#     '>=1000': 'greater 1000',
#     '100<=X<500': 'between 100 and 500',
#     'no known savings': 'no known savings',
# })
# print(dataset["savings_status"].unique())
# print(dataset["employment"].unique())
# dataset['employment'] = dataset['employment'].replace({
#     '>=7': 'greater 7 years',
#     '1<=X<4': 'between 1 and 4 years',
#     '4<=X<7': 'between 4 and 7 years',
#     'unemployed': 'unemployed',
#     '<1': 'less 1 year',
# })
# print(dataset["employment"].nunique())
# for col in dataset.columns:
#     print(f"'{col}':")
#     print(dataset[col].unique())
#     print()

# dataset.to_csv(name + "_rename_vals.csv", index=False)
# if dropped_cols:
#     dataset = dataset.drop(dropped_cols, axis=1)
#
X = dataset.drop("class", axis=1)
y = dataset["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=random_state
)
df_test = X_test.copy()
df_test["class"] = y_test

filtered_df = df_test[
    (df_test['checking_status'] == 'less 0') &
    # (df_test['credit_amount'] > 3331.0) &
    # (df_test['credit_amount'] <= 3676.0) &
    (df_test['duration'] > 15.099815) &
    # (df_test['foreign_worker'] <= 'yes') &
    (df_test['savings_status'] == 'less 100')
]
print()
# train_df = X_train.copy()
# train_df["class"] = y_train
#
# test_df = X_test.copy()
# test_df["class"] = y_test
#
# def print_class_distribution(name, y_series):
#     counts = y_series.value_counts()
#     percentages = y_series.value_counts(normalize=True) * 100
#     distribution = pd.DataFrame({
#         'Count': counts,
#         'Percentage': percentages.round(2)
#     })
#     return f"\n{name} class distribution:\n{distribution}"
#
# def nans_count_report(df, name):
#     nans = df.isna().sum()
#     return f"\n{name} NaN counts per column:\n{nans[nans > 0]}"
#
# print(print_class_distribution("Original", y))
# print(print_class_distribution("Train", y_train))
# print(print_class_distribution("Test", y_test))
# print(nans_count_report(dataset))

# with open(name+"_prepr_train"+'_info.txt', 'w') as output:
#     output.write(f"\ntest_size = {test_size}")
#     output.write(f"\nrandom_state = {random_state}")
#     output.write(f"\ndropped_cols = {dropped_cols}")
#     output.write("\n"+print_class_distribution("Original", y))
#     output.write("\n"+nans_count_report(dataset, "Original"))
#     output.write("\n"+print_class_distribution("Train", y_train))
#     output.write("\n"+nans_count_report(train_df, "Train"))
#
# with open(name+"_prepr_test"+'_info.txt', 'w') as output:
#     output.write(f"\ntest_size = {test_size}")
#     output.write(f"\nrandom_state = {random_state}")
#     output.write(f"\ndropped_cols = {dropped_cols}")
#     output.write("\n"+print_class_distribution("Original", y))
#     output.write("\n"+nans_count_report(dataset, "Original"))
#     output.write("\n"+print_class_distribution("Test", y_test))
#     output.write("\n"+nans_count_report(test_df, "Test"))


# train_df.to_csv(name + "_prepr_train.csv", index=False)
# test_df.to_csv(name + "_prepr_test.csv", index=False)