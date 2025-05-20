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

name = "german"
random_state = 42
dropped_cols = None

test_size = 0.3
dataset = read_file(name)
if dropped_cols:
    dataset = dataset.drop(dropped_cols, axis=1)

X = dataset.drop("class", axis=1)
y = dataset["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=random_state
)

train_df = X_train.copy()
train_df["class"] = y_train

test_df = X_test.copy()
test_df["class"] = y_test

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

print(print_class_distribution("Original", y))
print(print_class_distribution("Train", y_train))
print(print_class_distribution("Test", y_test))
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