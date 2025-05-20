import os
import pandas as pd
from alibi.utils import gen_category_map
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, StandardScaler


class dataset_object:
    def __init__(self):
        self.raw = None
        self.data = None
        self.numeric_cols = None
        self.categorical_cols = None
        self.numeric_col_names = None
        self.categorical_col_names = None
        self.target_names = None
        self.target_map = None
        self.categorical_map = None
        self.feature_names = None
        self.target = None
        self.ordinal_encoder = None
        self.onehot_encoder = None
        self.label_encoders = None
        self.reverse_target_map = None
        self.standard_scalers = {}
        self.continuous_cols = None
        self.continuous_col_names = None
        self.target_encoder = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def read_file(self, dataset_name: str, drop_cols_datasets = None, directory: str = './data/') -> str:
        _, _, stats_filenames = os.walk(directory).__next__()
        for stat_filename in stats_filenames:
            if dataset_name not in stat_filename:
                continue
            else:
                with open(f'{directory}' + stat_filename, 'r') as file:
                    if drop_cols_datasets:
                        self.raw = pd.read_csv(file, skipinitialspace=True, na_values='?').drop(columns=drop_cols_datasets)
                    else:
                        self.raw = pd.read_csv(file, skipinitialspace=True, na_values='?')
                    # self.raw.iloc[:, -1] = self.raw .iloc[:, -1].astype('object')
                    self.data = self.raw.iloc[:,:-1]
                    self.feature_names = self.raw.iloc[:, :-1].columns
                    return f"Reading {stat_filename} from {directory}"
                    # return pd.read_csv(file, skipinitialspace=True, na_values='?', keep_default_na=True)
        return "File not found"

    def read_split_files(self, dataset_name: str, drop_cols_datasets=None, directory: str = './data/') -> str:
        _, _, filenames = next(os.walk(directory))
        status_messages = []

        for file in filenames:
            if dataset_name in file and 'train' in file.lower():
                self.train_raw = pd.read_csv(os.path.join(directory, file))
                status_messages.append(f"Loaded train file: {file},")
            elif dataset_name in file and 'test' in file.lower():
                self.test_raw = pd.read_csv(os.path.join(directory, file))
                status_messages.append(f"Loaded test file: {file} from {directory}")

        self.raw = pd.concat([self.train_raw, self.test_raw], ignore_index=True)
        self.data = self.raw.iloc[:, :-1]
        self.feature_names = self.raw.iloc[:, :-1].columns

        if not status_messages:
            return f"No train/test files found for '{dataset_name}' in {directory}"

        return '\n'.join(status_messages)

    def get_cols_name(self, categorical_cols_names:list, numeric_cols_names:list, continuous_cols_names:list) -> str:
        self.numeric_col_names = numeric_cols_names
        self.continuous_col_names = continuous_cols_names
        self.categorical_col_names = categorical_cols_names
        if self.numeric_col_names:
            self.numeric_cols = [self.data.columns.get_loc(col) for col in self.numeric_col_names]
            if self.continuous_col_names:
                self.continuous_cols = [self.data.columns.get_loc(col) for col in self.continuous_col_names]
        if self.categorical_col_names:
            self.categorical_cols = [self.data.columns.get_loc(col) for col in self.categorical_col_names]

        return f"Numeric: {self.numeric_col_names} \n Categorical: {self.categorical_col_names}, Continuous: {self.continuous_col_names}"

    def impute_missing(self, num_strat='mean', cat_strat='most_frequent', cat_fill = "?"):
        num_imputer = SimpleImputer(strategy=num_strat)
        cat_imputer = SimpleImputer(strategy=cat_strat)
        if self.numeric_col_names:
            self.data[self.numeric_col_names] = num_imputer.fit_transform(self.data[self.numeric_col_names])
        if self.categorical_col_names:
            self.data[self.categorical_col_names] = self.data[self.categorical_col_names].astype('object')
            self.data[self.categorical_col_names] = cat_imputer.fit_transform(self.data[self.categorical_col_names])
        self.raw = pd.concat([self.data, self.raw.iloc[:, -1]], axis=1)
        self.data[self.categorical_col_names] = self.data[self.categorical_col_names].astype(str)
        self.raw[self.categorical_col_names] = self.raw[self.categorical_col_names].astype(str)
        self.categorical_map = gen_category_map(self.data, self.categorical_cols)
        return f"Num_strat: {num_strat}, Cat_strat: {cat_strat}"

    def target_ordinal_encode(self) -> str:

        self.target_encoder = LabelEncoder()
        self.target_encoder.fit(self.raw.iloc[:, -1].squeeze())
        self.target_names = self.raw.iloc[:, -1].squeeze().unique()
        self.target_map = {name: idx for idx, name in enumerate(self.target_names)}
        self.target = self.raw.iloc[:, -1].squeeze().map(self.target_map)
        self.reverse_target_map = {v: k for k, v in self.target_map.items()}
        return f"Target_map: {self.target_map}"

        # self.target_names = self.target_encoder.classes_  # Ordered list of labels
        # self.target_map = {name: idx for idx, name in enumerate(self.target_names)}
        # self.reverse_target_map = {idx: name for idx, name in enumerate(self.target_names)}
        # return f"Target_map: {self.target_map}"


    # def target_raw(self)->str:

    def init_encoders(self):
        unique_categories = [self.data.iloc[:, col].astype(str).unique().tolist() for col in self.categorical_cols]
        self.ordinal_encoder = ColumnTransformer(
            [('cat', OrdinalEncoder(categories=unique_categories, handle_unknown='use_encoded_value', unknown_value=-1),
              self.categorical_cols)], remainder='passthrough'
        ).fit(self.data)

        self.onehot_encoder = ColumnTransformer(
            [('cat', OneHotEncoder(categories=unique_categories, handle_unknown='ignore', sparse_output=False), self.categorical_cols)],
            remainder='passthrough'
        ).fit(self.data)

        # self.label_encoders = {col: LabelEncoder().fit(self.data[col]) for col in self.categorical_col_names}
        self.label_encoders = {col: LabelEncoder().fit(self.data.iloc[:, col]) for col in self.categorical_cols}
    def split_dataset(self, test_size=0.3,  random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.target, test_size=0.3, random_state=42, stratify=self.target)
        if self.continuous_col_names:
            for col in self.continuous_col_names:
                self.standard_scalers[col] = StandardScaler().fit(self.X_train[[col]])
        return f"Test_size: {test_size}, Random_state: {random_state}"

    def label_encode_features(self, X, categorical_cols, categorical_col_names) -> pd.DataFrame:
        X_copy = X.copy()
        for col in categorical_cols:
            if isinstance(X_copy, pd.DataFrame):  # If X is a DataFrame, use column names
                X_copy.iloc[:, col] = self.label_encoders[col].transform(X_copy.iloc[:, col])
            else:  # If X is a NumPy array, use array indexing
                X_copy[:, col] = self.label_encoders[col].transform(X_copy[:, col])
        return X_copy

    # def label_encode_features(self, X, categorical_cols, categorical_col_names) -> pd.DataFrame:
    #     X_copy = X.copy()
    #     if isinstance(X_copy, pd.DataFrame):
    #         for col in categorical_col_names:
    #             X_copy[col] = self.label_encoders[col].transform(X_copy[col])
    #     else:  # assume NumPy array
    #         for i, col in zip(categorical_cols, categorical_col_names):
    #             X_copy[:, i] = self.label_encoders[col].transform(X_copy[:, i])
    #     return X_copy

    def label_decode_features(self, X, categorical_cols, categorical_col_names) -> pd.DataFrame:
        X_copy = X.copy()
        for col in categorical_cols:
            if isinstance(X_copy, pd.DataFrame):  # If X is a DataFrame, use column names
                X_copy.iloc[:, col] = self.label_encoders[col].inverse_transform(X_copy.iloc[:, col].astype(int))
            else:  # If X is a NumPy array, use array indexing
                temp = self.label_encoders[col].inverse_transform(X_copy[:, col].astype(int))
                X_copy = X_copy.astype(type(temp))
                X_copy[:, col] = temp
        return X_copy

    # def label_decode_features(self, X, categorical_cols, categorical_col_names) -> pd.DataFrame:
    #     X_copy = X.copy()
    #     if isinstance(X_copy, pd.DataFrame):
    #         for col in categorical_col_names:
    #             X_copy[col] = self.label_encoders[col].inverse_transform(X_copy[col].astype(int))
    #     else:  # assume NumPy array
    #         for i, col in zip(categorical_cols, categorical_col_names):
    #             decoded = self.label_encoders[col].inverse_transform(X_copy[:, i].astype(int))
    #             X_copy[:, i] = decoded
    #     return X_copy