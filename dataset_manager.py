import os
from typing import Dict, List, Tuple
import pandas as pd
from alibi.utils import gen_category_map
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from collections import Counter
from collections import defaultdict

class dataset_object:
    def __init__(self, dataset_name, directory: str = './data/'):
        self.raw = self.read_file(dataset_name, directory)
        self.predef_preprocessing(dataset_name)
        self.raw_ohe = []
        self.data = self.raw.iloc[:,:-1]
        self.data_labeled = self.data.__deepcopy__()
        self.data_ohe = []
        self.multi_target = False
        self.target_names = self.raw.iloc[:, -1].squeeze().unique()
        self.target_map = {name: idx for idx, name in enumerate(self.target_names)}
        self.target = self.raw.iloc[:, -1].squeeze().map(self.target_map)
        self.feature_names = self.raw.iloc[:,:-1].columns
        self.feature_ohe_names = []
        self.category_map = gen_category_map(self.data)
        self.categorical_features = list(self.data.columns[list(self.category_map.keys())])
        self.numerical_features = [name for name in self.feature_names if name not in self.categorical_features]
        self.features_map = []

    def read_file(self, dataset_name: str, directory: str = './data/') -> pd.DataFrame:
        _, _, stats_filenames = os.walk(directory).__next__()
        for stat_filename in stats_filenames:
                if dataset_name not in stat_filename:
                    continue
                else:
                    with open(f'{directory}' + stat_filename, 'r') as file:
                        print(f"Reading {stat_filename} from {directory}")
                        return pd.read_csv(file)
        print("File not found")

    def predef_preprocessing(self, dataset_name):
        if dataset_name == "titanic_nan_prepr":
            self.raw.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)


    def class_distribution(self, y):
        counts = Counter(y)
        total = sum(counts.values())
        return {cls: f"{(count / total) * 100:.2f}%" for cls, count in counts.items()}

    def split_dataset(self, use_labeled = False, use_ohe = False, random_state=None, test_size = 0.3):
        if use_ohe:
            if random_state:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_ohe.to_numpy(), self.target, test_size=test_size,
                                                                random_state=random_state)
            else:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_ohe.to_numpy(), self.target,
                                                                                    test_size=test_size)
        elif use_labeled:
            if random_state:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_labeled.to_numpy(), self.target, test_size=test_size,
                                                                                        random_state=random_state)
            else:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_labeled.to_numpy(), self.target,
                                                                                        test_size=test_size)
        else:
            if random_state:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data.to_numpy(), self.target, test_size=test_size,
                                                                random_state=random_state)
            else:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data.to_numpy(), self.target,
                                                                                    test_size=test_size)
        return f"Training Set Balance: {self.class_distribution(self.y_train)}" + "\n" + f"Test Set Balance: {self.class_distribution(self.y_test)}"

    def init_preprocessor(self):
        ordinal_features = [x for x in range(len(self.feature_names)) if x not in list(self.category_map.keys())]
        # ordinal_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
        #                                       ('scaler', StandardScaler())])
        ordinal_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))])
        categorical_features = list(self.category_map.keys())
        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                                  ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        self.preprocessor = ColumnTransformer(transformers=[('num', ordinal_transformer, ordinal_features), ('cat', categorical_transformer, categorical_features)],
                                                  sparse_threshold=0)

        self.preprocessor.fit(self.X_train)

    def init_nan_preprocessor(self):
        ordinal_features = [x for x in range(len(self.feature_names)) if x not in list(self.category_map.keys())]
        # ordinal_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
        #                                       ('scaler', StandardScaler())])
        ordinal_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))])
        categorical_features = list(self.category_map.keys())
        # categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(missing_values="?", strategy='most_frequent'))])
        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(fill_value="?", strategy='constant'))])

        self.nan_preprocessor = ColumnTransformer(transformers=[('num', ordinal_transformer, ordinal_features), ('cat', categorical_transformer, categorical_features)],
                                                  sparse_threshold=0).set_output(transform='pandas')

        self.nan_preprocessor.fit(self.data)

    def nan_preproces(self):
        self.data = self.nan_preprocessor.transform(self.data)
        self.data_labeled = self.nan_preprocessor.transform(self.data_labeled)

    def onehot_encode(self):
        if not self.multi_target:
            dfX = pd.get_dummies(self.data, prefix_sep='_').astype(int)
            df = pd.concat([dfX, self.target], axis=1)
            df = df.reindex(dfX.index)
            self.raw_ohe = df
            self.data_ohe = dfX
            self.feature_ohe_names = dfX.columns
        else:  # isinstance(class_name, list)
            print("To Do")

    def get_features_map(self):
        features_map = defaultdict(dict)
        i = 0
        j = 0
        real_feature_names = self.numerical_features + self.categorical_features
        while i < len(self.feature_ohe_names) and j < len(real_feature_names):
            if self.feature_ohe_names[i] == real_feature_names[j]:
                features_map[j][self.feature_ohe_names[i].replace('%s_' % real_feature_names[j], '')] = i
                i += 1
                j += 1
            elif self.feature_ohe_names[i].startswith(real_feature_names[j]):
                features_map[j][self.feature_ohe_names[i].replace('%s_' % real_feature_names[j], '')] = i
                i += 1
            else:
                j += 1
        self.features_map = features_map

    def label_encoding(self):
        category_map = {}
        for f in self.categorical_features:
            le = LabelEncoder()
            data_tmp = le.fit_transform(self.data_labeled[f].values)
            self.data_labeled[f] = data_tmp
            category_map[list(self.feature_names).index(f)] = list(le.classes_)
        self.category_map = category_map

