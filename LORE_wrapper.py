import copy

from LORE import util
from LORE import lore
from LORE import neighbor_generator
from Rule_wrapper import rule_wrapper
from collections import Counter
import time
import numpy as np

class lore_object:

    def __init__(self, dataset_name, raw, encoded_data, encoders, X_train, continuous_col_names, numeric_cols_names_datasets, categorical_col_names, target_val, target_name = 'class'):
        self.X_train = X_train
        self.raw = raw
        self.dataset_name = dataset_name
        self.target_name = target_name
        self.continuous_col_names = continuous_col_names
        self.numeric_cols_names = numeric_cols_names_datasets
        self.categorical_col_names = categorical_col_names
        self.target_val = target_val
        self.encoded_data = encoded_data
        self.encoders = copy.deepcopy(encoders)
        self.inst = None

    def prepare_dataset(self, categorical_cols, target_encoder):
        # Features Categorization
        columns = self.raw.columns
        possible_outcomes = list(self.raw[self.target_name].squeeze().unique())

        type_features, features_type = util.recognize_features_type(self.raw, self.target_name)

        discrete, continuous = util.set_discrete_continuous(columns, type_features, self.target_name, self.categorical_col_names, continuous=self.continuous_col_names)

        columns_tmp = list(columns)
        columns_tmp.remove(self.target_name)
        idx_features = {i: col for i, col in enumerate(columns_tmp)}

        self.encoders = {
            col_name: self.encoders[col_idx]
            for col_idx, col_name in zip(categorical_cols, self.categorical_col_names)
        }
        self.encoders[self.target_name] = target_encoder

        dataset = {
            'name': self.dataset_name,
            'df': self.raw,
            'columns': list(columns),
            'class_name': self.target_name,
            'possible_outcomes': possible_outcomes,
            'type_features': type_features,
            'features_type': features_type,
            'discrete': discrete,
            'continuous': continuous,
            'idx_features': idx_features,
            'label_encoder': self.encoders,
            'X': self.encoded_data.values,
            'y': self.target_val,
        }

        return dataset

    def init_explainer(self, categorical_cols, target_encoder, iter_limit=15, ng_function = neighbor_generator.genetic_neighborhood, discrete_use_probabilities = True, continuous_function_estimation= False ):
        self.dataset = self.prepare_dataset(categorical_cols, target_encoder)
        self.ng_function = neighbor_generator.genetic_neighborhood
        self.discrete_use_probabilities = discrete_use_probabilities
        self.continuous_function_estimation = continuous_function_estimation
        self.iter_limit = iter_limit
        return f"Initializing LORE Explainer with params: ng_function {self.ng_function}, discrete_use_probabilities {self.discrete_use_probabilities},continuous_function_estimation {self.continuous_function_estimation}, iter_limit = {self.iter_limit}"

    def explain(self, amount, instance_from_encoded, predict_fn_anchor, path_data = 'data/'):
        informations = []
        explanations = []
        rejected_count = Counter({i: 0 for i in range(amount)})
        i = 0
        start_time = time.time()
        # while len(explanations) < amount:
        start_rule_time = time.time()
        for n in range(self.iter_limit):
            print(n)
            try:
                explanation, infos = lore.explain(instance_from_encoded, np.array(self.X_train), self.dataset, predict_fn_anchor,
                                                  ng_function=neighbor_generator.genetic_neighborhood,
                                                  discrete_use_probabilities=True,
                                                  continuous_function_estimation=False,
                                                  returns_infos=True,
                                                  path=path_data, sep=';', log=False)
            except Exception as e:
                print(f"Warning: Error during explanation generation at iteration {n}: {e}")
                continue
            if len(explanations) > 0:
                if amount - len(explanations) < self.iter_limit - n:
                    flag = False
                    for rule in explanations:
                        if rule.matches_raw_rule(explanation[0][1], explanation[0][0], self.continuous_col_names):
                            flag = True
                            rejected_count[i] += 1
                            break
                    if not flag:
                        elapsed = time.time() - start_rule_time
                        explanations.append(rule_wrapper.from_rule(explanation[0][1], explanation[0][0], "LORE", rejected_count[i], elapsed, False, self.numeric_cols_names))
                        i += 1
                        start_rule_time = time.time()
                        # print(explanation)
                else:
                    elapsed = time.time() - start_rule_time
                    explanations.append(
                        rule_wrapper.from_rule(explanation[0][1], explanation[0][0], "LORE", rejected_count[i], elapsed, True, self.numeric_cols_names))
                    i += 1
                    start_rule_time = time.time()
            else:
                elapsed = time.time() - start_rule_time
                explanations.append(rule_wrapper.from_rule(explanation[0][1], explanation[0][0], "LORE", rejected_count[i], elapsed, False, self.numeric_cols_names))
                i += 1
                start_rule_time = time.time()
            if len(explanations) >= amount:
                break
                # print(explanation)
            # if all(old_exp[0][1] != explanation[0][1] for old_exp in explanations):
            #     explanations.append(explanation)
            #     informations.append(infos)
        # return explanations, informations
        return explanations

    def print_explanation(self, explanation, information):
        rule_str = " AND ".join([f"{k}{v}" for k, v in explanation[0][1].items()])
        then_str = [f"{k} == '{v}'" for k, v in explanation[0][0].items()]
        result_string = f"LORE: IF {rule_str} THEN {then_str} Pre, Cov :"
        return result_string