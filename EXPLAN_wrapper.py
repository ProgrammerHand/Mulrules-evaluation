import copy

from EXPLAN.LORE import util
from EXPLAN import explan
from Rule_wrapper import rule_wrapper
import numpy as np
from collections import Counter
import time

class explan_object:

    def __init__(
            self,
            dataset_name,
            raw,
            encoded_data,
            encoders,
            X_test,
            continuous_col_names,
            numeric_cols_names_datasets,
            categorical_col_names,
            target_val,
            target_name='class'
    ):
        self.dataset_name = dataset_name
        self.raw = raw
        self.encoded_data = encoded_data
        self.encoders = copy.deepcopy(encoders)
        self.X_test = X_test
        self.continuous_col_names = continuous_col_names
        self.numeric_cols_names = numeric_cols_names_datasets
        self.categorical_col_names = categorical_col_names
        self.target_val = target_val
        self.target_name = target_name

        # optional/default attributes
        self.inst = None
        self.N_samples = None
        self.tau = None

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
            'discrete_indices': categorical_cols,
            'discrete_names': self.categorical_col_names,
            'feature_names': columns_tmp,
            'X': self.encoded_data.values,
            'y': self.target_val,
        }

        return dataset

    def init_explainer(self, categorical_cols, target_encoder, iter_limit=15, N_samples = 3000, tau = 250):
        self.dataset = self.prepare_dataset(categorical_cols, target_encoder)
        self.N_samples = N_samples
        self.tau = tau
        self.iter_limit = iter_limit
        return f"Initializing EXPLAN Explainer with config: N_samples={self.N_samples}, tau={self.tau}, iter_limit={self.iter_limit} "

    def explain(self, amount, idx, predict_fn_anchor):
        instance2explain = np.array(self.X_test.iloc[idx])
        explanations = []
        infos = []
        rejected_count = Counter({i: 0 for i in range(amount)})
        i = 0
        start_rule_time = time.time()
        # while len(explanations) < amount:
        for n in range(self.iter_limit):
            print(n)
            explanation, info = explan.Explainer(instance2explain,
                                                       predict_fn_anchor,
                                                       self.dataset,
                                                       N_samples=self.N_samples,
                                                       tau=self.tau)
            if len(explanations) > 0:
                if amount - len(explanations) < self.iter_limit - n:
                    flag = False
                    for rule in explanations:
                        if rule.matches_raw_rule(explanation[1], explanation[0], self.numeric_cols_names):
                            flag = True
                            rejected_count[i] += 1
                            break
                    if not flag:
                        elapsed = time.time() - start_rule_time
                        explanations.append(rule_wrapper.from_rule(explanation[1], explanation[0], "EXPLAN", rejected_count[i], elapsed, False, self.numeric_cols_names))
                        i += 1
                        start_rule_time = time.time()
                else:
                    elapsed = time.time() - start_rule_time
                    explanations.append(
                        rule_wrapper.from_rule(explanation[1], explanation[0], "EXPLAN", rejected_count[i], elapsed, True, self.numeric_cols_names))
                    i += 1
                    start_rule_time = time.time()

            else:
                elapsed = time.time() - start_rule_time
                explanations.append(rule_wrapper.from_rule(explanation[1], explanation[0], "EXPLAN", rejected_count[i], elapsed, False, self.numeric_cols_names))
                i += 1
                start_rule_time = time.time()
            # if all(old_exp != explanation for old_exp in explanations):
            #     explanations.append(explanation)
            #     infos.append(info)
            if len(explanations) >= amount:
                break
        # return explanations, infos
        return explanations

    def print_explanation(self, explanation, information):
        rule_str = " AND ".join([f"{k}{v}" for k, v in explanation[1].items()])
        then_str = information['y_x_bb']
        result_string = f"EXPLAN: IF {rule_str} THEN {then_str} Pre, Cov :"
        return result_string