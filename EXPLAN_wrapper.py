import copy

from EXPLAN.LORE import util
from EXPLAN import explan
from Rule_wrapper import rule_wrapper
import numpy as np

class explan_object:

    def __init__(self, dataset_name, raw, encoded_data, encoders, X_test, continuous_col_names, categorical_col_names, target_val, target_name = 'class'):
        self.X_test = X_test
        self.raw = raw
        self.dataset_name = dataset_name
        self.target_name = target_name
        self.continuous_col_names = continuous_col_names
        self.categorical_col_names = categorical_col_names
        self.target_val = target_val
        self.encoded_data = encoded_data
        self.encoders = copy.deepcopy(encoders)
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

    def init_explainer(self, categorical_cols, target_encoder, N_samples = 3000, tau = 250):
        self.dataset = self.prepare_dataset(categorical_cols, target_encoder)
        self.N_samples = N_samples
        self.tau = tau
        return f"Initializing EXPLAN Explainer with config: N_samples={N_samples} tau={tau}"

    def explain(self, amount, idx, predict_fn_anchor):
        instance2explain = np.array(self.X_test.iloc[idx])
        explanations = []
        infos = []
        while len(explanations) < amount:
            explanation, info = explan.Explainer(instance2explain,
                                                       predict_fn_anchor,
                                                       self.dataset,
                                                       N_samples=self.N_samples,
                                                       tau=self.tau)
            if len(explanations) > 0:
                flag = False
                for rule in explanations:
                    if rule.matches_raw_rule(explanation[1], explanation[0], self.continuous_col_names):
                        flag = True
                        break
                if not flag:
                    explanations.append(rule_wrapper.from_rule(explanation[1], explanation[0], "EXPLAN", self.continuous_col_names))
            else:
                explanations.append(rule_wrapper.from_rule(explanation[1], explanation[0], "EXPLAN", self.continuous_col_names))
            # if all(old_exp != explanation for old_exp in explanations):
            #     explanations.append(explanation)
            #     infos.append(info)
        # return explanations, infos
        return explanations

    def print_explanation(self, explanation, information):
        rule_str = " AND ".join([f"{k}{v}" for k, v in explanation[1].items()])
        then_str = information['y_x_bb']
        result_string = f"EXPLAN: IF {rule_str} THEN {then_str} Pre, Cov :"
        return result_string