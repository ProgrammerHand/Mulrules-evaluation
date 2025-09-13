
import pandas as pd
import numpy as np
from lore_updated.dataset import TabularDataset
from lore_updated.neighgen import GeneticGenerator
from lore_updated.encoder_decoder import ColumnTransformerEnc
from lore_updated.lore import Lore
from lore_updated.surrogate import DecisionTreeSurrogate
from Rule_wrapper import rule_wrapper
from collections import Counter
import time

class lore_sa_object:

    def __init__(self, name, numeric_col_names, categorical_col_names, X_test, y_test, raw, train_df, target_name = 'class'):
        self.dataset = TabularDataset(data=raw, class_name="class", categorial_columns=categorical_col_names, numerical_columns=numeric_col_names)
        self.dataset.df = train_df
        self.X_test = X_test
        self.y_test = y_test
        self.train_df = train_df
        self.tabularLore = None
        self.inst = None
        self.num_instances = None

    def init_explainer(self, bbox, iter_limit=15, ngen=20, cxpb=0.7, mutpb=0.5, num_instances = 500):
        enc = ColumnTransformerEnc(self.dataset.descriptor)
        generator = GeneticGenerator(bbox=bbox, dataset=self.dataset, encoder=enc,
                                    ngen=ngen,
                                    cxpb=cxpb,
                                    mutpb=mutpb
                                    )
        surrogate = DecisionTreeSurrogate()
        self.tabularLoreExplainer = Lore(bbox, self.dataset, enc, generator, surrogate)
        self.iter_limit = iter_limit
        self.num_instances = num_instances
        return f"Initializing LORE_SA Explainer with params: iter_limit = {self.iter_limit}. ngen = {ngen}, cxpb= {cxpb}, mutpb = {mutpb}, num_instances = {self.num_instances}"

    def get_instance(self, idx):
        if type(self.X_test) == np.ndarray:
            self.inst = self.X_test[idx]
        else:
            self.inst = self.X_test.to_numpy()[idx]

    def normalize_rule(self, rule):
        return {
            'premises': sorted((p['attr'], p['op'], str(p['val'])) for p in rule['premises']),
            'consequence': (rule['consequence']['attr'], rule['consequence']['op'], str(rule['consequence']['val']))
        }

    def explain(self, amount, instance_name):
        explanations = []
        rejected_count = Counter({i: 0 for i in range(amount)})
        i = 0
        start_rule_time = time.time()
        for n in range(self.iter_limit):
            print(f"Lore_SA {n}")
            explanation = self.tabularLoreExplainer.explain(self.inst, num_instances = self.num_instances)
            if len(explanations) > 0:
                if amount - len(explanations) < self.iter_limit - n:
                    flag = False
                    for rule in explanations:
                        if rule.matches_raw_rule(explanation["rule"]["premises"], explanation["rule"]["consequence"]):
                            flag = True
                            rejected_count[i] += 1
                            break
                    if not flag:
                        elapsed = time.time() - start_rule_time
                        explanations.append(rule_wrapper.from_rule(instance_name, explanation["rule"]["premises"], explanation["rule"]["consequence"], "LORE_SA", rejected_count[i], elapsed, False))
                        i += 1
                        start_rule_time = time.time()
                else:
                    elapsed = time.time() - start_rule_time
                    explanations.append(
                        rule_wrapper.from_rule(instance_name, explanation["rule"]["premises"], explanation["rule"]["consequence"],
                                               "LORE_SA", rejected_count[i], elapsed, True))
                    i += 1
                    start_rule_time = time.time()

            else:
                elapsed = time.time() - start_rule_time
                explanations.append(rule_wrapper.from_rule(instance_name, explanation["rule"]["premises"], explanation["rule"]["consequence"], "LORE_SA", rejected_count[i], elapsed, False))
                i += 1
                start_rule_time = time.time()

            if len(explanations) >= amount:
                break
        return explanations

    def print_explanation(self, explanation):
        conditions = " AND ".join(
            [f"{part['attr']} {part['op']} {part['val']}" for part in explanation['rule']['premises']])
        result_string = f"LORE_sa: IF {conditions} THEN {explanation['rule']['consequence']['val']} Pre, Cov : {self.calculate_precision_coverage(explanation)}"
        return result_string

    def apply_condition(self, condition):
        if condition["op"] == '=':
            return self.raw[condition["attr"]] == condition["val"]
        elif condition["op"] == '!=':
            return self.raw[condition["attr"]] != condition["val"]
        elif condition["op"] == '<=':
            return self.raw[condition["attr"]] <= condition["val"]
        elif condition["op"] == '<':
            return self.raw[condition["attr"]] < condition["val"]
        elif condition["op"] == '>=':
            return self.raw[condition["attr"]] >= condition["val"]
        elif condition["op"] == '>':
            return self.raw[condition["attr"]] > condition["val"]

    def calculate_precision_coverage(self, explanation):

        condition_mask = pd.Series([True] * len(self.raw))  # start with all True (all rows selected)

        for condition in explanation["rule"]["premises"]:
            condition_mask &= self.apply_condition(condition)

        # filter rows that satisfy the rule
        filtered_data = self.raw[condition_mask]

        # Coverage: Proportion of rows that satisfy the rule
        coverage = len(filtered_data) / len(self.raw) if len(self.raw) > 0 else 0

        # Precision: Proportion of rows that satisfy the rule and match the predicted class
        correct_predictions = filtered_data[filtered_data[explanation['rule']['consequence']['attr']] == explanation['rule']['consequence']['val']]
        precision = len(correct_predictions) / len(filtered_data) if len(filtered_data) > 0 else 0

        return precision, coverage