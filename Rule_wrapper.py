import re
from typing import List, Dict, Union, Optional, Tuple
import operator
import pandas as pd


class rule_wrapper:
    def __init__(
        self,
        explainer: str,
        premises: List[str],
        consequence: str,
    ):
        self.premises: List[Dict[str, Union[str, float]]] = premises
        self.consequence: Dict[str, Union[str, float]] = consequence
        self.explainer = explainer

    @classmethod
    def from_rule(cls, rule, outcome, explainer, numeric_cols=None):
        if explainer == 'ANCHOR':
            premises, consequence = [cls.anchor_parser(p) for p in rule], cls.anchor_parser(outcome)
            premises = sorted(premises, key=lambda r: r['attr'])
        elif explainer == 'LORE':
            premises = []
            for attr, cond_str in rule.items():
                if cls.is_compound_condition(cond_str):
                    for simple_rule in cls.split_compound_condition(attr, cond_str):
                        for attr_simple, cond_str_simple in simple_rule.items():
                            premises.append(cls.lore_parser(attr_simple, cond_str_simple, numeric_cols))
                else:
                    premises.append(cls.lore_parser(attr, cond_str, numeric_cols))
            for attr, cond_str in outcome.items():
                consequence = cls.lore_parser(attr, cond_str, None)
            premises = sorted(premises, key=lambda r: r['attr'])
        elif explainer == 'LORE_SA':
            premises, consequence = [cls.lore_sa_parser(p) for p in rule], cls.lore_sa_parser(outcome)
            premises = sorted(premises, key=lambda r: r['attr'])
        elif explainer == 'EXPLAN':
            premises = []
            for attr, cond_str in rule.items():
                if cls.is_compound_condition(cond_str):
                    for simple_rule in cls.split_compound_condition(attr, cond_str):
                        for attr_simple, cond_str_simple in simple_rule.items():
                            premises.append(cls.lore_parser(attr_simple, cond_str_simple, numeric_cols))
                else:
                    premises.append(cls.lore_parser(attr, cond_str, numeric_cols))
            for attr, cond_str in outcome.items():
                consequence = cls.lore_parser(attr, cond_str, None)
            premises = sorted(premises, key=lambda r: r['attr'])
        else:
            raise ValueError(f"Unknown explainer: {explainer}")

        return cls(explainer, premises, consequence)

    @staticmethod
    def anchor_parser(rule: str) -> Dict[str, Union[str, str | float]]:
        """parse a simple rule string into {'attr', 'op', 'val'} format."""
        pattern = re.compile(r"(.+?)\s*(<=|>=|!=|=|<|>)\s*(.+)")
        match = pattern.match(rule)
        if not match:
            raise ValueError(f"Cannot parse rule: {rule}")
        attr, op, val = match.groups()

        # converting val to float if possible
        try:
            val = float(val)
        except ValueError:
            val = val.strip()

        predicates = {'attr': attr.strip(), 'op': op.strip(), 'val': val}

        return predicates

    @staticmethod
    def lore_parser(attr: str, cond_str: str, numeric_cols: Optional[List[str]] = None) -> Dict[str, Union[str, float]]:
        """
        Parse a single predicate from attribute and condition string into {'attr', 'op', 'val'} format.
        """

        if numeric_cols is not None and attr in numeric_cols:
            pattern = re.compile(r"^(<=|>=|!=|=|<|>)?(.*)$")
            cond_str = cond_str.strip()
            match = pattern.match(cond_str)
            if not match:
                raise ValueError(f"Cannot parse condition: {cond_str} for attribute {attr}")

            op, val_str = match.groups()
            # converting val to float if possible
            val_str = val_str.strip()
            try:
                val = float(val_str)
            except ValueError:
                val = val_str
        else:
            val = cond_str
            op = '='

        return {'attr': attr, 'op': op, 'val': val}

    @staticmethod
    def lore_sa_parser(premises: Dict[str, Union[str, str|float]]) -> Dict[str, Union[str, str|float]]:
        return premises.copy()

    @staticmethod
    def _op_func(op_str):
        print(f"Taking {op_str}")
        return {
            '<': operator.lt,
            '<=': operator.le,
            '=': operator.eq,
            '!=': operator.ne,
            '>': operator.gt,
            '>=': operator.ge,
        }[op_str]

    @staticmethod
    def is_compound_condition(cond: str) -> bool:
        """
        Returns True if the condition string represents a compound condition
        like '1848.0< attr <=2005.0'.
        """
        # Remove surrounding whitespace
        cond = cond.strip()

        # match patterns like: 1848.0< attr <=2005.0 OR 10 <= attr < 20
        compound_pattern = re.compile(
            r"^\s*([0-9.eE+-]+)\s*(<|<=)\s*(\w+)\s*(<|<=|>=|>|=|!=)\s*([0-9.eE+-]+)\s*$"
        )

        return bool(compound_pattern.match(cond))

    @staticmethod
    def split_compound_condition(attr: str, cond: str) -> List[Tuple[str, str]]:
        """
        Detect and split a compound condition like '1848.0< capital.loss <=2005.0'
        into two simple conditions: 'capital.loss > 1848.0' and 'capital.loss <= 2005.0'
        """
        pattern = re.compile(
            r"^\s*([0-9.eE+-]+)\s*(<|<=)\s*(\w+)\s*(<|<=|>=|>|=|!=)\s*([0-9.eE+-]+)\s*$"
        )
        match = pattern.match(cond.strip())
        if not match:
            raise ValueError(f"Not a valid compound condition: {cond}")

        left_val, left_op, var, right_op, right_val = match.groups()

        inverse_op = {'<': '>', '<=': '>='}
        left_op_inv = inverse_op[left_op]

        left_condition = f"{left_op_inv}{left_val}"
        right_condition = f"{right_op}{right_val}"

        return [{var: left_condition}, {var: right_condition}]

    def matches_raw_rule(self, raw_rule, raw_outcome, numeric_cols=None):
        """Compares a raw rule to the stored one """

        if self.explainer == 'ANCHOR':
            compared_premises, compared_consequence = [self.anchor_parser(p) for p in raw_rule], self.anchor_parser(raw_outcome)
            compared_premises = sorted(compared_premises, key=lambda r: r['attr'])
        elif self.explainer == 'LORE':
            # compared_premises, compared_consequence = [self.lore_parser(p, numeric_cols) for p in raw_rule], self.lore_parser(raw_outcome)
            # compared_premises = sorted(compared_premises, key=lambda r: r['attr'])
            compared_premises = []
            for attr, cond_str in raw_rule.items():
                if self.is_compound_condition(cond_str):
                    for simple_rule in self.split_compound_condition(attr, cond_str):
                        for attr_simple, cond_str_simple in simple_rule.items():
                            compared_premises.append(self.lore_parser(attr_simple, cond_str_simple, numeric_cols))
                else:
                    compared_premises.append(self.lore_parser(attr, cond_str, numeric_cols))
            for attr, cond_str in raw_outcome.items():
                compared_consequence = self.lore_parser(attr, cond_str, None)
            compared_premises = sorted(compared_premises, key=lambda r: r['attr'])
        elif self.explainer == 'LORE_SA':
            compared_premises, compared_consequence = [self.lore_sa_parser(p) for p in raw_rule], self.lore_sa_parser(raw_outcome)
            compared_premises = sorted(compared_premises, key=lambda r: r['attr'])
        elif self.explainer == 'EXPLAN':
            compared_premises = []
            for attr, cond_str in raw_rule.items():
                if self.is_compound_condition(cond_str):
                    for simple_rule in self.split_compound_condition(attr, cond_str):
                        for attr_simple, cond_str_simple in simple_rule.items():
                            compared_premises.append(self.lore_parser(attr_simple, cond_str_simple, numeric_cols))
                else:
                    compared_premises.append(self.lore_parser(attr, cond_str, numeric_cols))
            for attr, cond_str in raw_outcome.items():
                compared_consequence = self.lore_parser(attr, cond_str, None)
            compared_premises = sorted(compared_premises, key=lambda r: r['attr'])

        return self.premises == compared_premises

    def _apply_condition(self, df: pd.DataFrame, cond: dict) -> pd.Series:
        op_func = self._op_func(cond['op'])
        attr_series = df[cond['attr']]
        try:
            attr_series = attr_series.astype(type(cond['val']))
        except Exception:
            pass
        return op_func(attr_series, cond['val'])
        # return op_func(cond_type(df[cond['attr']]), cond['val'])

    def evaluate_on(self, df: pd.DataFrame) -> dict:
        # premises mask
        mask = pd.Series([True] * len(df), index=df.index)
        for cond in self.premises:
            mask &= self._apply_condition(df, cond)
        covered = df[mask]
        covered_and_outcome = covered[self._apply_condition(covered, self.consequence)]
        total = len(df)
        # coverage
        coverage = len(covered) / total

        # class coverage (relative to outcome class)
        outcome_df = df[self._apply_condition(df, self.consequence)]
        local_cov_val = len(covered_and_outcome) / len(outcome_df)

        temp_local_cov_val = len(covered) / len(outcome_df)

        # precision
        precision = 0
        if len(covered) > 0:
            outcome_mask = self._apply_condition(covered, self.consequence)
            precision = outcome_mask.mean()


        return f"Cov,Cov_class,Cov_temp {round(coverage, 5)}; {round(local_cov_val, 5)}; {round(temp_local_cov_val, 5)}, Pre, Len {round(precision, 5)},{len(self.premises)}"

    def get_rule(self):
        # join premises as "attr op val" separated by " AND "
        premises_str = " AND ".join(
            f"{p['attr']} {p['op']} {p['val']}" for p in self.premises
        )
        consequence_str = f"{self.consequence['attr']} {self.consequence['op']} {self.consequence['val']}"

        return f"{self.explainer}: IF {premises_str} THEN {consequence_str}"

    def __repr__(self):
        return f"<rule_wrapper premises={self.premises}, consequence={self.consequence}>"