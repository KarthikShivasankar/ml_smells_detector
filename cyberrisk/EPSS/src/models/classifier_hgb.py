import sys
sys.path.append("..")
from .model import EPSS
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn import svm
from src.utils import numericalise_features, read_observations
import pandas as pd
import src.utils.evaluation as evaluation
import numpy as np

class ClassifierHGB(EPSS):
    def __init__(self, config):
        super().__init__(config)
        self.clf = HistGradientBoostingClassifier(learning_rate=0.01,
                                                  max_depth=None,
                                                  random_state=16)
        
    def format_data(self, features: pd.DataFrame) -> (pd.DataFrame, list):
        target = self.cast_target()
        features = pd.merge(features, target, on="cve", how="left")
        features["count"] = features["count"].fillna(0)
        data_with_counts = features[features["count"] > 0]
        data_without_counts = features[features["count"] == 0].sample(n=len(data_with_counts), random_state=16)
        data = pd.concat([data_with_counts, data_without_counts], axis=0, ignore_index=True)
        target = list(data["count"])
        X = data.drop(columns=["count"])
        return X, target
    
    def get_cve2days(self, observations: pd.DataFrame) -> dict:
        day = self.config.date
        CVEs = list(set(observations["cve"]))
        CVEs.sort()
        cve2days = {cve: 0 for cve in CVEs}
        for i, row in observations.iterrows():
            cve_day = row["formatted_date"]
            if cve_day <= day:
                continue
            if (cve_day - day).days >= 1 and (cve_day - day).days <= self.config.horizon:
                cve2days[row["cve"]] += 1
        return cve2days

    def train(self, features: pd.DataFrame) -> None:
        raw_model_input, target = self.format_data(features)
        for module in self.modules:
            module.train(features)
            raw_model_input = module.predict(raw_model_input)
        model_input = numericalise_features(raw_model_input, self.config)
        self.clf.fit(model_input, target)
    
    def predict(self, features: pd.DataFrame, evaluation: bool=False) -> np.array:
        for module in self.modules:
            features = module.predict(features)
        model_input = numericalise_features(features, self.config)
        return self.clf.predict(model_input)

    def predict_epss(self, features: pd.DataFrame) -> list:
        for module in self.modules:
            features = module.predict(features)
        model_input = numericalise_features(features, self.config)
        return [1 - x[0] for x in self.clf.predict_proba(model_input)]

    def evaluate(self, features: pd.DataFrame) -> None:
        raw_model_input, target = self.format_data(features)
        predictions = self.predict(raw_model_input)
        epss_scores = self.predict_epss(raw_model_input)
        print("Accuracy :", format(evaluation.eval_acc(target, predictions), ".3f"))
        print("MSE [p(seen|cve)] :", format(evaluation.eval_mse(target, predictions, self.config), ".3f"), " | Variance :", np.var([y/self.config.horizon for y in target]))
        print("R^2 [p(seen|cve)] :", format(evaluation.eval_r2(target, predictions, self.config), ".3f"))
        print("AUC [p(seen|cve)] :", format(evaluation.eval_auc([1 if t > 0 else 0 for t in target], epss_scores)))

        

    def cast_target(self) -> pd.DataFrame:
        observations = read_observations(self.config.observations_filepath)
        cve2days = self.get_cve2days(observations)
        df = pd.DataFrame()
        CVEs = list(set(observations["cve"]))
        CVEs.sort()
        counts = []
        for cve in CVEs:
            counts.append(cve2days[cve])
        df["cve"] = CVEs
        df["count"] = counts
        return df
