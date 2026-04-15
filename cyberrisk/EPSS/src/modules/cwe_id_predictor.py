from sklearn import svm
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

class ClassifierCWE(object):
    
    def __init__(self, lm: SentenceTransformer) -> None:
        self.lm = lm
        self.classifier = svm.SVC(kernel='linear', C=1, class_weight="balanced", random_state=17)
        
    def train(self, nvd_features: pd.DataFrame) -> None:
        labelled = nvd_features[nvd_features["cwe_id"] != "NVD-CWE-noinfo"]
        if len(labelled) > 10000:
            labelled = labelled.sample(n=10000, random_state=17)
        missing = nvd_features[nvd_features["cwe_id"] == "NVD-CWE-noinfo"]
        embedded_descriptions = [self.lm.encode(d) for d in labelled["description"]]
        self.cwe2id = self.get_cwe2id_dict(nvd_features)
        self.id2cwe = self.get_id2cwe_dict(nvd_features)
        target = self.numericalise_target(labelled["cwe_id"])
        self.classifier.fit(embedded_descriptions, target) 
        
    def predict(self, nvd_features: pd.DataFrame) -> pd.DataFrame:
        nvd_features["cwe_id"].mask(nvd_features["cwe_id"] == "NVD-CWE-noinfo", np.nan, inplace=True)
        missing = nvd_features[nvd_features["cwe_id"].isnull()]
        if len(missing) == 0:
            return nvd_features
        print("number of missing CWE labels", len(missing))
        missing_CVEs = missing["cve"]
        embedded_descriptions = [self.lm.encode(d) for d in missing["description"]]
        predictions = [self.id2cwe[p] for p in self.classifier.predict(embedded_descriptions)]
        predicted_df = pd.DataFrame()
        predicted_df["cve"] = missing_CVEs
        predicted_df["predicted_cwe_id"] = predictions
        nvd_features = pd.merge(nvd_features, predicted_df, on="cve", how="left")
        nvd_features["cwe_id"] = nvd_features["cwe_id"].fillna(nvd_features["predicted_cwe_id"])
        nvd_features["cwe_id"] = nvd_features["cwe_id"].fillna("NVD-CWE-noinfo")
        return nvd_features

    def get_cwe2id_dict(self, nvd_features: pd.DataFrame) -> dict:
        cwe_ids = [cwe for cwe in list(nvd_features["cwe_id"])]
        cwe_ids.sort()
        return {cwe: i for i, cwe in enumerate(cwe_ids)}

    def get_id2cwe_dict(self, nvd_features: pd.DataFrame) -> dict:
        cwe_ids = [cwe for cwe in list(nvd_features["cwe_id"])]
        cwe_ids.sort()
        return {i: cwe for i, cwe in enumerate(cwe_ids)}

    def numericalise_target(self, target: pd.Series) -> list:
        return [self.cwe2id[t] for t in target]
        
