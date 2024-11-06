from typing import Text, List, Dict
from pathlib import Path
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
import joblib
import json

CVSS_METRIC_ORDER = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
PARAMS_PATH = Path(__file__).parent.parent.parent / "configs" / "lr_cve_params.json"

class ClassifierCVSS():
    def __init__(self, lm):
        self.lm = lm
        #with open(PARAMS_PATH) as f:
        #    params = json.load(f)
        #self.clf = LogisticRegression(**params)
        self.clf = svm.SVC(
            kernel="linear", C=1, class_weight="balanced", random_state=17
        )
        self.multi_output_clf = MultiOutputClassifier(self.clf, n_jobs=4)
        self.label_encoders: Dict[str, LabelEncoder] = {}

    def train(self, nvd_features: pd.DataFrame) -> None:
        labeled_df = nvd_features.query('cve_vector != "NVD-CVE-vector-noinfo"')
        if len(labeled_df) > 10000:
            labeled_df = labeled_df.sample(n=10000)
        descriptions = labeled_df["description"].values
        cvss_vectors = labeled_df["cve_vector"].values
        X_embeddings = self.lm.encode(descriptions)
        encoded_cvss_vectors = self._prepare_cvss_vectors(cvss_vectors)

        if self.load_model():
            print("Loaded trained CVSS classifier")
        else:
            print("Training new CVSS classifier")
            #self.evaluate(X_embeddings, encoded_cvss_vectors)
            self.multi_output_clf.fit(X_embeddings, encoded_cvss_vectors)
            self.save_model()

    def predict_vector(self, description: Text):
        X_embedding = self.lm.encode([description])
        predicted_encoded_vectors = self.multi_output_clf.predict(X_embedding)
        predicted_labels = []
        for i, metric in enumerate(CVSS_METRIC_ORDER):
            le = self.label_encoders[metric]
            label = le.inverse_transform([predicted_encoded_vectors[0][i]])[0]
            predicted_labels.append(label)

        cvss_vector = "CVSS:3.1/" + "/".join(
            [
                f"{metric}:{value}"
                for metric, value in zip(CVSS_METRIC_ORDER, predicted_labels)
            ]
        )
        return cvss_vector

    def predict(self, nvd_features: pd.DataFrame):
        # TODO: Only fill missing values
        missing = nvd_features.query('cvss_vector == "NVD-CVE-vector-noinfo"')

        return nvd_features.assign(
            cvss_vector=lambda nvd_features: self.predict_vector(
                nvd_features.description
            )
        )

    def _prepare_cvss_vectors(self, cvss_vectors: List[Text]):
        parsed_cvss_vectors = [
            self._parse_cvss_vector(vector) for vector in cvss_vectors
        ]
        encoded_cvss_vectors = np.zeros_like(parsed_cvss_vectors, dtype=int)
        for i, metric in enumerate(CVSS_METRIC_ORDER):
            if metric not in self.label_encoders:
                self.label_encoders[metric] = LabelEncoder()
            le = self.label_encoders[metric]
            metric_values = [vec[i] for vec in parsed_cvss_vectors]
            le.fit(metric_values)
            encoded_cvss_vectors[:, i] = le.transform(metric_values)
        return encoded_cvss_vectors

    def _parse_cvss_vector(self, cvss_vector: Text) -> List:
        parts = cvss_vector.split("/")[1:]
        components = {part.split(":")[0]: part.split(":")[1] for part in parts}
        return [components[metric] for metric in CVSS_METRIC_ORDER]

    def evaluate(self, X_embeddings: np.ndarray, encoded_cvss_vectors: Dict):
        kf = KFold(n_splits=5, shuffle=True)
        scores = {
            metric: {"Accuracy": [], "f1": []}
            for metric in CVSS_METRIC_ORDER
        }

        for train_idx, test_idx in kf.split(X_embeddings):
            X_train, X_test = X_embeddings[train_idx], X_embeddings[test_idx]
            y_train, y_test = (
                encoded_cvss_vectors[train_idx],
                encoded_cvss_vectors[test_idx],
            )

            multi_output_clf = MultiOutputClassifier(self.clf, n_jobs=4)
            multi_output_clf.fit(X_train, y_train)
            y_pred = multi_output_clf.predict(X_test)

            for idx, metric in enumerate(CVSS_METRIC_ORDER):
                acc_score = metrics.accuracy_score(y_test[:, idx], y_pred[:, idx])
                scores[metric]["Accuracy"].append(acc_score)

                f1 = metrics.f1_score(y_test[:, idx], y_pred[:, idx], average="macro")
                scores[metric]["f1"].append(f1)

        average_scores = {
            metric: {
                score_type: np.mean(values)
                for score_type, values in metric_scores.items()
            }
            for metric, metric_scores in scores.items()
        }
        print("Average scores for each CVSS metric:")
        for metric, metric_scores in average_scores.items():
            print(f"{metric}:")
            for score_type, value in metric_scores.items():
                print(f"  {score_type}: {value:.4f}")

    def load_model(self):
        model_path = MODELS_DIR / "cvss_model.pkl"
        encoders_path = MODELS_DIR / "encoders.pkl"
        if model_path.is_file() and encoders_path.is_file():
            self.multi_output_clf = joblib.load(model_path)
            self.label_encoders = joblib.load(encoders_path)
            return True
        else:
            print("The file does not exist.")
            return False

    def save_model(self):
        model_path = MODELS_DIR / "cvss_model.pkl"
        encoders_path = MODELS_DIR / "encoders.pkl"
        joblib.dump(self.multi_output_clf, model_path)
        joblib.dump(self.label_encoders, encoders_path)


