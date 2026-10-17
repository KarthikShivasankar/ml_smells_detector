"""A churn model trained with scikit-learn, with leakage and reproducibility problems."""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

frame = pd.read_csv("/data/churn/customers.csv", usecols=["tenure", "spend", "churned"])  # expect: ML.HARDCODED_PATH
features = frame[["tenure", "spend"]]
labels = frame["churned"]

scaled = StandardScaler().fit_transform(features)  # expect: ML.DATA_LEAKAGE
X_train, X_test, y_train, y_test = train_test_split(scaled, labels)  # expect: ML.SCIKITLEARN.RANDOMNESS_CONTROL_CHECKER

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))
if accuracy_score(y_test, predictions) > 0.85:  # expect: ML.MAGIC_NUMBER
    joblib.dump(model, "churn.joblib")  # expect: ML.SCIKITLEARN.EXCEPTION_HANDLING_CHECKER

backup = SVC()

# File-level findings (reported at line 0):
# expect-file: ML.ACCURACY_ONLY
# expect-file: ML.MISSING_SEED
# expect-file: ML.NO_CROSS_VALIDATION
# expect-file: ML.NO_ERROR_HANDLING
# expect-file: ML.SAVE_WITHOUT_PREPROCESSING
# expect-file: ML.SCIKITLEARN.CROSS_VALIDATION_CHECKER
# expect-file: ML.SCIKITLEARN.DEPENDENT_THRESHOLD_CHECKER
# expect-file: ML.SCIKITLEARN.PIPELINE_CHECKER
# expect-file: ML.UNUSED_FEATURES
