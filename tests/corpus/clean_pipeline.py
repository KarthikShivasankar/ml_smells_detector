"""A well-behaved scikit-learn workflow: expected to produce no findings."""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 7
np.random.seed(SEED)

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=SEED)

pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced", random_state=SEED))
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scores = cross_val_score(pipeline, X_train, y_train, cv=folds)
pipeline.fit(X_train, y_train)
proba = pipeline.predict_proba(X_test)
print(scores.mean(), roc_auc_score(y_test, proba, multi_class="ovr"))
print(classification_report(y_test, pipeline.predict(X_test)))
