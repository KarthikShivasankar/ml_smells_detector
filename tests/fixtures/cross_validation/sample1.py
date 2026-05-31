import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Accuracy: {score}")
