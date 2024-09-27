import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer
from imblearn.over_sampling import SMOTE

# Load data
data = pd.read_csv('data.csv')

# Split data
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle imbalanced dataset
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Scikit-learn model
rf_model = RandomForestClassifier()
rf_model.fit(X_train_resampled, y_train_resampled)

# Cross-validation
cv_scores = cross_val_score(rf_model, X_train_resampled, y_train_resampled, cv=5)

# Predict and evaluate
y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# TensorFlow model
tf_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

tf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
tf_model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2)

# PyTorch model
class PyTorchModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(PyTorchModel, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, 64)
        self.layer2 = torch.nn.Linear(64, 32)
        self.layer3 = torch.nn.Linear(32, 1)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        return x

torch_model = PyTorchModel(X_train.shape[1])
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(torch_model.parameters())

# Hugging Face model
hf_model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=hf_model,
    args=training_args,
    train_dataset=X_train,  # This is a placeholder, you'd need to create proper datasets
    eval_dataset=X_test,
)

# Train Hugging Face model
trainer.train()

# Save models
rf_model.save('rf_model.pkl')
tf_model.save('tf_model.h5')
torch.save(torch_model.state_dict(), 'torch_model.pth')
trainer.save_model('hf_model')

# Load and use a model (example with scikit-learn)
loaded_rf_model = RandomForestClassifier()
loaded_rf_model.load('rf_model.pkl')
new_predictions = loaded_rf_model.predict(X_test_scaled)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"Cross-validation scores: {cv_scores}")
