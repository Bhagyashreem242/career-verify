import pandas as pd
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Load dataset
df = pd.read_csv("dataset/real_world_jobs_200.csv")
df.fillna("", inplace=True)

# Combine text
df["text"] = (
    df["job_title"] + " " +
    df["company"] + " " +
    df["location"] + " " +
    df["description"]
)

X = df["text"]
y = df["label"]

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=3000
)
X_tfidf = vectorizer.fit_transform(X)

# Train-test split (challenging split)
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Macro F1-score: {f1*100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)

# Save artifacts
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/fake_job_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

with open("model/metrics.json", "w") as f:
    json.dump(
        {
            "accuracy": accuracy,
            "f1_macro": f1,
            "confusion_matrix": conf_matrix.tolist()
        },
        f,
        indent=4
    )

print("✅ Model trained successfully")
