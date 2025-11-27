import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# 1. Load the dataset
data_path = os.path.join("..", "data", "disease_dataset.csv")
df = pd.read_csv(data_path)

print("Original columns:", len(df.columns))

# 2. Drop any unnamed / junk columns (like 'Unnamed: 133')
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print("After removing Unnamed columns:", len(df.columns))

# 3. Separate features (X) and label (y)
#   'prognosis' is the disease column in this dataset
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

print("Number of symptom columns (features):", X.shape[1])

# 4. Split into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Create and train the model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# 6. Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# 7. Save the model and the feature names
models_dir = os.path.join("..", "models")
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, "disease_model.pkl")
columns_path = os.path.join(models_dir, "symptom_columns.pkl")

joblib.dump(model, model_path)
joblib.dump(list(X.columns), columns_path)

print("✅ Model and symptom columns saved successfully!")
print("Model path:", model_path)
