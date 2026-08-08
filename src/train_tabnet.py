import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

from pytorch_tabnet.tab_model import TabNetClassifier

# ----------------------------
# Load Dataset
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "healthcare-dataset-stroke-data.csv"

df = pd.read_csv(DATA_PATH)

# ----------------------------
# Data Preprocessing
# ----------------------------

df["bmi"] = df["bmi"].fillna(df["bmi"].mean())

df.drop("id", axis=1, inplace=True)

encoder = LabelEncoder()

categorical_columns = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

X = df.drop("stroke", axis=1).values
y = df["stroke"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

joblib.dump(scaler, BASE_DIR / "models" / "scaler.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------
# Build TabNet Model
# ----------------------------

model = TabNetClassifier()

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    max_epochs=20,
    patience=5,
    batch_size=256
)

# ----------------------------
# Evaluate
# ----------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nTabNet Accuracy:", accuracy)

# ----------------------------
# Save Model
# ----------------------------

model.save_model(str(BASE_DIR / "models" / "tabnet_model"))

print("\n✅ TabNet Model Saved Successfully!")