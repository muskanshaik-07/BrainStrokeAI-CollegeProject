import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "healthcare-dataset-stroke-data.csv"

df = pd.read_csv(DATA_PATH)

# Fill missing BMI values
df["bmi"] = df["bmi"].fillna(df["bmi"].mean())

# Remove ID column
df.drop("id", axis=1, inplace=True)

# Encode categorical columns
encoders = {}

categorical_columns = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save encoders
joblib.dump(encoders, BASE_DIR / "models" / "encoders.pkl")

# Split features and target
X = df.drop("stroke", axis=1)
y = df["stroke"]

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, BASE_DIR / "models" / "scaler.pkl")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)