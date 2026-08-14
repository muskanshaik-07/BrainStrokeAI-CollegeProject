import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ----------------------------
# Load Dataset
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "healthcare-dataset-stroke-data.csv"

df = pd.read_csv(DATA_PATH)

# ----------------------------
# Data Preprocessing
# ----------------------------

# Fill missing BMI values
df["bmi"] = df["bmi"].fillna(df["bmi"].mean())

# Remove ID column
df.drop("id", axis=1, inplace=True)

# Encode categorical columns
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

# ----------------------------
# Build Deep Neural Network
# ----------------------------

model = Sequential()

# Input + Hidden Layer 1
model.add(Dense(64, activation="relu", input_shape=(X_train.shape[1],)))

# Hidden Layer 2
model.add(Dense(32, activation="relu"))

# Output Layer
model.add(Dense(1, activation="sigmoid"))

# ----------------------------
# Compile Model
# ----------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ----------------------------
# Train Model
# ----------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

# ----------------------------
# Evaluate Model
# ----------------------------

loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", accuracy)

# ----------------------------
# Save Model
# ----------------------------

model.save(BASE_DIR / "models" / "dnn_model.keras")

print("\n✅ DNN Model Saved Successfully!")