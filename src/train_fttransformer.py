import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# -----------------------
# Load Dataset
# -----------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "healthcare-dataset-stroke-data.csv"

df = pd.read_csv(DATA_PATH)

# -----------------------
# Preprocessing
# -----------------------

df["bmi"] = df["bmi"].fillna(df["bmi"].mean())
df.drop("id", axis=1, inplace=True)

encoder = LabelEncoder()

categorical = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

for col in categorical:
    df[col] = encoder.fit_transform(df[col])

X = df.drop("stroke", axis=1).values
y = df["stroke"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Convert to Torch tensors

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1,1)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=64,
    shuffle=True
)

# -----------------------
# FT-Transformer Model
# -----------------------

class FTTransformer(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.embedding = nn.Linear(input_dim,64)

        encoder = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=8,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder,
            num_layers=2
        )

        self.fc = nn.Linear(64,1)

    def forward(self,x):

        x = self.embedding(x)

        x = x.unsqueeze(1)

        x = self.transformer(x)

        x = x.squeeze(1)

        x = self.fc(x)

        return torch.sigmoid(x)

# -----------------------
# Training
# -----------------------

model = FTTransformer(X_train.shape[1])

criterion = nn.BCELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 20

for epoch in range(epochs):

    model.train()

    for xb,yb in train_loader:

        optimizer.zero_grad()

        pred = model(xb)

        loss = criterion(pred,yb)

        loss.backward()

        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} Loss:{loss.item():.4f}")

# -----------------------
# Testing
# -----------------------

model.eval()

with torch.no_grad():

    pred = model(X_test)

pred = (pred>0.5).float()

accuracy = accuracy_score(
    y_test.numpy(),
    pred.numpy()
)

print("\nFT-Transformer Accuracy:",accuracy)

# -----------------------
# Save Model
# -----------------------

torch.save(
    model.state_dict(),
    BASE_DIR/"models"/"ft_transformer.pth"
)

print("\n✅ FT-Transformer Model Saved Successfully!")