import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt

# Veri setini yükle
df = pd.read_csv("../data/carclaims.csv")

# Hedef sütunu sayısallaştır
df['FraudFound'] = df['FraudFound'].map({'Yes': 1, 'No': 0})

# Tüm object (kategorik) sütunları encode et
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Özellikler ve hedef ayrımı
X = df.drop('FraudFound', axis=1).values
y = df['FraudFound'].values

# Veriyi ayır
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalizasyon
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SMOTE ile veri dengeleme
print("Before SMOTE - Class distribution:", np.bincount(y_train))
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("After SMOTE - Class distribution:", np.bincount(y_train_smote))

# PyTorch tensor'ları
X_train_tensor = torch.tensor(X_train_smote, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_smote.reshape(-1, 1), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

# Model
class FraudNet(nn.Module):
    def __init__(self, input_size):
        super(FraudNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = FraudNet(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Eğitim
epochs = 30
for epoch in range(epochs):
    model.train()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

# Test
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor)
    y_pred_labels = (y_pred_test > 0.5).float().numpy().flatten()
    y_test_np = y_test_tensor.numpy().flatten()
    
    # Calculate accuracy
    acc = accuracy_score(y_test_np, y_pred_labels)
    print(f"Test Accuracy: {acc:.4f}")
    
    # Print full classification report and confusion matrix
    print("\nClassification Report:")
    print(classification_report(y_test_np, y_pred_labels, 
                               target_names=["Not Fraud", "Fraud"]))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test_np, y_pred_labels)
    print("\nConfusion Matrix:")
    print(cm)

# Save the trained model
model_path = "./fraud_detection_model_smote.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'label_encoders': label_encoders,
    'input_size': X_train_smote.shape[1]
}, model_path)
print(f"\nModel saved to {model_path}")
