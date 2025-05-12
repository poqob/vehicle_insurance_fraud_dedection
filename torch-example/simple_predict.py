import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Model definition
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

# Main function
def main():
    print("Loading model...")
    # Allow sklearn objects to be unpickled
    import torch.serialization
    torch.serialization.add_safe_globals([
        'sklearn.preprocessing._data',
        'sklearn.preprocessing._label',
        'sklearn.preprocessing._encoders'
    ])
    
    model_path = "./fraud_detection_model_smote.pth"
    try:
        checkpoint = torch.load(model_path, weights_only=False)
        print("Model loaded successfully!")
        
        # Create model with correct architecture
        model = FraudNet(checkpoint['input_size'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load data
        print("\nLoading data...")
        df = pd.read_csv("../data/carclaims.csv")
        
        # Prepare data for prediction
        print("Preparing data...")
        df_encoded = df.copy()
        
        # Handle target variable
        df_encoded['FraudFound'] = df_encoded['FraudFound'].map({'Yes': 1, 'No': 0})
        y_true = df_encoded['FraudFound'].values
        
        # Encode categorical features
        for col, encoder in checkpoint['label_encoders'].items():
            if col in df_encoded.columns:
                try:
                    df_encoded[col] = encoder.transform(df_encoded[col])
                except:
                    print(f"Warning: Issue with column {col}")
                    # Handle unseen values
                    for val in df_encoded[col].unique():
                        if val not in encoder.classes_:
                            df_encoded.loc[df_encoded[col] == val, col] = encoder.classes_[0]
                    df_encoded[col] = encoder.transform(df_encoded[col])
        
        # Extract features and scale
        X = df_encoded.drop('FraudFound', axis=1).values
        X_scaled = checkpoint['scaler'].transform(X)
        
        # Make predictions
        print("Making predictions...")
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            y_pred_prob = model(X_tensor)
            y_pred = (y_pred_prob > 0.5).float().numpy().flatten()
        
        # Calculate metrics
        print("\n----- Model Evaluation -----")
        print(f"Total samples: {len(y_true)}")
        print(f"Actual fraud cases: {sum(y_true)}")
        print(f"Predicted fraud cases: {sum(y_pred)}")
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        print("\nFormat: [[TN, FP], [FN, TP]]")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=["Not Fraud", "Fraud"]))
        
        # Save results to simplified CSV
        result_df = pd.DataFrame({
            'Original': df['FraudFound'],
            'Predicted': np.where(y_pred == 1, 'Yes', 'No'),
            'Probability': y_pred_prob.numpy().flatten()
        })
        
        result_df.to_csv("./simple_prediction_results.csv", index=False)
        print("\nResults saved to simple_prediction_results.csv")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
