import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

# Model definition (must match the saved model architecture)
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

def load_model(model_path):
    # Allow sklearn objects to be unpickled
    import torch.serialization
    torch.serialization.add_safe_globals(['sklearn.preprocessing._data'])
    
    # Load the saved model and preprocessing objects with weights_only=False
    checkpoint = torch.load(model_path, weights_only=False)
    
    # Create a model instance with the correct input size
    input_size = checkpoint['input_size']
    model = FraudNet(input_size)
    
    # Load the saved weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    return model, checkpoint['scaler'], checkpoint['label_encoders']

def prepare_data(file_path, label_encoders, scaler):
    # Load data
    df = pd.read_csv(file_path)
    
    # Create a copy of original data for reference
    df_original = df.copy()
    
    # Handle target column if it exists (for evaluation)
    if 'FraudFound' in df.columns:
        df_original['FraudFound_Original'] = df['FraudFound']
        df['FraudFound'] = df['FraudFound'].map({'Yes': 1, 'No': 0})
    
    # Apply label encoding to categorical columns
    for col, encoder in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col])
            except ValueError as e:
                print(f"Error transforming column {col}: {e}")
                # Handle unseen values
                print(f"Original values in training: {encoder.classes_}")
                print(f"Current values in data: {df[col].unique()}")
                # Use a safe approach (replace unseen values)
                for val in df[col].unique():
                    if val not in encoder.classes_:
                        # Replace with most frequent class
                        df.loc[df[col] == val, col] = encoder.classes_[0]
                df[col] = encoder.transform(df[col])
    
    # Extract features
    X = df.drop('FraudFound', axis=1).values if 'FraudFound' in df.columns else df.values
    
    # Apply scaling
    X_scaled = scaler.transform(X)
    
    # Convert to tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    # If target exists, return it for evaluation
    if 'FraudFound' in df.columns:
        y = df['FraudFound'].values
        y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
        return X_tensor, y_tensor, df_original
    else:
        return X_tensor, None, df_original

def make_predictions(model, X_tensor):
    with torch.no_grad():
        y_pred = model(X_tensor)
        y_pred_labels = (y_pred > 0.5).float().numpy()
    return y_pred_labels

def main():
    # Load the model
    model_path = "./fraud_detection_model.pth"
    model, scaler, label_encoders = load_model(model_path)
    
    # Load and prepare the data
    file_path = "../data/carclaims.csv"  # You can change this to a different file if needed
    X_tensor, y_tensor, df_original = prepare_data(file_path, label_encoders, scaler)
    
    # Make predictions
    predictions = make_predictions(model, X_tensor)
    
    # Convert predictions to 'Yes'/'No' format
    pred_labels = np.where(predictions == 1, 'Yes', 'No')
    
    # Add predictions to the original dataframe
    df_original['Predicted'] = pred_labels.flatten()
    
    # Evaluate if actual labels are available
    if y_tensor is not None:
        print("Model Evaluation:")
        y_true = df_original['FraudFound_Original'].values  # Original 'Yes'/'No' values
        y_pred = pred_labels.flatten()
        
        # Print classification report with text labels
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        conf_matrix = confusion_matrix(y_true, y_pred)
        print(conf_matrix)
        
        # Calculate accuracy
        accuracy = (y_true == y_pred).mean()
        print(f"\nAccuracy: {accuracy:.4f}")
    
    # Display sample predictions
    print("\nSample Predictions (first 10 rows):")
    sample_cols = ['Predicted']
    if 'FraudFound_Original' in df_original.columns:
        sample_cols.insert(0, 'FraudFound_Original')
    print(df_original[sample_cols].head(10))
    
    # Show distribution of predictions
    print("\nPrediction Distribution:")
    print(df_original['Predicted'].value_counts())
    
    # Save results to CSV
    output_path = "./predictions_results.csv"
    df_original.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
