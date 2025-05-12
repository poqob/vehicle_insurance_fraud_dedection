import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    # Add proper globals for sklearn modules
    torch.serialization.add_safe_globals([
        'sklearn.preprocessing._data',
        'sklearn.preprocessing._label',
        'sklearn.preprocessing._encoders'
    ])
    
    # Print debug info
    print(f"Loading model from {model_path}")
    
    # Load the saved model and preprocessing objects with weights_only=False
    checkpoint = torch.load(model_path, weights_only=False)
    print("Model loaded successfully")
    
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
                print(f"Warning: Handling unseen values in column {col}")
                # Handle unseen values
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

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('./confusion_matrix.png')
    print("Confusion matrix plot saved as 'confusion_matrix.png'")

def main():
    # Load the model
    model_path = "./fraud_detection_model_smote.pth"
    model, scaler, label_encoders = load_model(model_path)
    
    # Load and prepare the data
    file_path = "../data/carclaims.csv"  # You can change this to a different file if needed
    X_tensor, y_tensor, df_original = prepare_data(file_path, label_encoders, scaler)
    
    # Make predictions
    with torch.no_grad():
        y_pred_prob = model(X_tensor)
        y_pred_labels = (y_pred_prob > 0.5).float().numpy().flatten()
    
    # Convert predictions to 'Yes'/'No' format
    pred_labels = np.where(y_pred_labels == 1, 'Yes', 'No')
    
    # Add predictions and probabilities to the original dataframe
    df_original['Predicted'] = pred_labels
    df_original['Fraud_Probability'] = y_pred_prob.numpy().flatten()
    
    # Evaluate if actual labels are available
    if y_tensor is not None:
        print("\n==== Model Evaluation ====")
        y_true = df_original['FraudFound_Original'].values  # Original 'Yes'/'No' values
        y_pred = pred_labels
        
        # Print classification report with text labels
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        conf_matrix = confusion_matrix(y_true, y_pred)
        print(conf_matrix)
        
        # Plot confusion matrix
        plot_confusion_matrix(conf_matrix, classes=['No', 'Yes'])
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nAccuracy: {accuracy:.4f}")
        
        # Calculate precision and recall for fraud cases (Yes)
        from sklearn.metrics import precision_score, recall_score
        precision = precision_score(y_true, y_pred, pos_label='Yes')
        recall = recall_score(y_true, y_pred, pos_label='Yes')
        print(f"Precision for Fraud cases: {precision:.4f}")
        print(f"Recall for Fraud cases: {recall:.4f}")
    
    # Display sample predictions
    print("\n==== Sample Predictions (first 10 rows) ====")
    cols_to_show = ['FraudFound_Original', 'Predicted', 'Fraud_Probability']
    print(df_original[cols_to_show].head(10))
    
    # Show distribution of predictions
    print("\n==== Prediction Distribution ====")
    print(df_original['Predicted'].value_counts())
    
    # Find high probability fraud cases
    high_prob_threshold = 0.8
    high_prob_fraud = df_original[df_original['Fraud_Probability'] >= high_prob_threshold]
    print(f"\n==== High Probability Fraud Cases (>= {high_prob_threshold}) ====")
    print(f"Number of high probability fraud cases: {len(high_prob_fraud)}")
    if len(high_prob_fraud) > 0:
        print("Sample of high probability fraud cases:")
        print(high_prob_fraud[cols_to_show].head(5))
    
    # Save results to CSV
    output_path = "./predictions_results_smote.csv"
    df_original.to_csv(output_path, index=False)
    print(f"\nFull results saved to {output_path}")

if __name__ == "__main__":
    main()
