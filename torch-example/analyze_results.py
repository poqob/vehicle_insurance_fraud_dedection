import pandas as pd
import numpy as np

try:
    print("Loading predictions_results_smote.csv...")
    df = pd.read_csv("./predictions_results_smote.csv")
    
    print(f"\nData shape: {df.shape}")
    
    # Check if the required columns exist
    if 'FraudFound_Original' in df.columns and 'Predicted' in df.columns:
        print("\nAnalyzing prediction results:")
        print(f"Total samples: {len(df)}")
        
        # Count actual fraud cases
        actual_fraud = (df['FraudFound_Original'] == 'Yes').sum()
        print(f"Actual fraud cases: {actual_fraud} ({actual_fraud/len(df)*100:.2f}%)")
        
        # Count predicted fraud cases
        pred_fraud = (df['Predicted'] == 'Yes').sum()
        print(f"Predicted fraud cases: {pred_fraud} ({pred_fraud/len(df)*100:.2f}%)")
        
        # Calculate confusion matrix metrics
        tp = ((df['FraudFound_Original'] == 'Yes') & (df['Predicted'] == 'Yes')).sum()
        tn = ((df['FraudFound_Original'] == 'No') & (df['Predicted'] == 'No')).sum()
        fp = ((df['FraudFound_Original'] == 'No') & (df['Predicted'] == 'Yes')).sum()
        fn = ((df['FraudFound_Original'] == 'Yes') & (df['Predicted'] == 'No')).sum()
        
        print("\nConfusion Matrix:")
        print(f"True Positives (correctly identified fraud): {tp}")
        print(f"True Negatives (correctly identified non-fraud): {tn}")
        print(f"False Positives (incorrectly flagged as fraud): {fp}")
        print(f"False Negatives (missed fraud cases): {fn}")
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print("\nPerformance Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (fraud): {precision:.4f}")
        print(f"Recall (fraud): {recall:.4f}")
        print(f"F1 Score (fraud): {f1:.4f}")
        
        # Sample of high probability fraud predictions
        if 'Fraud_Probability' in df.columns:
            high_prob = df[df['Fraud_Probability'] > 0.8]
            print(f"\nHigh probability fraud cases (>80%): {len(high_prob)}")
            if len(high_prob) > 0:
                print("\nSample high probability cases:")
                display_cols = ['FraudFound_Original', 'Predicted', 'Fraud_Probability']
                print(high_prob[display_cols].head(10))
    else:
        print("Required columns not found in the CSV file")
        print("Available columns:", df.columns.tolist())
except Exception as e:
    print(f"Error analyzing results: {e}")
