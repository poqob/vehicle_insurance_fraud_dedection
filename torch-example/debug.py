print("Debug script starting")
import torch
print("Torch imported")
print("Torch version:", torch.__version__)
try:
    print("Loading model from ./fraud_detection_model_smote.pth")
    import torch.serialization
    torch.serialization.add_safe_globals(['sklearn.preprocessing._data'])
    checkpoint = torch.load("./fraud_detection_model_smote.pth", weights_only=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
print("Debug script completed")
