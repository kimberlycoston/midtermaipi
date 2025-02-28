import torch  # Import PyTorch for model operations
torch.backends.quantized.engine = "qnnpack"
import torch.nn as nn  # Import neural network module
import torch.nn.utils.prune as prune  # Import pruning utilities
import torch.quantization  # Import quantization utilities
import joblib  # Import joblib for saving and loading the trained model
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from ecgclassifier_model import ECGClassifier  # Import model definition

# Load the trained model
model = joblib.load("ecg_full_model.joblib")  # Load the original trained model
model.eval()  # Set model to evaluation mode
print("Model loaded successfully for optimization.")

# Apply structured pruning to convolutional layers
def apply_structured_pruning(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d): # Only prune convolutional layers
            if name not in ["conv1", "conv2"]:  # Skip pruning fully connected layers
                prune.ln_structured(module, name="weight", amount=0.3, n=2, dim=0)  # Prune 30% of weights
                prune.remove(module, "weight")  # Permanently remove pruned connections
    print("Selective pruning applied successfully (excluding conv1 & conv2 to protect class 1 & 3).")
    return model

model = apply_structured_pruning(model)  # Apply pruning

# Apply quantization
model = torch.quantization.quantize_dynamic(
    model, {model.fc1}, dtype=torch.qint8  # Applies dynamic quantization to Linear layers
)
print("Quantization applied successfully.")

# Save optimized model
joblib.dump(model, "ecg_optimized_model.joblib")  # Save pruned and quantized model
print("Optimized model saved as ecg_optimized_model.joblib.")

# ==============================
# Reload Test Data for Evaluation
# ==============================

# Load dataset again
data = pd.read_csv('mitbih_combined.csv', header=None)

# Separate features and labels
X = data.iloc[:, :-1].values  # Features (188 columns)
y = data.iloc[:, -1].values   # Labels (last column)

# Use the same train-test split as in train_model.py
_, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert to PyTorch tensors
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader for batch processing
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ==============================
# Evaluate the Optimized Model
# ==============================

model.eval()  # Set model to evaluation mode
all_preds, all_labels = [], []  # Initialize lists to store predictions and labels

with torch.no_grad():  # Disable gradient calculations for efficiency
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move batch data to the selected device
        outputs = model(inputs)  # Compute model output
        preds = torch.argmax(outputs, dim=1)  # Get the predicted class with highest probability
        all_preds.extend(preds.cpu().numpy())  # Store predictions
        all_labels.extend(labels.cpu().numpy())  # Store actual labels

# Generate classification report
target_names = [str(i) for i in range(5)]  # Create class names as strings from 0 to 4
print("Optimized Model Performance:")
print(classification_report(all_labels, all_preds, target_names=target_names))  # Print classification report
