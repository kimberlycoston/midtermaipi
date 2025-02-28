import torch  # Import PyTorch for model operations
torch.backends.quantized.engine = "qnnpack" # Set the quantized engine to 'qnnpack' for optimized mobile inference
import torch.nn as nn  # Import neural network module
import torch.nn.utils.prune as prune  # Import pruning utilities
import torch.quantization  # Import quantization utilities
import joblib  # Import joblib for saving and loading the trained model
from sklearn.metrics import classification_report # Import classification report for performance evaluation
import numpy as np # Import NumPy for numerical operations
import pandas as pd # Import Pandas for handling datasets
from sklearn.model_selection import train_test_split # Import function to split dataset into train/test sets
from torch.utils.data import DataLoader, TensorDataset # Import utilities to create
from ecgclassifier_model import ECGClassifier  # Import model definition

# Load the trained model
model = joblib.load("ecg_full_model.joblib")  # Load the original trained model
model.eval()  # Set model to evaluation mode (disables training behaviors like dropout)
print("Model loaded successfully for optimization.")

#################################################
# Apply pruning and quantization

# Apply structured pruning to convolutional layers
def apply_structured_pruning(model):
    for name, module in model.named_modules():# Iterate through all named modules in the model
        if isinstance(module, nn.Conv1d): # Only prune convolutional layers (1D Convolutions for ECG signals)
            if name not in ["conv1", "conv2"]:  # Exclude conv1 and conv2 to preserve key features
                prune.ln_structured(module, name="weight", amount=0.3, n=2, dim=0)  # Remove 30% of weights using L2 norm pruning
                prune.remove(module, "weight")  # Permanently remove pruned connections
    print("Selective pruning applied successfully (excluding conv1 & conv2 to protect class 1 & 3).")
    return model

model = apply_structured_pruning(model)  # Apply structured pruning to optimize model size

# Apply dynamic quantization to reduce memory usage and improve inference speed
model = torch.quantization.quantize_dynamic(
    model, {model.fc1}, dtype=torch.qint8  # Apply quantization to the first fully connected layer (fc1)
)
print("Quantization applied successfully.")

# Save the optimized model after pruning and quantization
joblib.dump(model, "ecg_optimized_model.joblib")  # Save as new file
print("Optimized model saved as ecg_optimized_model.joblib.")

#################################################
# Reload Test Data for Evaluation


# Load dataset again
data = pd.read_csv('mitbih_combined.csv', header=None) # Load the dataset without column headers

# Separate features and labels
X = data.iloc[:, :-1].values  # Features: Select all columns except the last one (188 features per sample)
y = data.iloc[:, -1].values   # Labels: Select the last column (indicating ECG class)

# Use the same train-test split as in train_model.py
_, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert to PyTorch tensors
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1) # Add extra dimension for 1D Conv input
y_test = torch.tensor(y_test, dtype=torch.long) # Convert labels to long integers (for classification)

# Create DataLoader for batch processing
test_dataset = TensorDataset(X_test, y_test) # Create a dataset
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True) # Load data in batches of 128

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) # Move model to the selected device for efficient computation

#################################################
# Evaluate the Optimized Model


model.eval()  # Set model to evaluation mode
all_preds, all_labels = [], []  # Initialize lists to store predictions and labels

with torch.no_grad():  # Disable gradient calculations for efficiency
    for inputs, labels in test_loader: # Iterate over batches from the test DataLoader
        inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to the same device
        outputs = model(inputs)  # Forward pass through the model
        preds = torch.argmax(outputs, dim=1)  # Get class with highest probability
        all_preds.extend(preds.cpu().numpy())  # Convert predictions to NumPy and store them
        all_labels.extend(labels.cpu().numpy())  # Convert actual labels to NumPy and store them

#################################################
# Generate classification report

target_names = [str(i) for i in range(5)]  # Create class names as strings from 0 to 4
print("Optimized Model Performance:")
print(classification_report(all_labels, all_preds, target_names=target_names))  # Print classification report
