import pandas as pd  # Import pandas for data manipulation and analysis
import torch  # Import PyTorch for building and training the model
import joblib  # Import joblib for saving and loading the trained model
from sklearn.model_selection import train_test_split  # Import function for splitting dataset into train and test sets
from sklearn.preprocessing import LabelEncoder  # Import label encoder for encoding categorical labels (not used here but useful for future use)
from sklearn.metrics import classification_report  # Import function for generating a classification report
import torch.nn as nn  # Import PyTorch's neural network module
import torch.nn.functional as F  # Import functional module for activation functions
import torch.optim as optim  # Import optimization module for training the model
from torch.utils.data import DataLoader, TensorDataset  # Import utilities for creating dataset loaders
import time  # Import time module (not used explicitly but useful for measuring execution time)

#################################################
# Load dataset
data = pd.read_csv('mitbih_combined.csv', header=None)  # Load the ECG dataset from a CSV file without headers

# Separate features and labels
X = data.iloc[:, :-1].values  # Select all columns except the last as features (188 features per sample)
y = data.iloc[:, -1].values   # Select the last column as the class label

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data into 80% training and 20% testing

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Convert training data to tensor and add a channel dimension for CNN
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)  # Convert test data to tensor and add a channel dimension
y_train = torch.tensor(y_train, dtype=torch.long)  # Convert training labels to tensor
Y_test = torch.tensor(y_test, dtype=torch.long)  # Convert test labels to tensor (correct variable should be 'y_test')

#################################################
# Create PyTorch DataLoader
train_dataset = TensorDataset(X_train, y_train)  # Create dataset object for training data
test_dataset = TensorDataset(X_test, y_test)  # Create dataset object for testing data

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Create DataLoader for training with batch size 64 and shuffling enabled
test_loader = DataLoader(test_dataset, batch_size=64)  # Create DataLoader for testing with batch size 64 (no shuffling needed)

#################################################
# Build the CNN Model for 5-class classification

# Define CNN model with dynamic calculation for fully connected layer input size
class ECGClassifier(nn.Module):
    def __init__(self):
        super(ECGClassifier, self).__init__()  # Initialize the parent class
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5)  # First convolutional layer (1 input channel, 32 output channels, kernel size 5)
        self.pool = nn.MaxPool1d(2)  # Max pooling layer with kernel size 2
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5)  # Second convolutional layer (32 input channels, 64 output channels, kernel size 5)
        
        # Dynamically calculate flattened size after conv and pool layers
        with torch.no_grad():  # Disable gradient calculation for efficiency
            dummy_input = torch.zeros(1, 1, 187)  # Create a dummy input tensor with shape (1 sample, 1 channel, 187 features)
            x = self.pool(F.relu(self.conv1(dummy_input)))  # Apply first conv layer, ReLU activation, and pooling
            x = self.pool(F.relu(self.conv2(x)))  # Apply second conv layer, ReLU activation, and pooling
            flattened_size = x.view(1, -1).shape[1]  # Compute flattened size for fully connected layer input

        self.fc1 = nn.Linear(flattened_size, 128)  # Fully connected layer with 128 output features
        self.fc2 = nn.Linear(128, 5)  # Output layer with 5 classes (assuming the dataset has 5 classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Apply first convolution, ReLU, and max pooling
        x = self.pool(F.relu(self.conv2(x)))  # Apply second convolution, ReLU, and max pooling
        x = x.view(x.size(0), -1)  # Flatten the feature maps before feeding into fully connected layer
        x = F.relu(self.fc1(x))  # Apply ReLU activation to first fully connected layer
        return self.fc2(x)  # Output predictions (no activation needed, as CrossEntropyLoss expects raw logits)
    
# Instantiate the model
model = ECGClassifier()  # Create an instance of the ECGClassifier model
print(model)  # Print model architecture

#################################################
# Train the Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available, otherwise use CPU
model.to(device)  # Move the model to the selected device
criterion = nn.CrossEntropyLoss()  # Define loss function (cross-entropy loss for multi-class classification)
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Define optimizer (Adam optimizer with learning rate 0.001)

epochs = 10  # Number of training epochs
for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0  # Initialize running loss
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move batch data to the selected device
        optimizer.zero_grad()  # Zero the gradients to prevent accumulation
        outputs = model(inputs)  # Forward pass: compute model output
        loss = criterion(outputs, labels)  # Compute loss between predicted and actual labels
        loss.backward()  # Backward pass: compute gradients
        optimizer.step()  # Update model weights based on computed gradients
        running_loss += loss.item()  # Accumulate batch loss
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")  # Print loss for the epoch

#################################################
# Evaluate the model

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
print(classification_report(all_labels, all_preds, target_names=target_names))  # Print classification report

#################################################
# Save the model via joblib

joblib.dump((model), "ecg_full_model.joblib")  # Save the trained model as a joblib file
print("Model saved successfully as ecg_full_model.joblib.")  # Confirm model saving
