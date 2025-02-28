import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import joblib
import numpy as np
import time
import spidev
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from scipy.signal import find_peaks

sns.set_style("darkgrid")

# ✅ Define ECGClassifier class (must match original definition)
class ECGClassifier(nn.Module):
    def __init__(self):
        super(ECGClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5)

        # Dynamically calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 187)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            flattened_size = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, 5)  # Assuming 5 classes based on dataset

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ✅ Load saved model and label encoder (ensure consistency with preprocessing)
model_loaded = joblib.load("ecg_full_model.joblib")
model_loaded.eval()

print("Original model and label encoder loaded successfully on Raspberry Pi.")

# ✅ Apply Structured Pruning
def apply_structured_pruning(model):
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            prune.ln_structured(module, name="weight", amount=0.3, n=2, dim=0)
            prune.remove(module, "weight")
    print("Structured pruning applied successfully.")
    return model

model_pruned = apply_structured_pruning(model_loaded)

# ✅ Save pruned model
joblib.dump(model_pruned, "ecg_full_model_pruned.joblib")
print("Pruned model saved as ecg_full_model_pruned.joblib.")

# ✅ SPI Configuration for MCP3008
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1350000

# ✅ Read data from MCP3008
def read_channel(channel=0):
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((adc[1] & 3) << 8) + adc[2]
    return data

# ✅ Heart Rate Calculation
def calculate_heart_rate(ecg_signal, sampling_rate=125):
    """
    Calculates heart rate from a 6-second ECG segment.
    Args:
        ecg_signal (np.ndarray): The ECG signal (1D array).
        sampling_rate (int): Sampling rate of the ECG signal (default: 125 Hz).
    Returns:
        float: Heart rate in BPM.
    """
    # Detect R-peaks
    peaks, _ = find_peaks(ecg_signal, distance=50, height=np.mean(ecg_signal) + 0.5 * np.std(ecg_signal))

    # Calculate heart rate
    num_peaks = len(peaks)
    heart_rate = num_peaks * (60 / 6)  # BPM (6-second window)
    return heart_rate

# Mapping from numerical classes to arrhythmia names
class_mapping = {
    0: "Normal",
    1: "Supraventricular Beat",
    2: "Ventricular Beat",
    3: "Fusion",
    4: "Unknown - Other"
}

# ✅ Real-time ECG visualization and prediction setup
plt.style.use('ggplot')
fig, ax = plt.subplots()
x_len = 200
y_range = [0, 1023]
ecg_line, = ax.plot([], [], lw=2)
abnormal_points, = ax.plot([], [], 'ro', label='Abnormal Beat', markersize=10)  # Larger markers for visibility
ax.set_ylim(y_range)
ax.set_xlim(0, x_len)
ax.set_title('Real-Time ECG with Arrhythmia Prediction')
ax.set_xlabel('Sample')
ax.set_ylabel('Amplitude')
ax.legend()
ax.grid()

data = [0] * x_len
abnormal_indices = []
prediction_window = []
window_size = 187

def init():
    ecg_line.set_data(range(x_len), data)
    abnormal_points.set_data([], [])
    return ecg_line, abnormal_points

def update(frame):
    global prediction_window, abnormal_indices
    new_value = read_channel(0)
    data.append(new_value)
    data.pop(0)
    ecg_line.set_ydata(data)

    prediction_window.append(new_value)
    if len(prediction_window) == window_size:
        with torch.no_grad():
            ecg_tensor = torch.tensor([prediction_window], dtype=torch.float32).unsqueeze(1)
            output = model_loaded(ecg_tensor)
            pred = torch.argmax(output, dim=1).item()
            # Use the mapping in the update function
            predicted_class_name = class_mapping.get(pred, "Gathering data...")
            print(f"Predicted Arrhythmia: {predicted_class_name}")

            # ✅ Update text below the graph
            pred_text.set_text(f"Predicted Arrhythmia: {predicted_class_name}")

            # Highlight abnormal beats if prediction is not '0' (normal)
            if pred != '0.0':
                # Add the index of the last sample in the prediction window
                abnormal_indices.append(len(data) - 1)

            # Calculate heart rate
            heart_rate = calculate_heart_rate(np.array(prediction_window))
            print(f"Heart Rate: {heart_rate:.2f} BPM")

        prediction_window = []

    # Clear old abnormal indices to avoid overplotting
    if len(abnormal_indices) > 5:  # Keep only the last 5 abnormal beats
        abnormal_indices = abnormal_indices[-5:]

    # Update abnormal beat highlights
    abnormal_y = [data[i] for i in abnormal_indices if i < len(data)]
    abnormal_x = [i for i in abnormal_indices if i < len(data)]
    abnormal_points.set_data(abnormal_x, abnormal_y)

    return ecg_line, abnormal_points

# Create a text label below the graph for displaying the predicted arrhythmia
# Create a global figtext for the predicted arrhythmia
plt.subplots_adjust(bottom=0.3)  # Increase bottom margin

pred_text = plt.figtext(0.5, 0.1, "Predicted Arrhythmia: Gathering data...", ha='center', va='center', fontsize=12, color='Red', fontweight='bold')
ani = FuncAnimation(fig, update, init_func=init, blit=False, interval=5)

try:
    plt.show()
except KeyboardInterrupt:
    spi.close()
    print("Plotting and prediction stopped by user.")

spi.close()
print("SPI connection closed successfully.")