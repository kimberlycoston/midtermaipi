import torch  # Import PyTorch for building and training the model
import torch.nn as nn  # Import neural network module
import torch.nn.functional as F  # Import functional module for activation functions
import joblib  # Import joblib for saving and loading the trained model
import numpy as np  # Import NumPy for numerical operations
import time  # Import time module for measuring execution time
import spidev  # Import SPI communication module for Raspberry Pi
import matplotlib
import seaborn as sns  # Import seaborn for aesthetic plots
from scipy.signal import find_peaks # Import peak detection function for ECG signal analysis
from scipy.interpolate import interp1d
import RPi.GPIO as GPIO  # Import GPIO library
import threading
import simpleaudio as sa  # Ensure simpleaudio is installed: `pip install simpleaudio`
import os
import subprocess
print("Current Working Directory:", os.getcwd())

os.environ['DISPLAY'] = ':0'
os.environ['XAUTHORITY'] = '/home/kimberlyaipi/.Xauthority'

# Activate the virtual environment
# Activate virtual environment
venv_path = "/home/kimberlyaipi/midterm_ecg/venv/bin/activate"
subprocess.call(f"source {venv_path} && python3 /home/kimberlyaipi/midterm_ecg/deploy_model.py", shell=True)

matplotlib.use('TkAgg')  # Use a for GUI display 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  # Import animation module for real-time plotting


# Setup button
BUTTON_PIN = 5  # Define the GPIO pin where the button is connected

# Setup GPIO
GPIO.setmode(GPIO.BCM)  # Use BCM numbering
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Set as input with pull-down resistor

print("Press the button to start ECG monitoring...")
while GPIO.input(BUTTON_PIN) == GPIO.HIGH:
    time.sleep(0.1)
print("Button pressed! Starting ECG monitoring...")


# Function to generate and play a beep sound
def play_beep():
    frequency = 1000  # Hz (adjust for desired pitch)
    duration = 0.1  # seconds
    sample_rate = 44100  # Hz
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    audio = (wave * 32767).astype(np.int16)
    sa.play_buffer(audio, 1, 2, sample_rate)


def resize_ecg(ecg_signal, target_size=187):
    x_old = np.linspace(0, len(ecg_signal) - 1, len(ecg_signal))
    x_new = np.linspace(0, len(ecg_signal) - 1, target_size)
    interpolator = interp1d(x_old, ecg_signal, kind='linear')
    return interpolator(x_new)

sns.set_style("darkgrid")  # Set seaborn plot style to dark grid

# Define ECGClassifier class (must match original definition)
class ECGClassifier(nn.Module):
    def __init__(self):
        super(ECGClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5)  # First convolutional layer
        self.pool = nn.MaxPool1d(2)  # Max pooling layer
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5)  # Second convolutional layer

        # Dynamically calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 187)  # Create a dummy input to compute size
            x = self.pool(F.relu(self.conv1(dummy_input)))  # Apply first conv layer
            x = self.pool(F.relu(self.conv2(x)))  # Apply second conv layer
            flattened_size = x.view(1, -1).shape[1]  # Compute output size for FC layer

        self.fc1 = nn.Linear(flattened_size, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 5)  # Output layer for classification (5 classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Apply first conv + pooling
        x = self.pool(F.relu(self.conv2(x)))  # Apply second conv + pooling
        x = x.view(x.size(0), -1)  # Flatten the feature maps
        x = F.relu(self.fc1(x))  # Fully connected layer with ReLU activation
        return self.fc2(x)  # Output logits

# Load saved model and set to evaluation mode
model_loaded = joblib.load("/home/kimberlyaipi/midterm_ecg/ecg_optimized_model.joblib")
model_loaded.eval()  # Set model to evaluation mode
print("Optimized model loaded successfully on Raspberry Pi.")

# SPI Configuration for MCP3008 (ADC for Raspberry Pi)
spi = spidev.SpiDev()  # Initialize SPI device
spi.open(0, 0)  # Open SPI bus
spi.max_speed_hz = 1350000  # Set SPI speed

# Read data from MCP3008


# Function to detect exit button press
def wait_for_exit():
    while GPIO.input(BUTTON_PIN) == GPIO.HIGH:
        time.sleep(0.1)  # Prevent CPU overuse
    print("Exit button pressed. Stopping ECG monitoring...")
    plt.close()  # Close the Matplotlib window
    spi.close()  # Close SPI connection
    GPIO.cleanup()  # Reset GPIO pins
    exit()  # Exit the script cleanly

def read_channel(channel=0):
    adc = spi.xfer2([1, (8 + channel) << 4, 0])  # Send read command to MCP3008
    data = ((adc[1] & 3) << 8) + adc[2]  # Convert received bytes to integer
    return data

# Mapping from numerical classes to arrhythmia names
class_mapping = {
    0: "Normal",
    1: "Supraventricular Beat",
    2: "Ventricular Beat",
    3: "Fusion",
    4: "Unknown - Other"
}

# Real-time ECG visualization setup
plt.style.use('ggplot')  # Set plot style
fig, ax = plt.subplots()
x_len = 200  # Number of samples displayed in real-time
ax.set_ylim([0, 1023])  # Set Y-axis range
ax.set_xlim(0, x_len)  # Set X-axis range
ecg_line, = ax.plot([], [], lw=2)  # ECG waveform line
abnormal_points, = ax.plot([], [], 'ro', markersize=10)  # Abnormal beat markers
ax.set_title('Real-Time ECG with Arrhythmia Prediction')
ax.set_xlabel('Sample')
ax.set_ylabel('Amplitude')
ax.legend()
ax.grid()
data = [0] * x_len
abnormal_indices = []
prediction_window = []
window_size = 50

def init():
    ecg_line.set_data(range(x_len), data)
    abnormal_points.set_data([], [])
    return ecg_line, abnormal_points, pred_text

def upsample_ecg(ecg_signal, original_rate=50, target_rate=125):
    x_old = np.linspace(0, len(ecg_signal) -1, len(ecg_signal))
    x_new = np.linspace(0, len(ecg_signal) -1, int(len(ecg_signal) * (target_rate / original_rate)))
    interpolator = interp1d(x_old, ecg_signal, kind='linear')
    return interpolator(x_new)

def update(frame):
    global prediction_window, abnormal_indices
    new_value = read_channel(0)  # Read ECG data from SPI
    data.append(new_value)
    data.pop(0)
    ecg_line.set_ydata(data)

   

    prediction_window.append(new_value)


    if len(prediction_window) >= 75:  # We now have enough 50Hz samples
        # 🚨 Instead of upsampling, we resize to match 187 samples
        ecg_window = np.array(prediction_window, dtype=np.float32)
        ecg_window = resize_ecg(ecg_window, 187)  # Resizes signal to model input size

        if len(ecg_window) == 187:  # Ensure correct length before passing to model
            print("ECG Input Shape:", ecg_window.shape)  # Debugging print

            with torch.no_grad():
                ecg_tensor = torch.tensor([ecg_window], dtype=torch.float32).unsqueeze(1)  # Add batch + channel dims
                output = model_loaded(ecg_tensor)
                pred = torch.argmax(output, dim=1).item()

                predicted_class_name = class_mapping.get(pred, "Gathering data...")
                print(f"Predicted Rhythm: {predicted_class_name}")
                print(f"Model Output Probabilities: {output.numpy()}")


                pred_text.set_text(f"Predicted Rhythm: {predicted_class_name}")

                # Beep only if an arrhythmia is detected (anything other than Normal)
                if pred != 0:
                    play_beep()

            if pred != 0:
                abnormal_indices.append(len(data) - 1)

        prediction_window = []  # Reset the window after prediction

    # Update abnormal points on the plot
    abnormal_y = [data[i] for i in abnormal_indices if i < len(data)]
    abnormal_x = [i for i in abnormal_indices if i < len(data)]
    abnormal_points.set_data(abnormal_x, abnormal_y)

    return ecg_line, abnormal_points, pred_text



plt.subplots_adjust(bottom=0.3)

pred_text = ax.text(0.5, 0.1, "Predicted Rhythm: Gathering data...", transform=ax.transAxes, ha="center", fontsize=12, color='Red')
#pred_text = plt.figtext(0.5, 0.1, "Predicted Arrhythmia: Gathering data...", ha='center', fontsize=12, color='Red')
ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=5)


# Start exit button listener in a separate thread
exit_thread = threading.Thread(target=wait_for_exit, daemon=True)
exit_thread.start()

try:
    plt.show()
except KeyboardInterrupt:
    spi.close()
    GPIO.cleanup()
    print("Plotting and prediction stopped by user.")

spi.close()
print("SPI connection closed successfully.")
