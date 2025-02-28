import time
import spidev
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import numpy as np
from scipy.signal import find_peaks
import simpleaudio as sa
import threading
import queue

# SPI Configuration for MCP3008 (ADC for Raspberry Pi)
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1350000

# Function to Read Data from MCP3008 (ECG Sensor)
def read_channel(channel=0):
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((adc[1] & 3) << 8) + adc[2]
    return data

# Beep Sound Function
def play_beep():
    frequency = 1000  # Hz (adjust for desired pitch)
    duration = 0.1  # seconds
    sample_rate = 44100  # Hz
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    audio = (wave * 32767).astype(np.int16)
    sa.play_buffer(audio, 1, 2, sample_rate)

# Initialize variables for real-time plotting
x_len = 500  # Number of samples to display
y_range = [0, 1023]  # Expected ADC range
ecg_data = deque([0] * x_len, maxlen=x_len)  # Buffer for storing ECG values
timestamps = np.arange(-x_len, 0)  # X-axis time window
r_peak_times = deque()  # Store last 10 R-peak timestamps

# Set up Matplotlib figure
plt.style.use('ggplot')
fig, ax = plt.subplots()
ax.set_ylim(y_range)
ax.set_xlim(-x_len, 0)
ax.set_title("Real-Time ECG")
ax.set_xlabel("Time (samples)")
ax.set_ylabel("ECG Amplitude")

# ECG signal and R-peak plots
ecg_line, = ax.plot(timestamps, ecg_data, lw=2, color="blue", label="ECG Signal")
r_peak_points, = ax.plot([], [], 'ro', markersize=6, label="Detected R-Peaks")

def on_key(event):
    if event.key == 'q':
        print("Exit key pressed. Closing ECG plot...")
        plt.close(fig)

fig.canvas.mpl_connect("key_press_event", on_key)

# Create a queue for communication between threads
data_queue = queue.Queue()

# Function to continuously read ECG data and detect R-peaks
def read_ecg_data():
    global ecg_data, r_peak_times
    last_update_time = time.time()

    while True:
        new_value = read_channel(0)
        ecg_data.append(new_value)

        # Detect R-peaks
        filtered_ecg = np.array(ecg_data)
        peaks, _ = find_peaks(filtered_ecg, height=600, distance=30)

        current_time = time.time()

        if len(r_peak_times) > 2:
            last_rr_interval = np.mean(np.diff(list(r_peak_times)[-3:]))
        elif len(r_peak_times) == 2:
            last_rr_interval = current_time - r_peak_times[-1]
        else:
            last_rr_interval = 60 / 120

        min_rr_interval = max(60 / 180, min(60 / 40, last_rr_interval))

        if len(peaks) > 0:
            if len(r_peak_times) == 0 or current_time - r_peak_times[-1] > min_rr_interval:
                r_peak_times.append(current_time)
                play_beep()
                print(f"R-Peak! Interval: {last_rr_interval:.3f} sec | Min Allowed: {min_rr_interval:.3f} sec")

        r_peak_times = deque([t for t in r_peak_times if current_time - t <= 6], maxlen=20)

        # Put the new data in the queue
        data_queue.put((new_value, peaks))

        # Control the sampling rate
        time.sleep(max(0, 0.008 - (time.time() - last_update_time)))
        last_update_time = time.time()

# Update plot function
def update_plot(frame):
    global ecg_data

    # Get new data from the queue
    try:
        while not data_queue.empty():
            new_value, peaks = data_queue.get_nowait()
    except queue.Empty:
        return ecg_line, r_peak_points

    # Update ECG plot
    r_peak_x = [i - x_len for i in peaks if i < len(ecg_data)]
    r_peak_y = [ecg_data[i] for i in peaks if i < len(ecg_data)]
    r_peak_points.set_data(r_peak_x, r_peak_y)
    ecg_line.set_data(timestamps, ecg_data)

    return ecg_line, r_peak_points

# Start the ECG data reading thread
ecg_thread = threading.Thread(target=read_ecg_data, daemon=True)
ecg_thread.start()

# Start real-time animation
ani = FuncAnimation(fig, update_plot, interval=8, blit=True)
plt.legend()
plt.show()

# Close SPI connection when script stops
spi.close()
print("ECG stopped")
