import spidev
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

sns.set_style("darkgrid")

# SPI Configuration for MCP3008
spi = spidev.SpiDev()
spi.open(0, 0)  # Bus 0, CE0
spi.max_speed_hz = 1350000

def read_channel(channel=0):
    """Read data from MCP3008 on specified channel (0-7)."""
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((adc[1] & 3) << 8) + adc[2]
    return data

# Initialize plot
plt.style.use('ggplot')
fig, ax = plt.subplots()
x_len = 200            # Width of the displayed ECG (number of data points)
y_range = [0, 1023]    # MCP3008 10-bit range (0-1023)
ecg_line, = ax.plot([], [], lw=2)
ax.set_ylim(y_range)
ax.set_xlim(0, x_len)
ax.set_title('Real-Time ECG Signal')
ax.set_xlabel('Sample')
ax.set_ylabel('Amplitude')
ax.grid()

# Data storage
data = [0] * x_len

def init():
    ecg_line.set_data(range(x_len), data)
    return ecg_line,

def update(frame):
    """Fetch new data point and update the plot."""
    data.append(read_channel(0))
    data.pop(0)
    ecg_line.set_ydata(data)
    return ecg_line,

# Real-time plot animation
ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=5)

try:
    plt.show()
except KeyboardInterrupt:
    spi.close()
    print("Plotting stopped by user.")

# Close SPI safely
spi.close()
