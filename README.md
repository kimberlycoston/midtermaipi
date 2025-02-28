# Real-Time ECG Prediction Model

My first machine learning project - an ML powered ECG classification system running on a Raspberry Pi for real-time arrhythmia detection.

## 📌 Table of Contents
Introduction  
Features  
Hardware & Software Requirements  
Installation & Setup  
How It Works  
Challenges  
Next Steps  
Acknowledgments  
License  

## 📝 Introduction
This project is an ECG prediction model deployed on a Raspberry Pi that detects and classifies heart rhythms in real time. The long-term goal is to help with early detection of arrhythmias such as Premature Ventricular Contractions (PVCs), prolonged QT intervals, and more.

### 💡 Why This Matters?

Early arrhythmia detection can prevent the effects of serious heart conditions.  
There are very few commercially available ECG devices for consumer use.  
Existing methods require expensive hardware or cloud computing.  

This project runs locally on a Raspberry Pi for low-cost, real-time processing.

### 🔹 Features
✔️ Collects real-time ECG data using a sensor  
✔️ Uses a machine learning model for classification  
✔️ Audio beep alert for arrhythmias  
✔️ Displays results on an LCD screen  
✔️ Runs entirely on a Raspberry Pi  

## 💻 Hardware & Software Requirements
### 🔧 Hardware:
ECG Sensor Module & Cable Lead Set (Model: AD8232)  
Raspberry Pi (Model: 4)  
LCD Display - DSI Cable  
Button  
USB Speaker  
SPI Communication Module (Model: MCP3008)  
USB Cables, Power Supply (5V Battery Pack), and Breadboard  

![image](https://github.com/user-attachments/assets/c1943440-8afb-480d-b33a-58094940ffde)

### 📦 Software:
Python 3.7+  
PyTorch  
NumPy  
Pandas  
SciPy  
Matplotlib (for continuous/real-time ECG visualization)  
MIT-BIH Arrhythmia Dataset for training data

## ⚙️ Installation & Setup
### Step 1: Clone the Repository
git clone https://github.com/kimberlycoston/midtermaipi.git  
cd midterm_ecg  

### Step 2: Install Dependencies
pip install -r requirements.txt  

### Step 3: Connect the ECGs & Circuit
Connect ECG Leads (1 Right Chest, 1 Left Chest, 1 Left Lower Leg/Left Lower Abdomen)  
Ensure the ECG sensor is plugged into audio jack  
Connect circuit as seen in image above and specific connections for the ADC (MCP3008) below  

<img width="473" alt="image" src="https://github.com/user-attachments/assets/f937173b-8ea5-4168-bcfc-e3ee31e5b76a" />

### Step 4: Run the Model
python train_model.py --> Output will be the ecg_full_model.joblib  
python optimize_model.py --> Output will be ecg_optimized_model.joblib  
python deploy_model.py on Raspberry Pi --> Push button --> You will see Real-Time ECG Arrhythmia Predictions :)  


## 📊 How It Works
Dataset created with CNN  
Optimized dataset to run locally on Raspberry Pi  
Deploy onto Raspberry Pi  
Machine Learning Model classifies the heartbeat into 1 of 5 categories.  
ECG Sensor captures real-time heart signals.  
Output is displayed on an LCD screen.  
Audio beep alert as additional indicator of arrhythmia.  
<img width="444" alt="image" src="https://github.com/user-attachments/assets/0588df66-634a-432f-9195-93611d01b034" />


### 📌 Flowchart of the System:
![image](https://github.com/user-attachments/assets/fe51d63c-fdf4-4657-998c-d1bf8c1ca1f7)


## 🚧 Challenges
🔴 Communication Issue (I2C vs. SPI)  
Originally used I2C, but it was too slow for real-time processing. ECG waveform did not look accurate, tried many different processing techniques before realizing the simple fix of switching to SPI.

🔴 Dataset Structure Issue
Training data was split into single-beat rows, which didn’t align well with real-time streaming data.

🔴 Individual Wave Detection Difficulties
Finding waves dynamically in real-time was harder than using pre-labeled peaks in a dataset.

🔴 Initializing Script on Raspberry Pi Start-Up
The specific part of the LCD Display connected via DSI Ribbon was cumbersome to overcome. For my particicular LCD Display, I had to find the 'dtoverlay=vc4-kms-v3d' in the boot/config.txt fiile and change it to 'dtoverlay=vc4-fkms-v3d'

## 🚀 Next Steps
✔️ Identify and parse all PQRST waves to pinpoint abnormalities  
✔️ Implement a rolling window for better real-time classification  
✔️ Add cloud logging or alert system for abnormal rhythms  
✔️ Explore different ML architectures for improved accuracy  

## 📜 License
This project is licensed under the MIT License – feel free to use and modify it.

