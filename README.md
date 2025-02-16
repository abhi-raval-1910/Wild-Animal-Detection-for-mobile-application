# Wild-Animal-Detection-for-mobile-application
Intelligent Wild Animal monitoring system using Raspberry Pi 4 plus Camera plus Dashboard, using YOLOv8 and improving efficiency and accuracy Mobilevnet3
# 🐾 Wild Animal Detection Dashboard

This project is a **Wild Animal Detection Dashboard** that uses **YOLOv8** for object detection and **MobileNetV3** for fine-grained classification. The system detects wild animals in real-time using a webcam feed and displays the results on a **PyQt5-based GUI**. When an animal is detected, an alarm is triggered, and the detection is logged with a timestamp.

## 🚀 Features
✅ Real-time wild animal detection using **YOLOv8**.  
✅ Fine-grained classification using **MobileNetV3**.  
✅ **PyQt5-based GUI** for displaying the video feed and detection logs.  
✅ Alarm system using **Pygame** for audio alerts.  
✅ Continuous updating of detected animals with bounding boxes drawn around them.  

## 🛠 Requirements
- Python 3.7+
- OpenCV
- PyTorch
- TensorFlow
- Pygame
- PyQt5
- Ultralytics YOLO
- TensorFlow Keras Applications
- Timm (PyTorch Image Models)

## 🔧 Installation
```sh
# Clone the repository
git clone https://github.com/your-username/wild-animal-detection.git
cd wild-animal-detection

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Ensure you have a 'buzzer.wav' file in the project directory for the alarm sound.
```

## ▶️ Usage
```sh
python main.py
```
The **PyQt5 GUI** will open, displaying the live video feed from your webcam.  
- When a **wild animal is detected**, an **alarm will sound**, and the detection will be **logged with a timestamp**.

## 📜 Code Overview

### 🔹 **Main Components**
- **WildAnimalDetector Class**: Sets up the PyQt5 GUI, handles the video feed, and runs the detection logic.
- **YOLOv8 Model**: Used for object detection.
- **MobileNetV3 Model**: Used for fine-grained classification of detected animals.
- **Pygame**: Used for playing the alarm sound.

### 🔹 **Key Functions**
- `update_frame()`: Updates the GUI with the latest video frame.
- `trigger_alarm(animal)`: Triggers an alarm and updates the UI when an animal is detected.
- `reset_alert()`: Resets the alert button after 3 seconds.
- `classify_animal(cropped_img)`: Classifies the cropped image using MobileNetV3.
- `run_detection()`: Runs YOLOv8 detection on the live feed.

## 🌍 Use Case
This project can be used for real-time **wildlife monitoring** in various environments such as:
- 🏞 **Forests**
- 🦁 **National Parks**
- 📡 **Wildlife Sanctuaries**

It helps in detecting and identifying wild animals, **triggering alarms** for potential threats, and **logging detections** for further analysis.

## 🖥️ Implementing in VS Code
### Step 1: Install VS Code
Download and install **[Visual Studio Code](https://code.visualstudio.com/)**.

### Step 2: Install Python Extension
In VS Code, go to **Extensions (Ctrl+Shift+X)** and install the **Python Extension**.

### Step 3: Open the Project
```sh
cd wild-animal-detection
code .
```
This will open the project folder in VS Code.

### Step 4: Set Up the Virtual Environment
If you haven't already:
```sh
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 5: Run the Code
In VS Code, open `main.py`, then press **F5** or run:
```sh
python main.py
```

### Step 6: Debugging & Logs
Use the built-in **Debugger (F5)** in VS Code for troubleshooting and logging outputs.

## 🤝 Contributing
Contributions are welcome! 🚀 If you find any issues or have improvements, please **open an issue** or **submit a pull request**.

## 📄 License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---
✨ **Happy Coding!** ✨
