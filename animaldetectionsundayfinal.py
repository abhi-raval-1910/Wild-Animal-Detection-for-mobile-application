import cv2
import torch
import time
import threading
import numpy as np
import tensorflow as tf
import pygame
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QListWidget, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input, decode_predictions

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Initialize Pygame for Alarm
pygame.mixer.init()
buzzer_sound = "buzzer.wav"

# Load Models
model = YOLO("yolov8x.pt")  # YOLOv8 Large Model
classifier = MobileNetV3Large(weights="imagenet")  # MobileNetV3 for fine classification

# Define Wild Animal List & Correction Mapping
wild_animals = ["dog", "cat", "bear", "deer", "elephant", "tiger", "leopard", "lion", "wolf"]
correction_map = {"dog": "lion", "cat": "leopard"}

# Open Webcam
cap = cv2.VideoCapture(0)

class WildAnimalDetector(QWidget):
    def __init__(self):
        super().__init__()

        # UI Setup
        self.setWindowTitle("Wild Animal Detector")
        self.setGeometry(100, 100, 900, 600)

        self.layout = QVBoxLayout()

        # Live Feed
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)
        self.layout.addWidget(self.video_label)

        # Animal Detection Log
        self.log_list = QListWidget(self)
        self.layout.addWidget(self.log_list)

        # Detection Alert
        self.alert_button = QPushButton("NO ANIMAL", self)
        self.alert_button.setStyleSheet("background-color: green; color: white; font-size: 16px;")
        self.layout.addWidget(self.alert_button)

        self.setLayout(self.layout)

        # Timer for updating UI
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 milliseconds

        # Start Detection Thread
        self.detection_thread = threading.Thread(target=self.run_detection, daemon=True)
        self.detection_thread.start()

    def update_frame(self):
        """Update the UI with the latest video frame"""
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def trigger_alarm(self, animal):
        """Trigger alarm and update UI when an animal is detected"""
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        message = f"{timestamp} - {animal} detected!"
        print(message)

        self.log_list.addItem(message)
        self.log_list.scrollToBottom()

        pygame.mixer.Sound(buzzer_sound).play()

        self.alert_button.setText("ANIMAL DETECTED!")
        self.alert_button.setStyleSheet("background-color: red; color: white; font-size: 16px;")
        QTimer.singleShot(3000, self.reset_alert)

    def reset_alert(self):
        """Reset the alert button after 3 seconds"""
        self.alert_button.setText("NO ANIMAL")
        self.alert_button.setStyleSheet("background-color: green; color: white; font-size: 16px;")

    def classify_animal(self, cropped_img):
        """Classify the cropped image using MobileNetV3"""
        img = cv2.resize(cropped_img, (224, 224))
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        preds = classifier.predict(img)
        label = decode_predictions(preds, top=1)[0][0][1]
        return correction_map.get(label, label)

    def run_detection(self):
        """Runs YOLOv8 detection on live feed"""
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            detected_animal = "No animal detected"

            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    confidence = float(box.conf[0])

                    if label in wild_animals and confidence > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cropped_img = frame[y1:y2, x1:x2]

                        # Classify using MobileNetV3 if needed
                        label = self.classify_animal(cropped_img) if label in correction_map else label

                        detected_animal = label
                        self.trigger_alarm(label)

                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            print(f"Detected: {detected_animal}")  # Log detected animal

        cap.release()
        cv2.destroyAllWindows()

# Run the App
app = QApplication([])
window = WildAnimalDetector()
window.show()
app.exec_()