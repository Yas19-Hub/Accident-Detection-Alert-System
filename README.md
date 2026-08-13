#  AI Accident Detection and Alert System

An AI-powered accident detection system that uses **Computer Vision** and **Deep Learning** to identify road accidents from video footage. When an accident is detected, the system automatically sends emergency SMS alerts to the **nearest hospital**, **police authorities**, and emergency contacts using the **Twilio API**, enabling faster emergency response.
## ✨ Feature
* 🎥 Detects road accidents from video footage.
* 🖼️ Extracts frames from accident and non-accident videos for model training.
* 🤖 Deep learning-based accident classification.
* 📱 Sends real-time SMS alerts using Twilio.
* 🚓 Notifies police authorities.
* 🏥 Notifies nearby hospitals.
* ⚡ Helps reduce emergency response time.
* 📊 Can be extended for real-time CCTV surveillance.

## 🛠️ Technologies Used
* Python
* OpenCV
* TensorFlow / Keras
* NumPy
* Twilio API
* Kaggle Accident Detection Dataset

## Project Workflow

1. Collect accident and non-accident videos.
2. Extract video frames using OpenCV.
3. Preprocess images (resize to 224×224).
4. Train the deep learning model.
5. Detect accidents from video.
6. If an accident is detected:

   * Send SMS alert to the police.
   * Send SMS alert to the hospital.
   * Notify emergency contacts via Twilio.

## Future Enhancements
* GPS location sharing.
* Live CCTV accident detection.
* Emergency call automation.
* Mobile application integration.
* Real-time dashboard for monitoring incidents.
