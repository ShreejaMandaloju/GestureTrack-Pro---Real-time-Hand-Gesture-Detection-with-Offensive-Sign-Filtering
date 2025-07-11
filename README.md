# 🤖 GestureTrack Pro — Real-Time Hand Gesture Detection with Offensive Sign Filtering

GestureTrack Pro is a real-time computer vision project built using **Python**, **OpenCV**, and **MediaPipe** that detects various hand gestures using a webcam and performs specific actions such as blurring offensive signs (like the middle finger) to ensure content safety. 

It also recognizes other gestures like ✌️ Peace sign, 🤘 Rock sign, 👋 Open palm, ✊ Closed fist, 👍 Thumbs up, 👎 Thumbs down — making it ideal for **AI-based gesture control, safety monitoring, and human-computer interaction**.

---

## 🚀 Features

- ✋ Detects and identifies hand gestures:
  - Middle Finger (blurred + beep alert)
  - Peace Sign
  - Rock Sign
  - Open Palm
  - Closed Fist
  - Thumbs Up / Down
- 🧠 Uses **MediaPipe Hands & Face Detection**
- 🔒 Automatically blurs faces for privacy
- 🔇 Blurs offensive gestures to avoid displaying them
- 📢 Plays beep sound when offensive gesture is detected
- ⚡ Works in real-time using webcam

---

## 🛠️ Tech Stack

| Component      | Technology                |
|----------------|---------------------------|
| Language       | Python                    |
| CV Library     | OpenCV                    |
| Hand Tracking  | MediaPipe Hands           |
| Face Detection | MediaPipe Face Detection  |
| Sound Alerts   | winsound (Windows only)   |
| Gesture Logic  | Custom logic using finger landmarks |
| IDE            | VS Code / Jupyter Notebook |

---

## 🧩 Installation

### Clone the repository

```bash
git clone https://github.com/ShreejaMandaloju/GestureTrack-Pro---Real-time-Hand-Gesture-Detection-with-Offensive-Sign-Filtering.git
cd GestureTrack-Pro---Real-time-Hand-Gesture-Detection-with-Offensive-Sign-Filtering
```
