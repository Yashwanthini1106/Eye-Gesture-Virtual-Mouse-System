
# 👁️🖱️Eye Gesture Virtual Mouse System
🚀 Project Overview: 
                 This project implements a Eye Gesture Virtual Mouse System powered by computer vision and facial landmark detection. By tracking eye movements and blink patterns through a webcam, users can control their computer’s mouse without any physical hardware. It demonstrates how real-time vision processing can enable hands-free human-computer interaction and improve accessibility.

# ✨Key Features:
- Eye movement tracking for cursor control
- Smooth and responsive cursor movement
- Single blink detection for left-click
- Double blink detection for double-click
- Eye direction–based scrolling
- Hands-free computer interaction

# 🛠️Tech Stack:
- Python
- OpenCV – Video capture and image processing
- MediaPipe – Facial landmark and eye tracking
- PyAutoGUI – Mouse control automation

# ⚙️How It Works:
👀 Cursor Movement

The system tracks eye landmarks using MediaPipe Face Mesh.
The position of the iris/pupil is mapped to screen coordinates to move the cursor.

### 😉 Left Click

A short blink of the left eye triggers a left-click.

### 😉 Right Click

A short blink of the right eye triggers a right-click.

### 👁️ Left Eye Long Blink — Selection

A long blink of the left eye triggers the selection action.

### 👁️ Right Eye Long Blink — Drag and Drop

A long blink of the right eye triggers the drag-and-drop action.

### ⬆️⬇️ Scrolling

Looking upward scrolls up.
Looking downward scrolls down.

# 📂 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/virtual-eye-mouse.git
cd virtual-eye-mouse
```

### 2️⃣ Install Dependencies
```bash
pip install opencv-python mediapipe pyautogui
```

### 3️⃣ Run the Project
```bash
python main.py
```

---

# 💻 Requirements

- Python 3.7+
- Webcam
- Good lighting conditions for accurate eye detection
