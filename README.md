# Hand–Boundary Interaction POC  

## 🚀 Overview

This project is a real-time computer vision prototype that tracks a user’s hand using a webcam and detects its interaction with a virtual object (boundary) drawn on the screen.

The system dynamically classifies the interaction into three states:

- **SAFE** – hand is far from the virtual boundary  
- **WARNING** – hand is approaching the boundary  
- **DANGER** – hand is extremely close / touching the boundary  

During the DANGER state, a large on-screen alert **“DANGER DANGER”** is displayed with a red flashing overlay.

All tracking and logic are implemented using **classical computer vision techniques only**, without MediaPipe, OpenPose, or any cloud-based AI APIs.

---

## ✨ Key Features

- Real-time webcam processing with **threaded frame capture** for smoother FPS.
- **Skin-based hand detection** using HSV color segmentation.
- **Face suppression** using Haar Cascade so that head movement does not affect detection.
- **Region of Interest (ROI)** around the virtual boundary for noise reduction.
- **Convex hull–based hand interaction point** for higher positional accuracy.
- **Distance-based state logic**: SAFE → WARNING → DANGER.
- **EMA smoothing + hysteresis** to avoid flickering.
- Red flashing overlay and large **“DANGER DANGER”** text in DANGER state.
- FPS, distance, and state displayed in real time.
- Debug mask window for visual verification.
- Runs fully on **CPU in real time (≥ 8 FPS)**.

---

## 🧠 Methodology (How It Works)

1. **Camera Input**  
   Webcam frames are captured using OpenCV with a threaded capture class.

2. **Preprocessing**  
   Frames are resized for faster processing. A virtual rectangle (boundary) is drawn.

3. **Hand Segmentation**  
   Skin pixels are extracted using HSV color thresholding and cleaned using morphological operations.

4. **Face Filtering**  
   A Haar cascade detects the face. The detected face region is masked out from the skin mask to avoid false triggers.

5. **ROI-Based Detection**  
   Detection is limited to a region around the boundary to improve robustness and speed.

6. **Hand Contour & Convex Hull**  
   The largest valid contour is selected as the hand.  
   The convex hull is computed and the **hull point closest to the virtual rectangle** is used as the interaction point.

7. **Distance Measurement & States**  
   The Euclidean distance from the interaction point to the rectangle is computed and mapped to:
   - SAFE
   - WARNING
   - DANGER  
   Hysteresis prevents rapid flickering.

8. **Visualization**  
   Rectangle color changes with state. A dot shows the interaction point.  
   In DANGER, a flashing red overlay and large warning text are shown.

---

