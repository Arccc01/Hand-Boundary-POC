# Hand–Boundary Interaction POC  
Arvyax Internship Assignment

Author: Anuj Sharma  
Tech Stack: Python, OpenCV, NumPy  
Execution: CPU-only (No MediaPipe / No Pose APIs)

---

## 1. Overview

This project is a real-time prototype that tracks a user’s hand from a webcam feed and detects when the hand approaches a virtual object (boundary) on the screen.

The system dynamically classifies the interaction into three states:

- SAFE – hand is far from the boundary  
- WARNING – hand is approaching the boundary  
- DANGER – hand is extremely close / touching the boundary  

When the DANGER state is triggered, a large on-screen warning **“DANGER DANGER”** is displayed.

All processing is done using **classical computer vision techniques only**, without using MediaPipe, OpenPose, or any cloud-based AI APIs.

---

## 2. Features

- Real-time webcam capture using a **threaded frame reader** for better FPS.
- **Skin-based hand detection** using HSV color segmentation.
- **Face suppression** using Haar Cascade so that head movement does not affect detection.
- **Region of Interest (ROI)** around the virtual boundary to reduce noise.
- **Convex hull–based hand interaction point** for accurate location of the hand closest to the boundary.
- **Distance-based state logic** (SAFE / WARNING / DANGER).
- **Hysteresis and EMA smoothing** to avoid flickering and jitter.
- Red flashing overlay and large **“DANGER DANGER”** warning in DANGER state.
- Debug mask window for visual verification.
- Runs in real time on CPU (≥ 8 FPS).

---

## 3. How It Works (Methodology)

1. **Camera Input**
   - Frames are captured from the webcam using OpenCV with a threaded capture class.

2. **Preprocessing**
   - The frame is resized for faster processing.
   - A virtual rectangle (boundary) is drawn on the screen.

3. **Hand Segmentation**
   - The frame is converted to HSV color space.
   - Skin color is segmented using fixed HSV thresholds.
   - Morphological operations (open, close) and blurring remove noise.

4. **Face Filtering**
   - A Haar cascade face detector finds the face.
   - The detected face area is masked out from the skin mask to prevent false detection.

5. **Motion Filtering (Optional)**
   - Background subtraction (MOG2) can be combined with the skin mask.

6. **Contour Detection**
   - The largest valid contour near the virtual boundary is assumed to be the hand.
   - Small or invalid contours are rejected using area and solidity checks.

7. **Accurate Hand Location**
   - The convex hull of the hand contour is computed.
   - The hull point closest to the virtual rectangle is selected as the interaction point.
   - This gives better accuracy than using only the hand centroid.

8. **Distance & State Classification**
   - The distance from the interaction point to the rectangle is measured.
   - Based on distance thresholds:
     - SAFE → WARNING → DANGER states are assigned.
   - Hysteresis prevents rapid state flickering.

9. **Visual Output**
   - Rectangle color changes with state.
   - The hand interaction point is marked with a dot.
   - State, distance, and FPS are shown.
   - In DANGER state, a red flashing overlay and “DANGER DANGER” text are shown.


## 4. Installation & Execution

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
