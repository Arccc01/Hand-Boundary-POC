import cv2
import numpy as np
import time
import threading

class ThreadedCapture:
    def __init__(self, src=0, width=None, height=None):
        self.cap = cv2.VideoCapture(src)
        if width: self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            grabbed, frame = self.cap.read()
            with self.lock:
                self.grabbed, self.frame = grabbed, frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.grabbed, self.frame.copy()

    def release(self):
        self.running = False
        try:
            self.thread.join(timeout=1)
        except Exception:
            pass
        self.cap.release()

# --------------- Configuration ----------------
FRAME_WIDTH = 640
PROCESS_WIDTH = 360
MIN_CONTOUR_AREA_RATIO = 0.007  
EMA_ALPHA = 0.30
HYSTERESIS_FRAMES = 4
REQUIRED_DETECT_FRAMES = 2

DANGER_FRAC = 0.06
WARNING_FRAC = 0.20

RECT_LEFT_FRAC = 0.45
RECT_TOP_FRAC = 0.25
RECT_W_FRAC = 0.15
RECT_H_FRAC = 0.25

ROI_MARGIN_PX = 100
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

DETECTION_MODE = 'SKIN_ONLY'  # 'SKIN_ONLY' or 'SKIN_AND_MOTION'
FACE_FILTER_ENABLED = True
# Face detection config (fractions of processed frame)
FACE_MIN_SIZE_FRAC = 0.10   # min face width/height relative to processed width (tune if needed)
FACE_PAD_FRAC = 0.30        # pad the face bbox by this fraction on each side (30% of face size)
FACE_DILATE_K = 9           # dilation kernel size for face mask (odd)
# -------------------------------------------------------

def nothing(x): pass

def create_hsv_controls(win='hsv'):
    cv2.createTrackbar('H_min', win, 0, 179, nothing)
    cv2.createTrackbar('H_max', win, 30, 179, nothing)
    cv2.createTrackbar('S_min', win, 30, 255, nothing)
    cv2.createTrackbar('S_max', win, 180, 255, nothing)
    cv2.createTrackbar('V_min', win, 40, 255, nothing)
    cv2.createTrackbar('V_max', win, 255, 255, nothing)

def read_hsv_controls(win='hsv'):
    hmin = cv2.getTrackbarPos('H_min', win)
    hmax = cv2.getTrackbarPos('H_max', win)
    smin = cv2.getTrackbarPos('S_min', win)
    smax = cv2.getTrackbarPos('S_max', win)
    vmin = cv2.getTrackbarPos('V_min', win)
    vmax = cv2.getTrackbarPos('V_max', win)
    return (hmin, smin, vmin), (hmax, smax, vmax)

def skin_mask_hsv(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([20, 150, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)

def marker_mask_hsv(frame, lower, upper):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))

def preprocess_mask(mask):
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=2)
    mask = cv2.GaussianBlur(mask, (7,7), 0)
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    return mask

def largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def rect_distance_to_point(rect, pt):
    x1,y1,x2,y2 = rect
    x,y = pt
    if x1 <= x <= x2 and y1 <= y <= y2:
        return 0.0
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return float(np.hypot(dx, dy))

def solidity_ok(contour):
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return False
    solidity = float(area) / hull_area
    return solidity > 0.25

# ---------- NEW: interaction point from convex hull ----------
def hand_point_towards_rect_proc(contour_roi, rect_proc, roi_x1, roi_y1):
    """
    contour_roi: contour in ROI coordinates
    rect_proc: rectangle in full processed coords (rx1,ry1,rx2,ry2)
    roi_x1, roi_y1: top-left of ROI in processed coords
    Returns: (x_proc, y_proc) in processed coords for the hull point closest
    to the rectangle (distance measured in processed coordinate space).
    """
    hull = cv2.convexHull(contour_roi)
    if hull is None or len(hull) == 0:
        return None

    best_pt_proc = None
    best_dist = None

    for p in hull[:,0,:]:
        x_roi, y_roi = int(p[0]), int(p[1])
        x_proc = x_roi + roi_x1
        y_proc = y_roi + roi_y1
        d = rect_distance_to_point(rect_proc, (x_proc, y_proc))
        if (best_dist is None) or (d < best_dist):
            best_dist = d
            best_pt_proc = (x_proc, y_proc)

    return best_pt_proc
# ------------------------------------------------------------

def draw_overlay(frame, state, hand_pt_disp, dist_px, fps, danger_flash_on, rect_disp, detection_confident):
    h,w = frame.shape[:2]
    x1,y1,x2,y2 = rect_disp
    color = (0,255,0) if state == 'SAFE' else ((0,255,255) if state == 'WARNING' else (0,0,255))
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)
    if hand_pt_disp is not None:
        dot_color = (0,255,0) if detection_confident else (255,0,0)
        cv2.circle(frame, (int(hand_pt_disp[0]), int(hand_pt_disp[1])), 10, dot_color, -1)
    cv2.putText(frame, f"State: {state}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"Dist(px): {int(dist_px) if dist_px is not None else -1}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
    if state == 'DANGER' and danger_flash_on:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (w,h), (0,0,255), -1)
        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
        cv2.putText(frame, "DANGER DANGER", (int(w*0.12), int(h*0.55)), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0,0,255), 4)

# Face detector (Haar cascade)
_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces_proc(frame_proc):
    # adaptive minSize computed from processed frame width/height
    h, w = frame_proc.shape[:2]
    min_sz = max(24, int(FACE_MIN_SIZE_FRAC * w))
    faces = _face_cascade.detectMultiScale(frame_proc, scaleFactor=1.08, minNeighbors=5, minSize=(min_sz, min_sz))
    return faces  # list of (x,y,w,h)

def build_face_mask(frame_proc, faces, pad_frac=FACE_PAD_FRAC, dilate_k=FACE_DILATE_K):
    h, w = frame_proc.shape[:2]
    face_mask = np.zeros((h, w), dtype=np.uint8)
    for (fx,fy,fw,fh) in faces:
        pad_w = int(fw * pad_frac)
        pad_h = int(fh * pad_frac)
        x1 = max(0, fx - pad_w)
        y1 = max(0, fy - pad_h)
        x2 = min(w, fx + fw + pad_w)
        y2 = min(h, fy + fh + pad_h)
        cv2.rectangle(face_mask, (x1,y1), (x2,y2), 255, -1)
    # dilate to be safe
    if dilate_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
        face_mask = cv2.dilate(face_mask, k, iterations=1)
    return face_mask

def run():
    global DETECTION_MODE, FACE_FILTER_ENABLED
    cap = ThreadedCapture(0, width=FRAME_WIDTH, height=int(FRAME_WIDTH*3/4))
    if not getattr(cap, 'cap', None) or not cap.cap.isOpened():
        print("Cannot open camera")
        return

    cv2.namedWindow('POC')
    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('mask', 480,360)
    cv2.namedWindow('hsv', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('hsv', 400,250)
    create_hsv_controls('hsv')

    backSub = cv2.createBackgroundSubtractorMOG2(history=250, varThreshold=40, detectShadows=False)

    use_marker_mode = False
    ema_pt = None
    last_state = 'SAFE'
    state_counter = 0
    fps_ema = None
    FPS_ALPHA = 0.15
    last_time = time.time()
    danger_flash_on = False
    flash_timer = 0
    flash_interval = 0.35

    process_w = PROCESS_WIDTH
    detect_frames = 0
    no_detect_decay = 999
    MAX_STALE_FRAMES = 6

    print("Press 'm' toggle marker mode, 'o' toggle detection mode (skin_only/skin_and_motion), 'f' toggle face filter, ESC to exit.")
    print("Face filter enabled by default:", FACE_FILTER_ENABLED)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h_orig, w_orig = frame.shape[:2]
        scale = process_w / float(w_orig)
        frame_proc = cv2.resize(frame, (process_w, int(h_orig*scale)), interpolation=cv2.INTER_AREA)
        proc_h, proc_w = frame_proc.shape[:2]

        # rectangle (processed coords)
        rx1 = int(proc_w * RECT_LEFT_FRAC)
        ry1 = int(proc_h * RECT_TOP_FRAC)
        rx2 = int(rx1 + proc_w * RECT_W_FRAC)
        ry2 = int(ry1 + proc_h * RECT_H_FRAC)
        rect_proc = (rx1, ry1, rx2, ry2)

        # --- Face detection on full processed frame ---
        faces = []
        face_mask_full = np.zeros((proc_h, proc_w), dtype=np.uint8)
        if FACE_FILTER_ENABLED:
            # detect faces on grayscale processed frame
            faces = detect_faces_proc(cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY))
            if len(faces) > 0:
                face_mask_full = build_face_mask(frame_proc, faces, pad_frac=FACE_PAD_FRAC, dilate_k=FACE_DILATE_K)
        # ------------------------------------------------

        # Build skin/marker mask on FULL processed frame first
        if use_marker_mode:
            lower, upper = read_hsv_controls('hsv')
            mask_skin_full = marker_mask_hsv(frame_proc, lower, upper)
        else:
            mask_skin_full = skin_mask_hsv(frame_proc)
        mask_skin_full = preprocess_mask(mask_skin_full)

        # Remove face pixels from the skin mask (global), so face never contributes
        if FACE_FILTER_ENABLED and face_mask_full is not None:
            # subtract face_mask_full (where face_mask_full==255 -> set skin to 0)
            mask_skin_full[face_mask_full == 255] = 0

        # motion mask (full frame)
        fg_full = backSub.apply(frame_proc, learningRate=0.01)
        fg_full = cv2.threshold(fg_full, 200, 255, cv2.THRESH_BINARY)[1]
        fg_full = cv2.morphologyEx(fg_full, cv2.MORPH_OPEN, KERNEL, iterations=1)
        fg_full = cv2.medianBlur(fg_full, 5)

        # Now crop ROI from the cleaned full skin mask and motion mask
        roi_x1 = max(0, rx1 - ROI_MARGIN_PX)
        roi_y1 = max(0, ry1 - ROI_MARGIN_PX)
        roi_x2 = min(proc_w, rx2 + ROI_MARGIN_PX)
        roi_y2 = min(proc_h, ry2 + ROI_MARGIN_PX)

        mask_skin_roi = mask_skin_full[roi_y1:roi_y2, roi_x1:roi_x2]
        fg_roi = fg_full[roi_y1:roi_y2, roi_x1:roi_x2]

        # choose detection mask according to mode (use skin-only by default)
        if DETECTION_MODE == 'SKIN_ONLY':
            detection_mask_roi = mask_skin_roi.copy()
        else:
            detection_mask_roi = cv2.bitwise_and(mask_skin_roi, fg_roi)
        detection_mask_roi = preprocess_mask(detection_mask_roi)

        # find largest contour in detection mask ROI
        c = largest_contour(detection_mask_roi)
        hand_point = None
        dist_px = None
        frame_area = proc_w * proc_h

        if c is not None and cv2.contourArea(c) > (MIN_CONTOUR_AREA_RATIO * frame_area) and solidity_ok(c):
            # NEW: use hull point closest to rectangle (in processed coords)
            tip_proc = hand_point_towards_rect_proc(c, rect_proc, roi_x1, roi_y1)
            if tip_proc is not None:
                cx_proc, cy_proc = tip_proc
                hand_point = (cx_proc, cy_proc)
                detect_frames += 1
                no_detect_decay = 0
            else:
                hand_point = None
                detect_frames = 0
                no_detect_decay += 1
        else:
            hand_point = None
            detect_frames = 0
            no_detect_decay += 1

        if detect_frames < REQUIRED_DETECT_FRAMES:
            hand_point = None

        # EMA smoothing on hand interaction point
        if hand_point is not None:
            if 'ema_pt' not in locals() or ema_pt is None:
                ema_pt = np.array(hand_point, dtype=np.float32)
            else:
                ema_pt = (1-EMA_ALPHA)*ema_pt + EMA_ALPHA*np.array(hand_point, dtype=np.float32)
            pt_disp_proc = (int(ema_pt[0]), int(ema_pt[1]))
            detection_confident = True
        else:
            if 'ema_pt' in locals() and ema_pt is not None and no_detect_decay <= MAX_STALE_FRAMES:
                pt_disp_proc = (int(ema_pt[0]), int(ema_pt[1]))
                detection_confident = False
            else:
                pt_disp_proc = None
                detection_confident = False

        # compute distance & state
        if pt_disp_proc is not None:
            dist_px = rect_distance_to_point(rect_proc, pt_disp_proc)
        else:
            dist_px = None

        danger_thresh = DANGER_FRAC * proc_w
        warning_thresh = WARNING_FRAC * proc_w

        if dist_px is None:
            new_state = 'SAFE'
        else:
            if dist_px <= danger_thresh:
                new_state = 'DANGER'
            elif dist_px <= warning_thresh:
                new_state = 'WARNING'
            else:
                new_state = 'SAFE'

        if new_state == last_state:
            state_counter = 0
        else:
            state_counter += 1
            if state_counter >= HYSTERESIS_FRAMES:
                last_state = new_state
                state_counter = 0

        # fps EMA
        now = time.time()
        dt = now - last_time
        last_time = now
        if dt > 0:
            fps_inst = 1.0 / dt
            fps_ema = fps_inst if fps_ema is None else FPS_ALPHA*fps_inst + (1-FPS_ALPHA)*fps_ema
        fps_disp = fps_ema if fps_ema is not None else 0.0

        # danger flash toggle
        flash_timer += dt
        if flash_timer >= flash_interval:
            danger_flash_on = not danger_flash_on
            flash_timer = 0

        # map to display coords
        scale_back_x = w_orig / float(proc_w)
        scale_back_y = h_orig / float(proc_h)
        disp_pt = None
        if pt_disp_proc is not None:
            disp_pt = (int(pt_disp_proc[0] * scale_back_x), int(pt_disp_proc[1] * scale_back_y))
        rect_disp = (int(rx1*scale_back_x), int(ry1*scale_back_y), int(rx2*scale_back_x), int(ry2*scale_back_y))

        draw_overlay(frame, last_state, disp_pt, dist_px * scale_back_x if dist_px is not None else None, fps_disp, danger_flash_on, rect_disp, detection_confident)

        # Debug mask: paste detection ROI back into full processed mask for visibility and show motion below
        mask_show = np.zeros((proc_h, proc_w), dtype=np.uint8)
        mask_show[roi_y1:roi_y2, roi_x1:roi_x2] = detection_mask_roi
        # subtract face area for visualization clarity
        if FACE_FILTER_ENABLED and np.any(face_mask_full):
            mask_show[face_mask_full==255] = 0
        motion_show = fg_full.copy()
        combined_debug = np.vstack([mask_show, motion_show])
        combined_debug = cv2.resize(combined_debug, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('mask', combined_debug)

        # draw face bboxes on display if face filter enabled (for debugging)
        if FACE_FILTER_ENABLED and len(faces)>0:
            for (fx,fy,fw,fh) in faces:
                fx_d = int(fx * (w_orig/float(proc_w)))
                fy_d = int(fy * (h_orig/float(proc_h)))
                fw_d = int(fw * (w_orig/float(proc_w)))
                fh_d = int(fh * (h_orig/float(proc_h)))
                # draw magenta box for face
                cv2.rectangle(frame, (fx_d, fy_d), (fx_d+fw_d, fy_d+fh_d), (255,0,255), 2)

        cv2.imshow('POC', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('m'):
            use_marker_mode = not use_marker_mode
            print("Marker mode:", use_marker_mode)
        elif key == ord('s'):
            ts = int(time.time()); cv2.imwrite(f'poc_handpoint_facefixed_{ts}.png', frame); print("Saved screenshot")
        elif key == ord('o'):
            DETECTION_MODE = 'SKIN_AND_MOTION' if DETECTION_MODE == 'SKIN_ONLY' else 'SKIN_ONLY'
            print("Switched detection mode to:", DETECTION_MODE)
        elif key == ord('f'):
            FACE_FILTER_ENABLED = not FACE_FILTER_ENABLED
            print("Face filter enabled:", FACE_FILTER_ENABLED)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()
