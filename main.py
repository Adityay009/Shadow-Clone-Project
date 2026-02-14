import cv2
import mediapipe as mp
import numpy as np
import time
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ================= CONFIG =================
DIST_THRESHOLD = 100
COOLDOWN = 1.5
HAND_MEMORY_FRAMES = 5   # allow detection drop

is_cloning = False
last_toggle_time = 0
flash_counter = 0
hand_memory = 0

# ================= MODEL =================
model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)

# ================= CLONE FUNCTION =================
def create_clones(frame, grid_size=3):
    h, w, _ = frame.shape
    small = cv2.resize(frame, (w // grid_size, h // grid_size))
    row = np.hstack([small] * grid_size)
    grid = np.vstack([row] * grid_size)
    return cv2.resize(grid, (w, h))

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
frame_id = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame
    )

    result = detector.detect_for_video(mp_image, frame_id)
    frame_id += 1

    detected_two = False

    if result and len(result.hand_landmarks) >= 2:

        detected_two = True
        hand_memory = HAND_MEMORY_FRAMES

        hand1 = result.hand_landmarks[0][8]
        hand2 = result.hand_landmarks[1][8]

        x1, y1 = int(hand1.x * w), int(hand1.y * h)
        x2, y2 = int(hand2.x * w), int(hand2.y * h)

        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

        cv2.circle(frame, (x1, y1), 8, (0, 255, 0), -1)
        cv2.circle(frame, (x2, y2), 8, (0, 255, 0), -1)

        cv2.putText(frame, f"Dist: {int(dist)}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)

        if dist < DIST_THRESHOLD:
            if time.time() - last_toggle_time > COOLDOWN:
                is_cloning = not is_cloning
                last_toggle_time = time.time()
                flash_counter = 6

    else:
        if hand_memory > 0:
            hand_memory -= 1
        else:
            detected_two = False

    # ================= RENDER =================
    if is_cloning:
        display_frame = create_clones(frame, 3)
        cv2.putText(display_frame,
                    "SHADOW CLONE JUTSU",
                    (w//6, 80),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1.2,
                    (0, 165, 255),
                    3)
    else:
        display_frame = frame.copy()

    if flash_counter > 0:
        white = np.full((h, w, 3), 255, dtype="uint8")
        display_frame = cv2.addWeighted(display_frame, 0.4, white, 0.6, 0)
        flash_counter -= 1

    cv2.imshow("Naruto Developer Project", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

detector.close()
cap.release()
cv2.destroyAllWindows()