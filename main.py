import cv2
import mediapipe as mp
from collections import deque
import winsound  # For beep sound

# Voting settings
VOTE_WIN = 10
VOTE_NEED = 3
gesture_history = deque(maxlen=VOTE_WIN)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection

cap = cv2.VideoCapture(0)

middle_confirmed = False
beep_played = False

with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3) as hands, \
    mp_face.FaceDetection(min_detection_confidence=0.5) as face_detection:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = frame.shape

        results = hands.process(frame_rgb)
        face_results = face_detection.process(frame_rgb)

        gesture_name = "No Gesture"
        landmarks = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = hand_landmarks.landmark

            # Gesture recognition logic
            wrist_y = landmarks[0].y
            tips_y = [landmarks[i].y for i in [8, 12, 16, 20]]
            top_tip_y = min(tips_y)
            hand_height = wrist_y - top_tip_y

            # Improved detection for thumb and other fingers
            finger_flags = []

            # Thumb: compare x for horizontal direction
            if landmarks[4].x < landmarks[3].x:
                finger_flags.append(1)  # Thumb is open (for right hand)
            else:
                finger_flags.append(0)

            # Index to Pinky: vertical comparison
            for tip_i, pip_i in zip([8, 12, 16, 20], [6, 10, 14, 18]):
                if landmarks[tip_i].y < landmarks[pip_i].y:
                    finger_flags.append(1)  # Finger is up
                else:
                    finger_flags.append(0)

            # Classify gesture based on finger_flags
            gestures = {
                (0, 0, 1, 0, 0): "Middle Finger",
                (0, 1, 1, 0, 0): "Peace Sign",
                (0, 1, 0, 0, 1): "Rock Sign",
                (1, 1, 1, 1, 1): "Open Palm",
                (0, 0, 0, 0, 0): "Closed Fist",
                (1, 0, 0, 0, 0): "Thumbs Up",      
                (0, 0, 0, 1, 0): "Thumbs Down",    
            }

            gesture_tuple = tuple(finger_flags)
            gesture_name = gestures.get(gesture_tuple, "Unknown Gesture")

            if gesture_name == "Middle Finger":
                gesture_history.append(True)
                middle_confirmed = sum(gesture_history) >= VOTE_NEED
            else:
                gesture_history.clear()
                middle_confirmed = False
                beep_played = False
        else:
            gesture_history.clear()
            middle_confirmed = False
            beep_played = False

        # Face Blur
        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * img_w)
                y1 = int(bbox.ymin * img_h)
                x2 = int((bbox.xmin + bbox.width) * img_w)
                y2 = int((bbox.ymin + bbox.height) * img_h)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)

                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    face_blur = cv2.GaussianBlur(roi, (99, 99), 30)
                    frame[y1:y2, x1:x2] = face_blur

        # Middle Finger Response (blur + beep)
        if middle_confirmed and landmarks:
            xs = [int(landmarks[i].x * img_w) for i in [9,10,11,12]]
            ys = [int(landmarks[i].y * img_h) for i in [9,10,11,12]]

            x1, x2 = max(min(xs) - 20, 0), min(max(xs) + 20, img_w)
            y1, y2 = max(min(ys) - 20, 0), min(max(ys) + 20, img_h)

            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                blur = cv2.GaussianBlur(roi, (99, 99), 0)
                frame[y1:y2, x1:x2] = blur
                if not beep_played:
                    winsound.Beep(1000, 300)
                    beep_played = True

        # Draw detected gesture name
        cv2.putText(frame, f"{gesture_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Gesture Detection", frame)

        key = cv2.waitKey(5) & 0xFF
        if key == 27 or key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
