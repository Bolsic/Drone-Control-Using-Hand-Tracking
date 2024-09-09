import numpy as np
import cv2
import mediapipe as mp

cap=cv2.VideoCapture(0)
if not (cap.isOpened()):
    print("Could not open video device")

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)
print("Window created")
while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        continue
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow('Display', image)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()


cap.release()


