import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import pygame
import numpy as np
import sys
from Control import Control


# Colors
dark_gray = (50, 50, 50)
light_gray = (200, 200, 200)
yellow = (255, 255, 0)

# Initialize Pygame
pygame.init()
win_width = 600
win_height = 600
window = pygame.display.set_mode((win_width, win_height))
pygame.display.set_caption("Arrows")
clock = pygame.time.Clock()

control = Control(win_height, win_width, light_gray)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize OpenCV Video Capture
cap = cv2.VideoCapture(0)

# Main loop
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Calculate the angles and distances
            control.calculate_signal(results)

    if cv2.waitKey(5) & 0xFF == 27:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            # Calibrate the arrows
            if event.key == pygame.K_SPACE:
                control.calibrate_vertical_arrows()
            if event.key == pygame.K_RETURN:
                control.calibrate_horizontal_arrows()
            # Quit the program
            if event.key == pygame.K_q:
                pygame.quit()
                sys.exit()

    # Send the signal to the arrows and drone
    control.send_signal()

    # Display the image
    cv2.imshow('Hand Keypoints', image)

    # Draw the arrows
    window.fill(dark_gray)
    control.draw(window)
    pygame.display.update()

cap.release()
cv2.destroyAllWindows()
pygame.quit()

