import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import pygame
import numpy as np
import sys
from Control import Control
from djitellopy import Tello


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

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize OpenCV Video Capture
cap = cv2.VideoCapture(0)

#CONNECT TO DRONE
# my_drone = Tello()
# my_drone.connect()
# my_drone.for_back_velocity = 0
# my_drone.left_right_velocity = 0
# my_drone.up_down_velocity = 0
# my_drone.yaw_velocity = 0
# my_drone.speed = 0

# print("######################")
# print("Drone Battery: ", my_drone.get_battery())
# print("######################")

drone_rotation_speed = 40
using_drone = False

# control = Control(win_height, win_width, drone=my_drone, drone_control=using_drone)
control = Control(win_height, win_width, drone=0, drone_control=using_drone)

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
                pass
            if event.key == pygame.K_RETURN:
                control.calibrate_horizontal_arrows()
            if event.key == pygame.K_BACKSPACE:
                control.takeoff()
            # Quit the program
            if event.key == pygame.K_q:
                pygame.quit()
                sys.exit()

    # Send the signal to the arrows and drone
    control.send_arrow_signal()
    # Display the image
    #cv2.imshow('Hand Keypoints', image)
    cv2.waitKey(1)
    # Draw the arrows
    window.fill(dark_gray)
    control.draw(window)
    pygame.display.update()

cap.release()
cv2.destroyAllWindows()
pygame.quit()

