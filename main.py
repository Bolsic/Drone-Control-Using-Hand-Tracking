import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import pygame
import numpy as np
import sys

def normalise_angle(angle):
    if angle < -np.pi:
        return -1
    elif angle > np.pi:
        return 1
    return angle / (np.pi /2)

def normalise_height(height):
    # Normalise the values between 0.15 and 0.02 to -1 and 1
    normalized_value = ((height - 0.01) / (0.15 - 0.01)) * (1 - (-1)) + (-1)
    # Flip the value so that the higher the hand the higher the value
    normalized_value = -1 * normalized_value
    # Make sure the value is between -1 and 1
    if normalized_value > 1:
        return 1
    elif normalized_value < -1:
        return -1
    return normalized_value

# Make Arrow class
class Arrow:
    def __init__(self, coordinates, color, key, threshold_step):
        self.coordinates = coordinates
        self.color = color
        self.key = key
        self.threshold = threshold_step
        self.threshold_step = threshold_step

    def draw(self):
        pygame.draw.polygon(window, self.color, self.coordinates)

    def calibrate(self, signal):
        self.threshold = signal + self.threshold_step
    
    def activate(self, signal):
        opacity = int(abs(signal-self.threshold) * 255)
        if opacity > 255: opacity = 255
        if opacity < 0: opacity = 0
        self.color = (opacity, opacity, 0)

    def recive_signal(self, signal):
        # if the signal and threshold have the same sign and 
        if self.threshold_step > 0 and signal > self.threshold:
            print("Activate", self.threshold_step, signal, self.threshold)
            self.activate(signal)
        elif self.threshold_step < 0 and signal < self.threshold:
            print("Activate", self.threshold_step, signal, self.threshold)
            self.activate(signal)
        else: 
            self.color = (0, 0, 0)

class Arrows:
    # This class will be used to group all the arrows
    # forvard, back, left, right, up, down
    def __init__(self, arrows):
        self.arrows = arrows
    
    def draw(self):
        for arrow in self.arrows:
            arrow.draw()

    def calibrate_horizontal_arrows(self, signal):
        # Recive a signal composed of 3 numbers: FB_angle, RL_angle, hand_height
        # and calibrate all the arrows
        for i in range(4):
            self.arrows[i].calibrate(signal[i//2])
        
    def calibrate_vertical_arrows(self, signal):
        # Recive a signal composed of 3 numbers: FB_angle, RL_angle, hand_height
        # and calibrate all the arrows
        for i in range(4,6):
            self.arrows[i].calibrate(signal[i//2])
    
    def recive_signal(self, signal):
        # Recive a signal composed of 3 numbers: FB_angle, RL_angle, hand_height
        # and activate all the arrows
        for i in range(6):
            self.arrows[i].recive_signal(signal[i//2])
        
    def print_thresholds(self):
        print("Thresholds:")
        print("Forward: ", self.arrows[0].threshold)
        print("Backward: ", self.arrows[1].threshold)
        print("Left: ", self.arrows[2].threshold)
        print("Right: ", self.arrows[3].threshold)
        print("Up: ", self.arrows[4].threshold)
        print("Down: ", self.arrows[5].threshold)



# colors
dark_gray = (50, 50, 50)
light_gray = (200, 200, 200)
yellow = (255, 255, 0)

horizontal_angle_threshold_step = 0.05
height_threshold_step = 0.005

RL_angle = 0
FB_angle = 0
hand_height = 0

hand_height_keypoint_indices = [0, 5, 9, 13, 17]
previous_hand_heights = [0, 0, 0, 0, 0]
signal = [0, 0, 0]

pygame.init()
win_width = 600
win_height = 600
window = pygame.display.set_mode((win_width, win_height))
pygame.display.set_caption("Arrows")
clock = pygame.time.Clock()

# Initialise arrows and their locations
forward_arrow = Arrow([(win_width//2, win_height//2), (win_width//2-60, win_height//2-100), (win_width//2+60, win_height//2-100)],
                       light_gray, pygame.K_UP, horizontal_angle_threshold_step)
back_arrow = Arrow([(win_width//2, win_height//2), (win_width//2-60, win_height//2+100), (win_width//2+60, win_height//2+100)],
                    light_gray, pygame.K_DOWN, -horizontal_angle_threshold_step)
left_arrow = Arrow([(win_width//2, win_height//2), (win_width//2-100, win_height//2-60), (win_width//2-100, win_height//2+60)],
                    light_gray, pygame.K_LEFT, -horizontal_angle_threshold_step)
right_arrow = Arrow([(win_width//2, win_height//2), (win_width//2+100, win_height//2-60), (win_width//2+100, win_height//2+60)],
                     light_gray, pygame.K_RIGHT, horizontal_angle_threshold_step)
up_arrow = Arrow([(win_width//5-40, win_height//5-100), (win_width//5-100, win_height//5), (win_width//5+20, win_height//5)],
                 light_gray, pygame.K_UP, height_threshold_step)
down_arrow = Arrow([(win_width//5-40, win_height//5+120), (win_width//5-100, win_height//5+20), (win_width//5+20, win_height//5+20)],
                   light_gray, pygame.K_DOWN, -height_threshold_step)
# Group the arrows
arrows = Arrows([forward_arrow, back_arrow, left_arrow, right_arrow, up_arrow, down_arrow])

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

    hand_height = 0
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Calculate the angle between the tumb and pinky
            RL_height = hand_landmarks.landmark[4].z - hand_landmarks.landmark[20].z
            RL_width = hand_landmarks.landmark[4].x - hand_landmarks.landmark[20].x
            RL_angle = -np.arctan(RL_height/RL_width)

            # Calculate the angle between the middle finger and wrist
            FB_height = hand_landmarks.landmark[12].z - hand_landmarks.landmark[0].z
            FB_width = hand_landmarks.landmark[12].y - hand_landmarks.landmark[0].y
            FB_angle = np.arctan(FB_height/FB_width)

            for idx in hand_height_keypoint_indices:
                hand_height += hand_landmarks.landmark[idx].z

            hand_height /= len(hand_height_keypoint_indices)
            hand_height = -1 * hand_height
            break
    else:
        RL_angle = 0
        FB_angle = 0
        hand_height = 0
    
    # Calculate the average hand height of the last 5 frames
    previous_hand_heights.pop(0)
    previous_hand_heights.append(hand_height)
    average_hand_height = sum(previous_hand_heights) / len(previous_hand_heights)

    # Normalise the angles
    RL_angle = normalise_angle(RL_angle)
    FB_angle = normalise_angle(FB_angle)
    hand_height = normalise_height(hand_height)

    signal = [FB_angle, RL_angle, hand_height]

    # Display the image
    cv2.imshow('Hand Keypoints', image)

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
                arrows.calibrate_vertical_arrows(signal)
            if event.key == pygame.K_RETURN:
                arrows.calibrate_horizontal_arrows(signal)
            # Quit the program
            if event.key == pygame.K_q:
                pygame.quit()
                sys.exit()

    # Activate the arrows
    arrows.print_thresholds()
    print(signal)
    arrows.recive_signal(signal)

    # Draw the arrows
    window.fill(dark_gray)
    arrows.draw()
    pygame.display.update()

cap.release()
cv2.destroyAllWindows()
pygame.quit()

