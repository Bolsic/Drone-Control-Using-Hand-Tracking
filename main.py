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
    return angle / np.pi

def draw_arrows(window, arrows):
    for arrow in arrows:
        arrow.draw()

# Make Arrow class
class Arrow:
    def __init__(self, coordinates, color, key, threshold):
        self.coordinates = coordinates
        self.color = color
        self.key = key
        self.threshold = threshold

    def draw(self):
        pygame.draw.polygon(window, self.color, self.coordinates)
    
    def activate(self, signal):
        # if the signal and threshold have the same sign and 
        if abs(signal) > abs(self.threshold) and signal*self.threshold > 0:
            opacity = int(255*abs(signal))
            if opacity > 255: opacity = 255
            if opacity < 0: opacity = 0
            self.color = (opacity, opacity, 0)
        else: 
            self.color = (0, 0, 0)

# make subclass UpArrow>
class UpArrow(Arrow):
    def __init__(self, coordinates, color, key, threshold):
        super().__init__(coordinates, color, key, threshold)
        self.step = 0.01

    def calibrate(self, signal):
        print("Previous threshold", self.threshold)
        self.threshold = signal - self.step
        print("New threshold", self.threshold)
    
    def activate(self, signal):
        # Lower value means its farther away
        if abs(signal) < abs(self.threshold) and signal*self.threshold > 0:
            opacity = abs(int(255*abs(signal - self.threshold))) * 10
            if opacity > 255: opacity = 255
            if opacity < 0: opacity = 0
            self.color = (opacity, opacity, 0)
        else: 
            self.color = (0, 0, 0)
        print(abs(signal - self.threshold))

# make subclass DownArrow
class DownArrow(Arrow):
    def __init__(self, coordinates, color, key, threshold):
        super().__init__(coordinates, color, key, threshold)
        self.step = 0.01

    def calibrate(self, signal):
        print("Previous threshold", self.threshold)
        self.threshold = signal + self.step
        print("New threshold", self.threshold)
    
    def activate(self, signal):
        # Lower value means its farther away
        if abs(signal) > abs(self.threshold) and signal*self.threshold > 0:
            opacity = abs(int(255*abs(signal - self.threshold))) * 5
            if opacity > 255: opacity = 255
            if opacity < 0: opacity = 0
            self.color = (opacity, opacity, 0)
        else: 
            self.color = (0, 0, 0)

# colors
dark_gray = (50, 50, 50)
light_gray = (200, 200, 200)
yellow = (255, 255, 0)

RL_angle = 0
left_angle_threshold = -0.05
right_angle_threshold = 0.05
FB_angle = 0
forward_angle_threshold = 0.05
back_angle_threshold = -0.05
hand_height = 0
up_height_threshold = 0.005
down_height_threshold = -0.005

previous_hand_heights = [0, 0, 0, 0, 0]

pygame.init()
win_width = 600
win_height = 600
window = pygame.display.set_mode((win_width, win_height))
pygame.display.set_caption("Arrows")
clock = pygame.time.Clock()

# Make the same arows but half the size
forward_arrow = Arrow([(win_width//2, win_height//2), (win_width//2-60, win_height//2-100), (win_width//2+60, win_height//2-100)],
                       light_gray, pygame.K_UP, forward_angle_threshold)
back_arrow = Arrow([(win_width//2, win_height//2), (win_width//2-60, win_height//2+100), (win_width//2+60, win_height//2+100)],
                    light_gray, pygame.K_DOWN, back_angle_threshold)
left_arrow = Arrow([(win_width//2, win_height//2), (win_width//2-100, win_height//2-60), (win_width//2-100, win_height//2+60)],
                    light_gray, pygame.K_LEFT, left_angle_threshold)
right_arrow = Arrow([(win_width//2, win_height//2), (win_width//2+100, win_height//2-60), (win_width//2+100, win_height//2+60)],
                     light_gray, pygame.K_RIGHT, right_angle_threshold)
up_arrow = UpArrow([(win_width//5-40, win_height//5-100), (win_width//5-100, win_height//5), (win_width//5+20, win_height//5)],
                 light_gray, pygame.K_UP, up_height_threshold)
down_arrow = DownArrow([(win_width//5-40, win_height//5+120), (win_width//5-100, win_height//5+20), (win_width//5+20, win_height//5+20)],
                   light_gray, pygame.K_DOWN, down_height_threshold)

arrows = [forward_arrow, back_arrow, left_arrow, right_arrow, up_arrow, down_arrow]

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize OpenCV Video Capture
cap = cv2.VideoCapture(0)


while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display
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

            for landmark in hand_landmarks.landmark:
                hand_height += landmark.z
            hand_height /= len(hand_landmarks.landmark)
            hand_height = -1 * hand_height
            break
    else:
        RL_angle = 0
        FB_angle = 0
        hand_height = 0

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
            print("NESTO")
            if event.key == pygame.K_SPACE:
                arrows[4].calibrate(hand_height) 
                arrows[5].calibrate(hand_height)
            if event.key == pygame.K_q:
                pygame.quit()
                sys.exit()


    window.fill(dark_gray)

    # Normalise the angles
    RL_angle = normalise_angle(RL_angle)
    FB_angle = normalise_angle(FB_angle)

    # Activate the arrows
    arrows[0].activate(FB_angle)
    arrows[1].activate(FB_angle)
    arrows[2].activate(RL_angle)
    arrows[3].activate(RL_angle)

    previous_hand_heights.pop(0)
    previous_hand_heights.append(hand_height)
    #calculate the average hand height
    average_hand_height = sum(previous_hand_heights) / len(previous_hand_heights)
    #print(average_hand_height)

    arrows[4].activate(average_hand_height)
    arrows[5].activate(average_hand_height)

    #print(hand_height)

    draw_arrows(window, arrows)
    pygame.display.update()

cap.release()
cv2.destroyAllWindows()
pygame.quit()

