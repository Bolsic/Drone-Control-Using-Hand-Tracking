import math
import numpy as np
from Arrows import Arrows
from Arrows import Arrow

def normalize_angle(angle):
    # Normalize the values between -pi and pi to -1 and 1
    if angle < -np.pi:
        return -1
    elif angle > np.pi:
        return 1
    return angle / (np.pi /2)

def normalize_height(height):
    # Normalize the values between 0.25 and 0.05 to -1 and 1
    normalized_value = ((height - 0.05) / (0.25 - 0.0)) * (1 - (-1)) + (-1)
    # Flip the value so that the higher the hand the higher the value
    normalized_value = -1 * normalized_value
    # Make sure the value is between -1 and 1
    if normalized_value > 1:
        return 1
    elif normalized_value < -1:
        return -1
    return normalized_value


class Control:
    def __init__(self, win_height, win_width, drone, drone_control=False):
        self.RL_angle = 0
        self.FB_angle = 0
        self.hand_height = 0
        self.signal = np.array([0, 0, 0])
        self.drone_signal = np.array([0, 0, 0])
        
        self.drone = drone
        self.drone_control = drone_control
        self.drone_top_speed = 100 # The max is 100
        self.flying = False

        # Thresholds
        self.horizontal_angle_threshold_step = 0.10
        self.height_threshold_step = 0.25

        # Initialise arrows and their locations
        forward_arrow = Arrow([(win_width//2, win_height//2), (win_width//2-60, win_height//2-100), (win_width//2+60, win_height//2-100)],
                            self.horizontal_angle_threshold_step)
        back_arrow = Arrow([(win_width//2, win_height//2), (win_width//2-60, win_height//2+100), (win_width//2+60, win_height//2+100)],
                            -self.horizontal_angle_threshold_step)
        left_arrow = Arrow([(win_width//2, win_height//2), (win_width//2-100, win_height//2-60), (win_width//2-100, win_height//2+60)],
                            -self.horizontal_angle_threshold_step)
        right_arrow = Arrow([(win_width//2, win_height//2), (win_width//2+100, win_height//2-60), (win_width//2+100, win_height//2+60)],
                            self.horizontal_angle_threshold_step)
        up_arrow = Arrow([(win_width//5-40, win_height//5-100), (win_width//5-100, win_height//5), (win_width//5+20, win_height//5)],
                        self.height_threshold_step)
        down_arrow = Arrow([(win_width//5-40, win_height//5+120), (win_width//5-100, win_height//5+20), (win_width//5+20, win_height//5+20)],
                        -self.height_threshold_step)
        # Group the arrows
        self.arrows = Arrows([forward_arrow, back_arrow, left_arrow, right_arrow, up_arrow, down_arrow])


    def normalize_signals(self):
        self.RL_angle = normalize_angle(self.RL_angle)
        self.FB_angle = normalize_angle(self.FB_angle)
        self.hand_height = normalize_height(self.hand_height)

    def print_signals(self):
        # Print information
        self.arrows.print_thresholds()
        print("RL: ", self.RL_angle, "FB: ", self.FB_angle, "Height: ", self.hand_height)
        print("Drone Signal: ", self.drone_signal)


    def calculate_signal(self, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                # Calculate the angle between the tumb and pinky
                self.RL_height = hand_landmarks.landmark[5].z - hand_landmarks.landmark[17].z
                self.RL_width = math.sqrt((hand_landmarks.landmark[5].x - hand_landmarks.landmark[17].x)**2 + (hand_landmarks.landmark[5].y - hand_landmarks.landmark[17].y)**2)
                self.RL_angle = np.arctan(self.RL_height / self.RL_width)

                # Calculate the angle between the middle finger and wrist
                self.FB_height = hand_landmarks.landmark[9].z - hand_landmarks.landmark[0].z
                self.FB_width = math.sqrt((hand_landmarks.landmark[9].y - hand_landmarks.landmark[0].y)**2 + (hand_landmarks.landmark[9].x - hand_landmarks.landmark[0].x)**2)
                self.FB_angle = -np.arctan(self.FB_height / self.FB_width)


                # Calculate the distance between points 5 and 17
                x = hand_landmarks.landmark[5].x - hand_landmarks.landmark[17].x
                y = hand_landmarks.landmark[5].y - hand_landmarks.landmark[17].y
                self.hand_height = np.sqrt(x**2 + y**2)
                
                break
        else:
            self.RL_angle = 0
            self.FB_angle = 0
            self.hand_height = 0

        self.normalize_signals()
        self.signal = np.array([self.FB_angle, self.RL_angle, self.hand_height])

        # Calculate the drone signal
        self.drone_signal = self.signal.copy()
        activated_axis = self.arrows.activated_axis()

        if activated_axis[0] == False and activated_axis[1] == False:
            self.drone_signal[0] = 0
        if activated_axis[2] == False and activated_axis[3] == False:
            self.drone_signal[1] = 0
        if activated_axis[4] == False and activated_axis[5] == False:
            self.drone_signal[2] = 0

        self.drone_signal *= self.drone_top_speed

    def send_drone_signal(self):
        self.drone.send_rc_control(int(self.drone_signal[1]),
                                   int(self.drone_signal[0]), 
                                   int(self.drone_signal[2]), 0)

    def send_arrow_signal(self):
        # Send the signal to the arrows and drone
        self.arrows.recive_signal(self.signal)
        if self.drone_control and self.flying:
            self.send_drone_signal()
        
    def draw(self, window):
        self.arrows.draw(window)

    def calibrate_vertical_arrows(self):
        self.arrows.calibrate_vertical_arrows(self.signal)

    def calibrate_horizontal_arrows(self):
        self.arrows.calibrate_horizontal_arrows(self.signal)

    def takeoff(self):
        if self.drone_control:
            if self.flying:
                print("#########################")
                print("Landing...")
                print("#########################")
                self.drone.land()
                self.flying = False
            else:
                print("#########################")
                print("Takeoff...")
                print("#########################")
                self.drone.takeoff()
                self.flying = True
    


    
