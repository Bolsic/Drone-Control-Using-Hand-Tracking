import numpy as np
from Arrows import Arrows
from Arrows import Arrow

def normalize_angle(angle):
    # normalize the values between -pi and pi to -1 and 1
    if angle < -np.pi:
        return -1
    elif angle > np.pi:
        return 1
    return angle / (np.pi /2)

def normalize_height(height):
    # normalize the values between 0.25 and 0.05 to -1 and 1
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
    def __init__(self, win_height, win_width, drone=0, drone_control=False):
        self.RL_angle = 0
        self.FB_angle = 0
        self.hand_height = 0
        self.signal = [0, 0, 0]
        
        self.drone = drone
        self.drone_control = drone_control
        self.drone_speed_low = 20
        self.drone_speed_high = 100

        # Thresholds
        horizontal_angle_threshold_step = 0.1
        height_threshold_step = 0.1

        # Initialise arrows and their locations
        forward_arrow = Arrow([(win_width//2, win_height//2), (win_width//2-60, win_height//2-100), (win_width//2+60, win_height//2-100)],
                            horizontal_angle_threshold_step)
        back_arrow = Arrow([(win_width//2, win_height//2), (win_width//2-60, win_height//2+100), (win_width//2+60, win_height//2+100)],
                            -horizontal_angle_threshold_step)
        left_arrow = Arrow([(win_width//2, win_height//2), (win_width//2-100, win_height//2-60), (win_width//2-100, win_height//2+60)],
                            -horizontal_angle_threshold_step)
        right_arrow = Arrow([(win_width//2, win_height//2), (win_width//2+100, win_height//2-60), (win_width//2+100, win_height//2+60)],
                            horizontal_angle_threshold_step)
        up_arrow = Arrow([(win_width//5-40, win_height//5-100), (win_width//5-100, win_height//5), (win_width//5+20, win_height//5)],
                        height_threshold_step)
        down_arrow = Arrow([(win_width//5-40, win_height//5+120), (win_width//5-100, win_height//5+20), (win_width//5+20, win_height//5+20)],
                        -height_threshold_step)
        # Group the arrows
        self.arrows = Arrows([forward_arrow, back_arrow, left_arrow, right_arrow, up_arrow, down_arrow])


    def normalize_signals(self):
        self.RL_angle = normalize_angle(self.RL_angle)
        self.FB_angle = normalize_angle(self.FB_angle)
        self.hand_height = normalize_height(self.hand_height)

    def calculate_signal(self, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                # Calculate the angle between the tumb and pinky
                self.RL_height = hand_landmarks.landmark[5].z - hand_landmarks.landmark[17].z
                self.RL_width = hand_landmarks.landmark[5].x - hand_landmarks.landmark[17].x
                self.RL_angle = -np.arctan(self.RL_height / self.RL_width)

                # Calculate the angle between the middle finger and wrist
                self.FB_height = hand_landmarks.landmark[9].z - hand_landmarks.landmark[0].z
                self.FB_width = hand_landmarks.landmark[9].y - hand_landmarks.landmark[0].y
                self.FB_angle = np.arctan(self.FB_height / self.FB_width)


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
        self.signal = [self.FB_angle, self.RL_angle, self.hand_height]

        # Print information
        self.arrows.print_thresholds()
        print(self.signal)

    def send_signal(self):
        # Send the signal to the arrows and drone
        self.arrows.recive_signal(self.signal)
        if self.drone_control:        
            self.drone.send_rc_control(self.arrows.get_signal())
        
    def draw(self, window):
        self.arrows.draw(window)

    def calibrate_vertical_arrows(self):
        self.arrows.calibrate_vertical_arrows(self.signal)

    def calibrate_horizontal_arrows(self):
        self.arrows.calibrate_horizontal_arrows(self.signal)

    


    
