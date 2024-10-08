import pygame

class Arrow:
    def __init__(self, coordinates, threshold_step, color=(200, 200, 200)):
        self.coordinates = coordinates
        self.color = color
        self.threshold = threshold_step
        self.threshold_step = threshold_step
        self.activated = False

    def draw(self, window):
        pygame.draw.polygon(window, self.color, self.coordinates)

    def calibrate(self, signal):
        self.threshold = signal + self.threshold_step
    
    def activate(self, signal):
        opacity = int(abs(signal-self.threshold) * 255)
        if opacity > 255: opacity = 255
        if opacity < 0: opacity = 0
        self.color = (opacity, opacity, 0)
        self.activated = True

    def deactivate(self):
        self.color = (0, 0, 0)
        self.activated = False

    def recive_signal(self, signal):
        # if the signal and threshold have the same sign and 
        if self.threshold_step > 0 and signal > self.threshold:
            self.activate(signal)
        elif self.threshold_step < 0 and signal < self.threshold:
            self.activate(signal)
        else: 
            self.deactivate()
            

class Arrows:
    # This class will be used to group all the arrows
    # forvard, back, left, right, up, down
    def __init__(self, arrows):
        self.arrows = arrows
    
    def draw(self, window):
        for arrow in self.arrows:
            arrow.draw(window)

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
        for i in range(4):
            self.arrows[i].recive_signal(signal[i//2])

        # If any of the horizontal arrows are activated deactivate the vertical arrows
        any_horizontal_activated = False
        for i in range(4):
            if self.arrows[i].activated:
                any_horizontal_activated = True
                break
        
        if not any_horizontal_activated:
            self.arrows[4].recive_signal(signal[2])
            self.arrows[5].recive_signal(signal[2])
        else:
            self.arrows[4].deactivate()
            self.arrows[5].deactivate()
    
    def activated_axis(self):
        # Return the activated axis
        activated_list = []
        for i in range(6):
            activated_list.append(self.arrows[i].activated)
        return activated_list

    def print_thresholds(self):
        print("Thresholds:")
        print("Forward: ", self.arrows[0].threshold)
        print("Backward: ", self.arrows[1].threshold)
        print("Left: ", self.arrows[2].threshold)
        print("Right: ", self.arrows[3].threshold)
        print("Up: ", self.arrows[4].threshold)
        print("Down: ", self.arrows[5].threshold)
        