import pygame
from djitellopy import Tello
import sys
import logging

# CONNECT TO DRONE
me = Tello()
me.connect()
me.for_back_velocity = 0
me.left_right_velocity = 0
me.up_down_velocity = 0
me.yaw_velocity = 0
me.speed = 0

speed = 170
rotation_speed = 40
takeoff = False

print("Current Battery: ", me.get_battery())

# Set up the pygame window
screen = win_width, win_height = 600, 400
clock = pygame.time.Clock()
window = pygame.display.set_mode((win_width, win_height))
pygame.display.set_caption("Screen")

left_right_velocity, forward_backward_velocity, up_down_velocity = 0, 0, 0
rotation_velocity = 0

run = True
while run: 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    keys = pygame.key.get_pressed()

    # SPACE is for takeoff and land
    if keys[pygame.K_SPACE]:
        if takeoff:
            takeoff = False
            print("Landing")
            me.land()
        else:
            takeoff = True
            print("Taking off")
            me.takeoff()

    # Q is for quit
    if keys[pygame.K_q]:
        me.land()
        pygame.quit()
        sys.exit()

    # Reset the velocities
    left_right_velocity, forward_backward_velocity, up_down_velocity, rotation_velocity = 0, 0, 0, 0

    # Change the velocities depending on the keys pressed
    if keys[pygame.K_UP]:
        forward_backward_velocity = speed
    if keys[pygame.K_DOWN]:
        forward_backward_velocity = -speed
    if keys[pygame.K_RIGHT]:
        left_right_velocity = speed
    if keys[pygame.K_LEFT]:
        left_right_velocity = -speed
    if keys[pygame.K_a]:
        rotation_velocity = rotation_speed
    if keys[pygame.K_d]:
        rotation_velocity = -rotation_speed
    if keys[pygame.K_w]:
        up_down_velocity = speed
    if keys[pygame.K_s]:
        up_down_velocity = -speed
    
    # Send the velocities to the drone
    print(left_right_velocity, forward_backward_velocity, up_down_velocity, rotation_velocity)
    me.send_rc_control(left_right_velocity, forward_backward_velocity, up_down_velocity, rotation_velocity)
    
# Land before finishing the program
me.land()
