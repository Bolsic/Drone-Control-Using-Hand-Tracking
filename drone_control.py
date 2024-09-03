import pygame
from djitellopy import Tello
import sys
import logging
#Tello.LOGGER.setLevel(logging.DEBUG)

start_counter = 1 # 0 for flight 1 for testing
takeoff = False

# CONNECT TO DRONE
me = Tello()
me.connect()
me.for_back_velocity = 0
me.left_right_velocity = 0
me.up_down_velocity = 0
me.yaw_velocity = 0
me.speed = 0

speed = 30
rotation_speed = 40

print("Current Battery: ", me.get_battery())

screen = win_width, win_height = 600, 400
clock = pygame.time.Clock()
window = pygame.display.set_mode((win_width, win_height))
pygame.display.set_caption("Ekran")

left_right_velocity, forward_backward_velocity, up_down_velocity = 0, 0, 0
rotation_velocity = 0

# me.streamoff()
# me.streamon()

run = True
while run: 
    clock.tick(10)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    keys = pygame.key.get_pressed()

    if keys[pygame.K_SPACE]:
        if takeoff:
            takeoff = False
            print("Landing")
            me.land()
        else:
            takeoff = True
            print("Taking off")
            me.takeoff()

    if keys[pygame.K_q]:
        me.land()
        pygame.quit()
        sys.exit()

    if keys[pygame.K_UP]:
        forward_backward_velocity = speed
    if keys[pygame.K_DOWN]:
        forward_backward_velocity = -speed
    if keys[pygame.K_RIGHT]:
        left_right_velocity = -speed
    if keys[pygame.K_LEFT]:
        left_right_velocity = speed
    if keys[pygame.K_a]:
        rotation_velocity = rotation_speed
    if keys[pygame.K_d]:
        rotation_velocity = -rotation_speed
    if keys[pygame.K_w]:
        up_down_velocity = speed
    if keys[pygame.K_s]:
        up_down_velocity = -speed

    me.send_rc_control(left_right_velocity, forward_backward_velocity, up_down_velocity, rotation_velocity)
    
me.land()
