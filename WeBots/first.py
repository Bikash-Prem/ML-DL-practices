import sys
sys.path.append("C:/Program Files/Webots/lib/controller/python")

from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

while robot.step(timestep) != -1:

    # RL will choose this later
    left_speed = 2
    right_speed = 2

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)