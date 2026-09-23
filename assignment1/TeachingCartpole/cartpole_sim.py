'''
This file holds a cartpole simulator using physics functions borrowed from a previous
research project. Those are:
Copyright (c) 2017, Juan Camilo Gamboa Higuera, Anqi Xu, Victor Barbaros, Alex Chatron-Michaud, David Meger

The GUI is new in 2020 and was started from the pendulum code of Wesley Fernandes
https://pastebin.com/zTZVi8Yv
python simple pendulum with pygame

The rest of the file and instructions are written by David Meger for the purposes of supporting
his teaching in RL and Robotics. Please use this freely for any purpose, but acknowledgement of
sources is always welcome.
'''

import pygame
import csv
import argparse
import logging
import numpy as np

from datetime import datetime

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

from cartpole_envs import CartPole, DoubleCartPole
from cartpole_control import Controller

# The very basic code you should know and interact with starts here. Sets some variables that you
# might change or add to, then defines a function to do control that is currently empty. Add
# more logic in and around that function to make your controller work/learn!


# After this is all the code to run the cartpole physics, draw it on the screen, etc.
# You should not have to change anything below this, but are encouraged to read and understand
# as much as possible.

# set the width and height of the window
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 400 # keep even numbers
WHITE = (255,255,255)
GRAY = (150, 150, 150)

Done = False                # if True,out of while loop, and close pygame
Pause = False               # when True, freeze the pendulum. This is
                            # for debugging purposes

# Initialize GUI
pygame.init()
background = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

# The next two are just helper functions for the display.
# Draw a grid behind the cartpole
def grid():
    for x in range(50, SCREEN_WIDTH, 50):
        pygame.draw.lines(background, GRAY, False, [(x, 0), (x, SCREEN_HEIGHT)])
        for y in range(50, SCREEN_HEIGHT, 50):
            pygame.draw.lines(background, GRAY, False, [(0, y), (SCREEN_WIDTH, y)])

# Clean up the screen and draw a fresh grid and the cartpole with its latest state coordinates
def redraw():
    background.fill(WHITE)
    grid()
    pendulum.draw(background)
    pygame.display.update()

# Setup for arg parser
def setup_parser():
    parser = argparse.ArgumentParser()


    # Env arguments
    parser.add_argument('--env', type=str, default='CartPole',
        help='Switch between single and double cartpole envs: CartPole, DoubleCartPole.')
    parser.add_argument('--refresh-rate', type=int, default=240, help='GUI refresh rate.')
    parser.add_argument('--control-rate', type=int, default=1, help='Control input will be applied once per timestep by default!')
    parser.add_argument('--initial-offset', type=float, default=np.pi/40.0,
            help='Set an initial offset for the pole angle.')

    # Data collection tools
    parser.add_argument('--data-collection', action='store_true', default=False,
                        help='Log data in a CSV file, later to be used for learning dynamics and more! This will override the control inputs.')
    parser.add_argument('--sampling-rate', type=int, default=1,
                        help='Samples are taken at each step by default.')
    parser.add_argument('--reset-rate', type=int, default=2000,
                        help='Resets the environment if data collection mode is active.')
    parser.add_argument('--dataset-size', type=int, default=1e6,
                        help='Size of the collected dataset.')
    parser.add_argument('--control-min', type=int, default=-10,
                        help='Minimum control input for sampling from a uniform distribution.')
    parser.add_argument('--control-max', type=int, default=10,
                        help='Maximum control input for sampling from a uniform distribution.')
    parser.add_argument('--dataset-name', type=str, default='dynamics-data',
        help='Pick a name for the collected dataset.')
    
    return parser

# Save a numpy array of data
def save_data(file_name: str, data: np.ndarray):
    # Write the data to a CSV file
    with open(file_name + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write data
        for row in data:
            writer.writerow(row)


if __name__ == '__main__':
    args = setup_parser().parse_args()

    # Starting here is effectively the main function.
    # It's a simple GUI drawing loop that calls to your code to compute the control, sets it to the
    # cartpole class and loops the GUI to show what happened.
    pendulum = CartPole(initial_offset=args.initial_offset) if args.env == 'CartPole' else DoubleCartPole(initial_offset=args.initial_offset)
    state = pendulum.get_state()
    controller = Controller()

    # Data collection vars
    if args.data_collection:
        logger.warning('Data collection mode enabled! This will overriding usual control inputs.')
        data = []
        data_index = 0

    # Time step and control vars
    time_step = 0
    control = 0

    while not Done:
        time_step += 1

        clock.tick(args.refresh_rate) # GUI refresh rate

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                Done = True
            if event.type == pygame.KEYDOWN:    # "r" key resets the simulator
                if event.key == pygame.K_r:
                    pendulum.reset()
                if event.key == pygame.K_p:     # holding "p" key freezes time
                    Pause = True
            if event.type == pygame.KEYUP:      # releasing "p" makes us live again
                if event.key == pygame.K_p:
                    Pause = False

        if not Pause:
            # Set the default control

            if args.data_collection:
                # Control in data collection mode
                if time_step % args.control_rate == 0:
                    control = np.random.uniform(low=args.control_min, high=args.control_max)
                    logger.debug(f'Time step {time_step}, applied control is {control}')

                # Reset rate for data collection, this resets the pendulum to the initial state
                if time_step % args.reset_rate == 0:
                    pendulum.reset()

            else:
                # Normal operation mode
                # TODO: This is the call to the code you write
                if time_step % args.control_rate == 0:
                    control = controller.compute_control(state)

            # Step simulation
            next_state = pendulum.step(control)

            # Save the (s', s, u) tuple
            if time_step % args.sampling_rate == 0 and args.data_collection:
                data.append(np.concatenate((next_state, state, np.array(control).reshape(1,))))

                data_index += 1
                if data_index >= args.dataset_size:
                    break

            # Transition to the next state
            state = next_state

            # Redraw the GUI
            redraw()

    # Save the collected data
    if args.data_collection:
        save_data(file_name=args.dataset_name, data=np.array(data))

    # Quit
    pygame.quit()