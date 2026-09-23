try:
    import pygame
except ImportError:  # Physics-only experiments do not need the GUI dependency.
    pygame = None
import math
import numpy as np

from scipy.integrate import ode

# Env Configuration
BLACK = (0,0,0)
DARK_RED = (150, 0, 0)
RADIUS = 7
CART_WIDTH = 60
CART_HEIGHT = 30
POLE_LENGTH = 100
SCALE_X = 100.0
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 400

# A simple class to simulate cartpole physics using an ODE solver
class CartPole(object):

    # State holds x, x_dot, theta_dot, theta (radians)
    def __init__(self, x0=[0.0, 0.0, 0.0, np.pi], initial_offset=0.0):
        self.g = 9.82
        self.m = 0.5
        self.M = 0.5
        self.l = 0.5
        self.b = 1.0

        self.x0 = np.array(x0, dtype=np.float64).flatten()
        self.x = self.x0
        self.x[3] = self.x[3] + initial_offset
        self.t = 0

        self.u = 0

        # This is a key line that makes this class an accurate version of cartpole dynamics.
        # The ODE solver is connected with our instantaneous dynamics equations so it can do
        # the hard work of computing the motion over time for us.
        self.solver = ode(self.dynamics).set_integrator('dopri5', atol=1e-12, rtol=1e-12)
        self.set_state(self.x)

    # For internal use. This connects up the local state in the class
    # with the variables used by our ODE solver.
    def set_state(self, x):
        if (self.x is None or np.linalg.norm(x-self.x) > 1e-12):
            self.x = np.array(x,dtype=np.float64).flatten()
        self.solver = self.solver.set_initial_value(self.x)
        self.t = self.solver.t

    # Convenience function. Allows for quickly resetting back to the initial state to
    # get a clearer view of how the control works.
    def reset(self, x0=None):
        self.x = self.x0 if x0 is None else x0
        self.t = 0
        self.set_state(self.x)

    # Draw the cart and pole
    def draw(self, bg):
        cart_centre = (int(SCREEN_WIDTH/2+self.x[0]*SCALE_X), int(SCREEN_HEIGHT/2))
        pole_end = (int(cart_centre[0] + POLE_LENGTH * math.sin(self.x[3])), int(cart_centre[1]+ POLE_LENGTH*math.cos(self.x[3])))
        pygame.draw.rect(bg, BLACK, [cart_centre[0]-CART_WIDTH/2, cart_centre[1]-CART_HEIGHT/2, CART_WIDTH, CART_HEIGHT])
        pygame.draw.lines(bg, BLACK, False, [cart_centre, pole_end], 2)
        pygame.draw.circle(bg, DARK_RED, cart_centre, RADIUS - 2)
        pygame.draw.circle(bg, DARK_RED, pole_end, RADIUS)

    # These equations are simply typed in from the dynamics
    # on the assignment document. They have been derived
    # for a pole of uniform mass using the Lagrangian method.
    def dynamics(self,t,z):

        f = np.array([self.u])

        sz = np.sin(z[3])
        cz = np.cos(z[3])
        cz2 = cz*cz

        a0 = self.m*self.l*z[2]*z[2]*sz
        a1 = self.g*sz
        a2 = f[0] - self.b*z[1]
        a3 = 4*(self.M+self.m) - 3*self.m*cz2

        dz = np.zeros((4,1))
        dz[0] = z[1]                                                            # x
        dz[1] = (  2*a0 + 3*self.m*a1*cz + 4*a2 )/ ( a3 )                       # dx/dt
        dz[2] = -3*( a0*cz + 2*( (self.M+self.m)*a1 + a2*cz ) )/( self.l*a3 )   # dtheta/dt
        dz[3] = z[2]                                                            # theta

        return dz

    # Takes the command, u, and applies it to the system for dt seconds.
    # Note that the solver has already been connected to the dynamics
    # function in the constructor, so this function is effectively
    # evaluating the dynamics. The solver does this in an "intelligent" way
    # that is more accurate than dt * accel, rather it evaluates the dynamics
    # at several points and correctly integrates over time.
    def step(self, u, dt=None):

        self.u = u

        if dt is None:
            dt = 0.005
        t1 = self.solver.t + dt
        while self.solver.successful and self.solver.t < t1:
            self.solver.integrate(self.solver.t+ dt)
        self.x = np.array(self.solver.y)
        self.t = self.solver.t
        return self.x

    def get_state(self):
        return self.x


# A simple class to simulate double cartpole physics using an ODE solver
"""The meaning of the state dimensions are:
                x[0] : cart position (x)
                x[1] : cart velocity (x_dot)
                x[2] : pole 1 angular velocity
                x[3] : pole 2 angular velocity
                x[4] : pole 1 angle
                x[5] : pole 2 angle
"""
class DoubleCartPole(object):

    def __init__(self, x_init=[0.0, 0.0, 0.0, 0.0, np.pi, np.pi], initial_offset=0.0):
        
        self.x_init = x_init
        self.x_init[4] = self.x_init[4] + initial_offset
        self.x_init[5] = self.x_init[5] + initial_offset
        self.g = 9.82
        self.m = 0.5
        self.M = 0.5
        self.l = 0.5
        self.b = 1.0

        self.x0 = np.array(x_init, dtype=np.float64).flatten()
        self.x = self.x0
        self.t = 0

        self.u = 0

        # This is a key line that makes this class an accurate version of cartpole dynamics.
        # The ODE solver is connected with our instantaneous dynamics equations so it can do
        # the hard work of computing the motion over time for us.
        self.solver = ode(self.dynamics).set_integrator('dopri5', atol=1e-12, rtol=1e-12)
        self.set_state(self.x)

    # For internal use. This connects up the local state in the class
    # with the variables used by our ODE solver.
    def set_state(self, x):
        if (self.x is None or np.linalg.norm(x-self.x) > 1e-12):
            self.x = np.array(x,dtype=np.float64).flatten()
        self.solver = self.solver.set_initial_value(self.x)
        self.t = self.solver.t

    # Convenience function. Allows for quickly resetting back to the initial state to
    # get a clearer view of how the control works.
    def reset(self, x0=None):
        self.x = self.x0 if x0 is None else x0
        self.t = 0
        self.set_state(self.x)

    # Draw the cart and pole
    def draw(self, bg):
        cart_centre = (int(SCREEN_WIDTH/2+self.x[0]*SCALE_X), int(SCREEN_HEIGHT/2))
        pole_end1 = (int(cart_centre[0] + POLE_LENGTH * math.sin(self.x[4])), int(cart_centre[1]+ POLE_LENGTH*math.cos(self.x[4])))

        pole_end2 = (int(pole_end1[0] + POLE_LENGTH * math.sin(self.x[5])), int(pole_end1[1]+ POLE_LENGTH*math.cos(self.x[5])))

        pygame.draw.rect(bg, BLACK, [cart_centre[0]-CART_WIDTH/2, cart_centre[1]-CART_HEIGHT/2, CART_WIDTH, CART_HEIGHT])
        pygame.draw.lines(bg, BLACK, False, [cart_centre, pole_end1], 2)
        pygame.draw.lines(bg, BLACK, False, [pole_end1, pole_end2], 2)
        pygame.draw.circle(bg, DARK_RED, cart_centre, RADIUS - 2)
        pygame.draw.circle(bg, DARK_RED, pole_end1, RADIUS)
        pygame.draw.circle(bg, DARK_RED, pole_end2, RADIUS)

    # These equations are simply typed in from the dynamics
    # on the assignment document. They have been derived
    # for a pole of uniform mass using the Lagrangian method.
    def dynamics(self,t,z):

        f = np.array([self.u])

        # set up the system
        m1 = 0.5;  # [kg]     mass of cart
        m2 = 0.5;  # [kg]     mass of 1st pendulum
        m3 = 0.5;  # [kg]     mass of 2nd pendulum
        l2 = 0.6;  # [m]      length of 1st pendulum
        l3 = 0.6;  # [m]      length of 2nd pendulum
        b  = 0.1;  # [Ns/m]   coefficient of friction between cart and ground
        g  = -9.82; # [m/s^2]  acceleration of gravity

        A = np.asmatrix( [ [ 2*(m1+m2+m3), -(m2+2*m3)*l2*math.cos(z[4]), -m3*l3*math.cos(z[5])],
                      [ -(3*m2+6*m3)*math.cos(z[4]), (2*m2+6*m3)*l2, 3*m3*l3*math.cos(z[4]-z[5])],
                      [  -3*math.cos(z[5]), 3*l2*math.cos(z[4]-z[5]), 2*l3] ] )

        b = np.asmatrix( [ [ 2*f[0]-2*b*z[1]-(m2+2*m3)*l2*z[2]*z[2]*math.sin(z[4])-m3*l3*z[3]*z[3]*math.sin(z[5])],
       [(3*m2+6*m3)*g*math.sin(z[4])-3*m3*l3*z[3]*z[3]*math.sin(z[4]-z[5])],
       [3*l2*z[2]*z[2]*math.sin(z[4]-z[5])+3*g*math.sin(z[5])  ] ]   )

        x = np.linalg.solve(A,b);

        dz = np.zeros((6,1))
        dz[0] = z[1]                                                            # x
        dz[1] = x[0]                                                            # dx/dt
        dz[2] = x[1]                                                            # dtheta1/dt
        dz[3] = x[2]                                                            # dtheta2/dt
        dz[4] = z[2]                                                            # theta1
        dz[5] = z[3]                                                            # theta2

        return dz

    # Takes the command, u, and applies it to the system for dt seconds.
    # Note that the solver has already been connected to the dynamics
    # function in the constructor, so this function is effectively
    # evaluating the dynamics. The solver does this in an "intelligent" way
    # that is more accurate than dt * accel, rather it evaluates the dynamics
    # at several points and correctly integrates over time.
    def step(self,u,dt=None):

        self.u = u

        if dt is None:
            dt = 0.005
        t1 = self.solver.t + dt
        while self.solver.successful and self.solver.t < t1:
            self.solver.integrate(self.solver.t+ dt)
        self.x = np.array(self.solver.y)
        self.t = self.solver.t
        return self.x

    def get_state(self):
        return self.x
