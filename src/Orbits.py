"""
Simulation of orbits of a planet.

Authors: Akhil Premkumar
Date: September 23rd, 2015

References:
[1] http://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html
[2] http://matplotlib.org/examples/pylab_examples/polar_demo.html
"""

import matplotlib
matplotlib.use('TKAgg')

import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time

class Planet:
    def __init__(self,
                 name='XYZ',
                 init_state = [0, 1, 0], # This is [r', r, phi]. Begin from 1 at rest, by default
                 l = 1,
                 m = 2,
                 C = 1):
        
        self.name = name
        self.state = np.asarray(init_state, dtype='float')
        self.params = (l, m, C)
        
        self.time_elapsed = 0
        
    def dstate_dt(self, y, t):
        """
        This function computes the derivative of the state vector y.
        
        For the equation: mr'' = F(r) + l^2/m*r^3 , we take
        y[0] = r', the radial velocity of the planet
        y[1] = r, the radial position of the planet
        y[2] = phi
        
        and its derivative is:
        y[0]' = (F(y[1]) + l^2/m*y[1]^3)/m
        y[1]' = y[0]
        y[2]' = l/m*y[1]^2
        """
        (l, m, C) = self.params
        F = -C*y[1]**6;
        return [(F + (l*l)/(m*y[1]*y[1]*y[1]))/m, y[0], l/(m*y[1]*y[1])]
    
    def step(self, dt):
        """
        This function updates the state.
        
        We use 'odeint' method from SciPy which will give us the precise change
        in our state vector no matter the time interval dt. The latter just needs
        to be small enough to let us observe smooth oscillations.
        """
        self.state = integrate.odeint(self.dstate_dt, self.state, [0, dt])[1]
        self.time_elapsed += dt
#         print self.state, self.time_elapsed

#------------------------------------------------------------

class Instrument:
    """
    This class monitors the planet and charts the time evolution of its state.
    """
    def __init__(self, planet): # Time window in seconds
        self.planet = planet
        self.radial_arr = []
        self.theta_arr = []
#         self.time_arr = []
        
        # Set up the plot and animation
        self.fig = plt.figure()
        self.fig.suptitle('Motion of %s' % planet.name)
        bob_ax = plt.subplot(111, polar=True)
#         bob_ax.set_title('Bob representation', {'fontsize': 12})
        bob_ax.set(aspect='equal', autoscale_on=False) # <--
        bob_ax.set_rmax(2) # <--
        bob_ax.grid(True)        
        # bob_pos, for tuple unpacking. See http://matplotlib.org/users/pyplot_tutorial.html
        self._bob_pos, = bob_ax.plot([], [], 'o', markersize=8)
        self._bob_traj, = bob_ax.plot([], [], '-r')
        # The default transform specifies that text is in data coords, change it to axes coords
        self._bob_time_text = bob_ax.text(0.05, 0.95, '', transform=bob_ax.transAxes, color='black', bbox = {'fill': False, 'linewidth': 1})
        
    def measure(self):        
        self.radial_arr.append(self.planet.state[1])
        self.theta_arr.append(self.planet.state[2])

    def init_anim(self):
        """initialize animation"""
        self._bob_pos.set_data([], [])
        self._bob_traj.set_data([], [])
        self._bob_time_text.set_text('')
        
        return self._bob_pos, self._bob_traj, self._bob_time_text

    def animate(self, i):
        """perform animation step"""
        global planet, dt        
        planet.step(dt)
        
        instrument.measure()
        
        self._bob_pos.set_data([self.planet.state[2]], [self.planet.state[1]])
        self._bob_traj.set_data(self.theta_arr, self.radial_arr) # theta, r
        self._bob_time_text.set_text('time = %.1f s' % planet.time_elapsed)
        
        # Return the changed objects so that blit animates the portions which have changed
        return self._bob_pos, self._bob_traj, self._bob_time_text

#------------------------------------------------------------
# Set up the planet

planet = Planet('Earth', init_state = [0.2, 1, 0], l=1, m=1, C=1) # F = -C*y[1]**6;
# planet = Planet('Earth', init_state = [0.1, 2, 0], l=2, m=2, C=1) # F = -C/(y[1]**2);

dt = 1./20 # 50 ms

instrument = Instrument(planet)

#------------------------------------------------------------

# choose the interval based on dt and the time to animate one step
t0 = time()
instrument.animate(0)
t1 = time()
interval = 1000 * dt - (t1 - t0)

ani = animation.FuncAnimation(instrument.fig, instrument.animate, frames=300,
                              interval=interval, blit=True, init_func=instrument.init_anim)

figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
plt.show()