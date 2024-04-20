# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 17:03:19 2024

@author: Ri Yang
"""

# --- import ---
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# --- functions ---
def odes(t, Y, m1, m2, l1, l2, g):
    # Y = phi1, phi2, w1, w2 
    phi1, phi2, w1, w2 = Y[0], Y[1], Y[2], Y[3]
    A = np.array([[(m1+m2)*l1, m2*l2*np.cos(phi1-phi2)],
                  [m2*l1*np.cos(phi1-phi2), m2*l2]])
    b = np.array([-1*(m1+m2)*g*np.sin(phi1)-m2*l2*np.sin(phi1-phi2)*w2*w2,
                  m2*l1*np.sin(phi1-phi2)*w1*w1-m2*g*np.sin(phi2)])
    dw1dt, dw2dt = np.linalg.solve(A, b)
    dydt = np.array([w1, w2, dw1dt, dw2dt]) 
    return dydt

def animate(i):
    # position
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    # trajectory
    history_x = x2[:i]
    history_y = y2[:i]

    line.set_data(thisx, thisy)
    trace.set_data(history_x[-20:], history_y[-20:]) # choose, for example, the last 20 points
    time_text.set_text(time_template % (i*dt)) 

    return line, trace, time_text


# --- main ---
if __name__ == "__main__":
    # parameters
    m1, m2 = 1, 1 # mass
    l1, l2 = 1, 1 # length  
    g = 9.8 # gravity acceleration
    
    # set initial conditions
    phi10, phi20 =  90, 30 # initial angle 
    w10, w20 = 0, 0 # initial angular velocity, w = phi dot
    y0 = np.radians([phi10, phi20, w10, w20]) 
    
    dt = 0.04 # time step in s / s per frame
    t_span = (0, 10) # time span / real time
    t_eval = np.arange(0, 10, dt) # time eval / frames

    # solve ivp
    res = solve_ivp(odes, t_span, y0, t_eval=t_eval, args=(m1, m2, l1, l2, g))
    
    # coordinates transformation to Cartesian
    x1, y1 = l1 * np.sin(res.y[0]), -l1 * np.cos(res.y[0])
    x2, y2 = x1 + l2 * np.sin(res.y[1]), y1 - l2 * np.cos(res.y[1])
    
    # animation
    fig = plt.figure(figsize=(4, 3))
    L = l1 + l2 + 1 # limit
    ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, 1))
    ax.set_aspect('equal') 
    ax.grid(linestyle='-.') 
    
    ax.plot(0, 0, 'rx', ms=10) # pin point
    line, = ax.plot([], [], 'o-', lw=2) # line and pendulum
    trace, = ax.plot([], [], '.-', lw=1, ms=2) # trajectory of pendulum
    time_template = 'time = %.1f s'
    time_text = ax.text(0.05, 0.9, '', alpha=0.8, transform=ax.transAxes)
    
    ani = animation.FuncAnimation(
        fig, animate, len(t_eval), interval=dt*1000, blit=True)
    plt.show()
    
    # ani.save("DoublePendulum.gif")
