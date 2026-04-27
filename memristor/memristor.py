import numpy as np
from typing import Optional
import random

class Memristor:
    def __init__(self, cur_func, plast_func, obs_func, window_pts, dt, w0: Optional[float] = None, Q_history: Optional[np.ndarray] = None):

        self.dt = dt
        self.window_pts = window_pts

        
        # Setup weights
        
        if w0 is None:
            self.w = random.random()
        else:
            self.w = w0
        
        # history setup

        if window_pts <= 0:
            raise ValueError("window_pts must be positive")

        if Q_history is None:
            self.Q_history = np.zeros(self.window_pts)
        elif np.isscalar(Q_history):
            self.Q_history = np.full(self.window_pts, Q_history)
        else:
            if len(Q_history) == self.window_pts:
                self.Q_history = np.array(Q_history)
            else:
                raise ValueError("Wrong history record dimension")
        
        
        
        self.cur = cur_func
        self.plast = plast_func
        self.obs = obs_func

        self.running_sum = np.sum(self.Q_history)
        self.idx = 0


    def current(self, V):
        return self.cur(V, self.w)

    

    def update_window(self, Q):
        old = self.Q_history[self.idx]
        
        # update running sum
        self.running_sum += Q - old
        
        # overwrite oldest value
        self.Q_history[self.idx] = Q

        # move pointer
        self.idx = (self.idx + 1) % self.window_pts

    
    def Q_avg(self):
        return self.running_sum / self.window_pts

    def dw_dt(self):
        Q_av = self.Q_avg()
        return self.plast(Q_av, self.w)

    def step(self, V):
        Q = self.obs(V, self.current(V))

        # update memory
        self.update_window(Q)

        # evolve state
        self.w += self.dt * self.dw_dt()

        # enforce bounds
        self.w = np.clip(self.w, 0.0, 1.0)