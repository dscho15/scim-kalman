import numpy as np

# define constants

dT = 0.001
noise = 0.001

# define the constant matrices for the kalman filter:

# process matrix or model dynamics
F = np.eye(N = 6, M = 6)

# define the input dynamics
B = np.eye(N = 6, M = 6) * dT

# input matrix from measurements
H = np.eye(N = 6, M = 6)

# covariance matrix for the "entire" noise in the system
P = np.eye(N = 6, M = 6)

# measurement noise, assume high measurement noise
R = np.eye(N = 6, M = 6) * noise

# process noise
Q = np.eye(N = 6, M = 6) * noise*noise

class kalman_filter:
    def __init__(self, x0, F, B, H, P, Q, R):
        # matrices for the kalman-filter
        self.F = F
        self.B = B
        self.H = H
        self.P = P
        self.Q = Q
        self.R = R

        # initial state
        self.x = x0
    
    def prediction(self, u):
        # u is a 6x1 vector of acceleration predictions
        self.x = self.F * self.x + self.B * u
        self.P = self.F * self.P * self.F.T + self.Q
    
    def correct(self, meas):
        # measure is a 6x1 vector of velocities measured from sensors

        # determine residuals and kalman gain
        y = meas - self.H * self.x
        self.K = self.P * self.H.T * np.linalg.pinv( self.R + self.H * self.P * self.H.T )

        # correct the state
        self.x = self.x + self.K*y
        self.P = (np.eye(self.K.shape[0], H.shape[1]) - self.K * self.H) * self.P

    def get_state(self):
        return self.x, self.P, self.K

x0 = np.ones((6,1))

kalman = kalman_filter(x0, F, B, H, P, Q, R)

u = np.zeros((6,1))

kalman.prediction(u)
kalman.correct(x0)
kalman.get_state()

