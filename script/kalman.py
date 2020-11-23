import csv
import numpy as np
import matplotlib.pyplot as plt

# define constants

dT = 0.001
noise = 0.001

# define the constant matrices for the kalman filter:

# process matrix or model dynamics
F = np.eye(N = 7, M = 7)

# define the input dynamics
B = np.eye(N = 7, M = 7) * dT

# input matrix from measurements
H = np.eye(N = 7, M = 7)

# covariance matrix for the "entire" noise in the system
P = np.eye(N = 7, M = 7)

# measurement noise, assume high measurement noise
R = np.load('settings/cov_mat.npy')

# process noise
Q = np.eye(N = 7, M = 7) * noise ** 3

class data_loader:

    def __read_data__(self, data):
        temp = list(data)
        return temp[0], np.array(temp[1:]).astype("float")

    def __init__(self, q, qdot, qddot_d):
        
        self.q_name  = []
        self.q_data  = []

        for i, data in enumerate([q, qdot, qddot_d]):

            q_name_r, q_data_r = self.__read_data__(csv.reader(open(data, "r"), delimiter=","))

            self.q_name.append(q_name_r)
            self.q_data.append(q_data_r)
        
        self.q_name = np.asarray(self.q_name)
        self.q_data = np.asarray(self.q_data)

    def row(self, row):
        return self.q_data[0,row,:].reshape((-1,1)), self.q_data[1,row,:].reshape((-1,1)), self.q_data[2,row,:].reshape((-1,1))
    
    def name(self):
        return self.q_name
    
    def len(self):
        return self.q_data.shape[1]

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
        # u is a 7x1 vector of acceleration predictions
        self.x = self.F @ self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def correct(self, meas):
        # measure is a 7x1 vector of velocities measured from sensors

        # determine residuals and kalman gain
        y = meas - self.H @ self.x
        self.K = self.P @ self.H.T @ np.linalg.pinv( self.R + self.H @ self.P @ self.H.T )

        # correct the state
        self.x = self.x + self.K @ y
        self.P = (np.eye(self.K.shape[0], H.shape[1]) - self.K @ self.H) @ self.P

    def get_state(self):
        return self.x, self.P, self.K

class plotting:
    
    def __init__(self, name, len):
        
        self.name  = name
        self.data  = np.zeros((name.shape[0], name.shape[1], len))
    
    def insert(self, data, row, col = 1):
        self.data[col, :, row] = data
    
    def plot(self, col = 1):
        
        fig, axs = plt.subplots(7)

        for i in range(7):
            axs[i].plot(self.data[col, i, :])

        plt.show()
    


# init the data loader
loader = data_loader("data_osc/q.csv", "data_osc/qdot.csv", "data_osc/qddot_d.csv")

# define kalman filter parameters
qdot_pred  = np.zeros((7,1))

# define the kalman filter
kalman = kalman_filter(qdot_pred, F, B, H, P, Q, R)

# plotting purposes
plot_tool = plotting(loader.name(), loader.len())

# for loop for the kalman filter to work
for i in range(0, loader.len()):

    # read the first set of measurement
    q_mes, qdot_mes, q_ddot_d = loader.row(i)

    # do the kalman prediction
    kalman.prediction(q_ddot_d)
    kalman.correct(qdot_mes)

    # update the prediced steta and K etc
    qdot_pred, P, K = kalman.get_state()

    # insert the data
    plot_tool.insert(qdot_pred.reshape((-1)), i, col = 1)

plot_tool.plot()