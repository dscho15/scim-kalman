import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

# define constants

dT = 0.001
noise = 0.1

# define the constant matrices for the kalman filter:

# process matrix or model dynamics
F = np.eye(N = 7, M = 7)

# define the input dynamics
B = np.eye(N = 7, M = 7) * dT

# input matrix from measurements
H = np.eye(N = 7, M = 7)

# covariance matrix for the "entire" noise in the system
P = np.eye(N = 7, M = 7) * 199

# measurement noise, assume high measurement noise
R = np.load('settings/cov_mat.npy')
print(np.diag(R))


# process noise
Q = R * 10000000

# Q = R / 10

Q[0,0] = Q[0,0] / 1
Q[1,1] = Q[1,1] / 1
Q[2,2] = Q[2,2] / 1
Q[3,3] = Q[3,3] / 1
Q[4,4] = Q[4,4] / 1
Q[5,5] = Q[5,5] / 1
Q[6,6] = Q[6,6] / 1

class data_loader:

    def __read_data__(self, data):
        temp = np.array(list(data))
        temp = np.array(temp[:, 1:8])
        return np.array(temp).astype('float')

    def __init__(self, q, qdot, qddot_d):
        
        self.q_data  = []

        for i, data in enumerate([q, qdot, qddot_d]):

            q_data_r = self.__read_data__(csv.reader(open(data, "r"), delimiter=","))

            self.q_data.append(q_data_r)
        
        self.q_data = np.asarray(self.q_data)

    def row(self, row):
        return self.q_data[0,row,:].reshape((-1,1)), self.q_data[1,row,:].reshape((-1,1)), self.q_data[2,row,:].reshape((-1,1))

    
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
        self.y = meas - self.H @ self.x
        
        self.K = self.P @ self.H.T @ np.linalg.inv( self.R + self.H @ self.P @ self.H.T )

        # correct the state
        self.x = self.x + self.K @ self.y

        self.P = (np.eye(self.K.shape[0], H.shape[1]) - self.K @ self.H) @ self.P

    def get_state(self):
        return self.x, self.P, self.K

class plotting:
    
    def __init__(self, length):
    
        self.data  = np.zeros( ( 7, length ) )
    
    def insert(self, data, row):
        self.data[:, row] = data
    
    def plot(self, idx):
        
        # fig, axs = plt.subplots(7)
        
        t = np.linspace(0, 1, self.data.shape[1], endpoint=False)

        plt.figure(idx)

        for i in range(7):
            plt.subplot(7, 1, i+1)
            plt.plot(t, self.data[i, :] )
            plt.xlabel("t")
            plt.ylabel("rad/s")

        np.savetxt("kalman_arb.csv", self.data[:,:], delimiter=",")
    


# init the data loader
loader = data_loader("data/osc/q_frq1.3.csv", "data/osc/qdot_frq1.3.csv", "data/osc/qddot_frq1.3.csv")

# define kalman filter parameters
qdot_pred  = np.zeros((7,1))

# define the kalman filter
kalman = kalman_filter(qdot_pred, F, B, H, P, Q, R)

# plotting purposes
plot_tool = plotting( loader.len())

# plotting cov
plot_tool_cov = plotting( loader.len() )

# plot tool acceleration
plot_tool_acc = plotting( loader.len() )

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
    plot_tool.insert(qdot_pred.reshape((-1)), i)

    # plot the diag P
    #plot_tool_cov.insert(np.diag(K).reshape((-1)), i)

    # plot the acceleration
    # plot_tool_acc.insert(q_ddot_d.reshape((-1)), i)

plot_tool.plot(1)
#plot_tool_cov.plot(2)
#plot_tool_acc.plot(3)

plt.show()

# https://quant.stackexchange.com/questions/8501/how-to-tune-kalman-filters-parameter