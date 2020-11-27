import csv
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from scipy.io import savemat

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

# measurement noise, assume high measurement noise
R = np.load('settings/cov_mat.npy') * 1

P = R * 0.1

# process noise
Q = R * 1

# Q = R / 10

Q[0,0] = Q[0,0] / 1
Q[1,1] = Q[1,1] / 1
Q[2,2] = Q[2,2] / 1
Q[3,3] = Q[3,3] / 1
Q[4,4] = Q[4,4] / 1
Q[5,5] = Q[5,5] / 1
Q[6,6] = Q[6,6] / 1

# init the data loader
loader = data_loader("data/osc/q_frq1.3.csv", "data/osc/qdot_frq1.3.csv", "data/osc/qddot_frq1.3.csv")

# define kalman filter parameters
qdot_pred  = np.zeros((7,1))

# kalmanfilter
kalman_filt = KalmanFilter(7, 7, 7)

kalman_filt.F = F
kalman_filt.H = H
kalman_filt.Q = Q*0.01
kalman_filt.R = R
kalman_filt.x = np.array([0, 0, 0, 0, 0, 0, 0])
kalman_filt.B = B
kalman_filt.P = P

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
    kalman_filt.predict(q_ddot_d.reshape((-1)))
    kalman_filt.update(qdot_mes)

    # insert the data
    plot_tool.insert(kalman_filt.x.reshape((-1)), i)

    # plot the diag P
    plot_tool_cov.insert(kalman_filt.y.reshape((-1)), i)

    # plot the acceleration
    plot_tool_acc.insert(np.diag(kalman_filt.K).reshape((-1)), i)

plot_tool.plot(1)
plot_tool_cov.plot(2)
plot_tool_acc.plot(3)

plt.show()

# https://quant.stackexchange.com/questions/8501/how-to-tune-kalman-filters-parameter