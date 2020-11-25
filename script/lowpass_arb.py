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
P = np.eye(N = 7, M = 7)

# measurement noise, assume high measurement noise
R = np.load('settings/cov_mat.npy')

# process noise
Q = np.eye(N = 7, M = 7) * noise ** 10

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

class lowpass_filter:

    def __init__(self, p):

        self.a = np.array([1-p, 1-p, 1-p, 1-p, 1-p, 1-p, 1-p]).reshape(7, 1)
        self.b = np.array([p, p, p, p, p, p, p]).reshape(7, 1)
        self.y_prev = np.array([0, 0, 0, 0, 0, 0, 0]).reshape(7, 1)
        self.y_curr = np.array([0, 0, 0, 0, 0, 0, 0]).reshape(7, 1)

    def filter(self, x):
        self.y_prev = self.y_curr
        self.y_curr = self.a * x + self.b * self.y_prev

    def get_state(self):
        return self.y_curr

class plotting:
    
    def __init__(self, name, len):
        
        self.name  = name
        self.data  = np.zeros((name.shape[0], name.shape[1], len))
    
    def insert(self, data, row, col = 1):
        self.data[col, :, row] = data
    
    def plot(self, col = 1):
        
        # fig, axs = plt.subplots(7)
        
        t = np.linspace(0, 1, self.data[col, 0, :].size, endpoint=False)

        for i in range(7):
            plt.subplot(7, 1, i+1)
            plt.plot(t, self.data[col, i, :])
            plt.xlabel("t")
            plt.ylabel("rad/s")

        plt.show()
        np.savetxt("lp_arb.csv", self.data[col,:,:], delimiter=",")
    


# init the data loader
loader = data_loader("arbitrary/q_des_1/1/q.csv", "arbitrary/q_des_1/1/qdot.csv", "arbitrary/q_des_1/1/qddot_d.csv")

# define kalman filter parameters
qdot_pred  = np.zeros((7,1))

# define the kalman filter
lowpass = lowpass_filter(0.75)

# plotting purposes
plot_tool = plotting(loader.name(), loader.len())

# for loop for the kalman filter to work
for i in range(0, loader.len()):

    # read the first set of measurement
    q_mes, qdot_mes, q_ddot_d = loader.row(i)

    # do the kalman prediction
    lowpass.filter(qdot_mes)

    # update the prediced steta and K etc
    qdot_pred = lowpass.get_state()

    # insert the data
    plot_tool.insert(qdot_pred.reshape((-1)), i, col = 1)

plot_tool.plot() 