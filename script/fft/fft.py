import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

class data_loader:

    def __read_data__(self, data):
        temp = list(data)
        return temp[0], np.array(temp[1:]).astype("float")

    def __init__(self, qdot):
        
        self.q_name  = []
        self.q_data  = []

        for i, data in enumerate([qdot]):

            q_name_r, q_data_r = self.__read_data__(csv.reader(open(data, "r"), delimiter=","))

            self.q_name.append(q_name_r)
            self.q_data.append(q_data_r)
        
        self.q_name = np.asarray(self.q_name)
        self.q_data = np.asarray(self.q_data)

    def row(self, row):
        return self.q_data[0,row,:].reshape((-1,1)), self.q_data[1,row,:].reshape((-1,1)), self.q_data[2,row,:].reshape((-1,1))
    
    def name(self):
        return self.q_name
    
    def data(self):
        return self.q_data
    
    def len(self):
        return self.q_data.shape[1]

# init the data loader
loader = data_loader("../data_osc/qdot.csv")
y = loader.data().reshape(-1,7)
Y = scipy.fftpack.fft(y[:,4])
N  = y.shape[0]
fs = 1000
T  = 1.0 / fs
xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
plt.plot(xf, 2.0/N * np.abs(Y[:N//2]))
plt.show()
