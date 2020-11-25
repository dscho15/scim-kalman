import csv
import numpy as np
import pandas as pd
import matplotlib as mpl

# mpl.use("pgf")
# mpl.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

import matplotlib.pyplot as plt

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
loader = data_loader("data_static/qdot_1_1.csv")

# read the data
df = pd.DataFrame(loader.data().reshape((-1, 7)), columns=loader.name().reshape(-1))

# calculate quantiles
q1 = df.quantile(0.25)
q3 = df.quantile(0.75)
iqr = q3-q1
df = df[~((df < (q1 - 1.5 * iqr)) |(df > (q3 + 1.5 * iqr))).any(axis=1)]
pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(10, 10))
plt.show()
# plt.savefig('scattermatrix.png')

# boxplot
df.quantile()
df.boxplot(grid=False)
plt.show()

# determine covariance matrix
with open('settings/cov_mat.npy', 'wb') as f:
    data = loader.data().T.reshape(7, -1)
    cov = np.diag(np.cov(data).diagonal())
    np.save(f, cov)