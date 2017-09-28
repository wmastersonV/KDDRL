# GRADED FUNCTION: two_layer_model
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_utils import *
import torch as tc
import numpy as np
import pandas as pd
from torch.autograd import Variable

dat = pd.read_csv("/Users/dmaste/Desktop/repos/KDDRL/data/kdd1998tuples.csv")
dat.columns = ["customer","period","r0","f0","m0","ir0","if0","gender","age","income","zip_region","zip_la","zip_lo", "a","rew","r1","f1","m1","ir1","if1","gender","age","income","zip_region","zip_la","zip_lo"]
dat = dat.drop(dat.columns[[0,1,7,8,9,10,11,12,21,22,23,24,25]], axis=1)
dat2 = dat.as_matrix()
dat.columns

# RFM at time 0, RFM at time 1
X = dat2[:, [0,1,2,3,4,7,8,9,10,11] ]
# action, reward
Y = dat2[:, [5,6]]
Y.shape
# create neural network
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
# 2 hidden layers (40 and 15 neurons respectively with ReLU with ReLU activation function of form max(0, x)), 12 regression output neurons - one for each discrete action (including inaction - action 0),
# 5 or 6 variable inputs (5 RFM-I state variables + 1 continuous action variable),
N, D_in, H1, H2, D_out = 64, 5, 40, 15, len(np.unique(dat['a']))
layers_dims = (D_in, H1, H2, D_out)


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
n_x = 12288     # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)
parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)


parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)
