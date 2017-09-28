from dnn_utils_dqn import *
import numpy as np
import pandas as pd

dat = pd.read_csv("/Users/dmaste/Desktop/repos/KDDRL/data/kdd1998tuples.csv")
dat.columns = ["customer","period","r0","f0","m0","ir0","if0","gender","age","income","zip_region","zip_la","zip_lo", "a","rew","r1","f1","m1","ir1","if1","gender","age","income","zip_region","zip_la","zip_lo"]
dat = dat.drop(dat.columns[[0,1,7,8,9,10,11,12,21,22,23,24,25]], axis=1)
# dat2 = dat.as_matrix()
dat.columns


# dat2 = dat.iloc[rnd,:]
dat2 = dat
# RFM at time 0, RFM at time 1
X = dat2.as_matrix()[:, [0,1,2,3,4,7,8,9,10,11] ].T
# Y: action, reward
L = 3
# Y = {}
# Y['a'] = (dat['a'].as_matrix().T).reshape(X.shape[1], 1)
# Y['r'] = (dat['rew'].as_matrix().T).reshape(X.shape[1], 1)
y_action = (dat2['a'].as_matrix().T).reshape(X.shape[1], 1)
y_reward = (dat2['rew'].as_matrix().T).reshape(X.shape[1], 1)
X_t0 = X[0:5, :]
X_t1 = X[5:10, :]
N, D_in, H1, H2, D_out = 64, 5, 40, 15, len(np.unique(dat['a']))
layers_dims = (D_in, H1, H2, D_out)

# to do:
# fix backprop, implement gradient checking

# run for X epochs
n_epochs = 1
m1_params = L_layer_model(X, y_action, y_reward, layers_dims, learning_rate = 0.001, decay_rate = .99,
                          num_iterations = n_epochs, print_cost=True, discount_rate = .9, batch_size = 200, plot =True)


    # Each model is trained with mini-batch gradient descent (batches of 200 transitions each) using RMSProp algorithm [Hinton et al., 2012] for 100 epochs,
# where epoch marks a point when each sample in the training data set has been sampled once (training length of 100 epochs means that each observation
# contributed to gradient update of the neural network 100 times). We begin with 0.001 learning rate and use 0.99 decay rate (learning rate is multiplied by
# decay rate every epoch, allowing us to further fine-tune the learning process as we observe more samples). We use 0.9 discount rate. Additionally, as a way
# to facilitate convergence (discussed earlier), every 10, 000 iterations we clone the network Q to obtain a target network Q*, which we then use for generating
#     maxa Q∗(s′, a′) components of updates to Q.

# gradient checking
parameters = initialize_parameters_deep(layers_dims)
AL, caches = L_model_forward(X_t0, parameters, y_action, compute_max_action=False)

# Compute cost.
discount_rate = .9
q_max, _ = L_model_forward(X_t1, parameters, y_action.T, compute_max_action=True)
Y = y_reward.T + discount_rate * q_max
cost = compute_cost_q(AL, Y)

# Backward propagation
grads = L_model_backward(AL, Y, caches, y_action)
difference = gradient_check_n(parameters, grads, X, Y)
