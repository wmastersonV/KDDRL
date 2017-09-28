from dnn_utils_dqn import *
import pandas as pd
import chainer
import chainer.functions as F
import chainerrl
import numpy as np
from chainerrl import explorers
from chainerrl import replay_buffer
from chainerrl.links.mlp_bn import MLPBN
import sklearn as sk
from sklearn.model_selection import train_test_split

class QFunc(chainerrl.q_functions.SingleModelStateQFunctionWithDiscreteAction):

    """Fully-connected state-input Q-function with discrete actions.
    Args:
    n_dim_obs: number of dimensions of observation space
    n_dim_action: number of dimensions of action space
    n_hidden_channels: number of hidden channels
    n_hidden_layers: number of hidden layers
    """

    def __init__(self, ndim_obs, n_actions, n_hidden_channels, n_hidden_layers, nonlinearity=F.relu, last_wscale=1.0):

        super(QFunc,self).__init__(model=MLPBN(
        in_size=ndim_obs, out_size=n_actions,
        hidden_sizes=[n_hidden_channels] * n_hidden_layers))

def reset():
    try:
        del q_func
    except:
        pass
    try:
        del agent
    except:
        pass
    try:
        del opt
    except:
        pass

dat0 = pd.read_csv("/Users/dmaste/Desktop/repos/KDDRL/data/kdd1998tuples.csv")
dat0.columns = ["customer","period","r0","f0","m0","ir0","if0","gender","age","income","zip_region","zip_la","zip_lo",
                "a","rew","r1","f1","m1","ir1","if1","gender","age","income","zip_region","zip_la","zip_lo"]
dat0 = dat0.drop(dat0.columns[[0,1,7,8,9,10,11,12,21,22,23,24,25]], axis=1)


# select 1.6M training, 0.5M validation balanced by action
dat_train, dat_val = train_test_split(dat0, test_size= 500000, train_size = 1600000, random_state=42, shuffle = True,
                                                      stratify = dat0['a'])
pd.value_counts(dat_train['a'].values) / dat_train.shape[0]
pd.value_counts(dat_val['a'].values) / dat_val.shape[0]

dat_train = dat_train.astype(np.float32)
dat_val = dat_val.astype(np.float32)

a_num = 12
states_num  = 5
learning_rate = np.float32(.001)
discount = .9
batch_size = 256
target_update_frequency = 10000
num_batches_per_epoch = round(dat_train.shape[0] / batch_size)
clip_err = np.float32(5)
errors = [] # list to keep track of training loss
q_vals = [] # list to keep track of q-values


# create input dataset
experiences = [{'state': dat_train.as_matrix(['r0', 'f0', 'm0', 'if0', 'if0'])[i],
 'action':int(dat_train['a'].values[i]),
 'reward': np.float32(dat_train['rew'].values[i]),
 'next_state':dat_train.as_matrix(['r1', 'f1', 'm1', 'if1', 'if1'])[i],
 'next_action': np.empty(0),
 'is_state_terminal': False
  } for i in range(0, dat_train.shape[0])]

# populate off policy memory (use gym's simulator on live data)
experiences = experiences[1:len(experiences)]
rbuf = replay_buffer.ReplayBuffer(len(experiences))


# need to rerun and clear memory due to internal chainer bug!
reset()

opt = chainer.optimizers.RMSprop(lr=learning_rate, alpha=0.99)
# def __init__(self, ndim_obs, n_actions, n_hidden_channels, n_hidden_layers, nonlinearity=F.relu, last_wscale=1.0):
qfunc = QFunc(ndim_obs = states_num, n_actions = a_num, n_hidden_channels = (15,40), n_hidden_layers = 2)
opt.setup(qfunc)
opt.add_hook(chainer.optimizer.GradientClipping(clip_err))

# you will not be using this, but you need to pass this as an input to the agent
explorer = explorers.LinearDecayEpsilonGreedy(0.1, 1, 10**4, range(a_num))

agent = chainerrl.agents.DoubleDQN(qfunc, opt, replay_buffer = rbuf, gamma=discount,
                                    explorer=explorer,
                                    replay_start_size=1000,
                                    target_update_interval=target_update_frequency,
                                    update_interval=1,
                                    phi=lambda x: x.astype(np.float32, copy=False),
                                    minibatch_size=batch_size,
                                    target_update_method='hard',
                                    clip_delta=False,
                                    average_q_decay=0,
                                    average_loss_decay=0.9)

for i in experiences:
    agent.replay_buffer.append(
        state=i['state'],
        action=i['action'],
        reward=i['reward'],
        next_state=i['next_state'],
        next_action=i['next_action'],
        is_state_terminal=i['is_state_terminal'])

n_epochs = 2
print_freq = 1000
# train deep Q-N
for e in range(n_epochs):
    for i in range(1, int(num_batches_per_epoch)+1) :
        print(i)
        # clone network every x steps
        if i % agent.target_update_interval == 0:
            print("cloning network")
            agent.sync_target_network()
        # sampling batches of observations
        transitions = agent.replay_buffer.sample(batch_size)
        # training without acting
        agent.update(transitions, errors)
        if i % print_freq == 0:
            action_value = agent.model(agent.batch_states([k['state'] for k in transitions], agent.xp, agent.phi))
            q = np.mean(action_value.max.data)
            agent.average_q *= agent.average_q_decay
            agent.average_q += (1 - agent.average_q_decay) * q
            q_vals.append(q)
            print ("Epoch " + str(e) + ", Iteration " + str(i) + ", " + str(round(100.0*i/num_batches_per_epoch,4)) + "% complete")
            print(agent.get_statistics())
