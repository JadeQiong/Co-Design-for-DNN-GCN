# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from keras.datasets import cifar10
from keras.datasets import mnist
from keras.utils import to_categorical
from controller import Controller, StateSpace
from manager import NetworkManager
from model import model_fn
from PARA import DESIGN_PARA
from genetic_algorithm import GeneticAlgorithm
from network import Network
import math
import networkx as nx
import csv
import time
import sys

# sys.stdout = open("log.txt", "w")
# print("test sys.stdout")

data_file = open("output.csv", 'w', encoding='utf-8')
csv_writer = csv.writer(data_file)
csv_writer.writerow(["iter", "fitness", "time/ms", "area/mm^2", "power/pJ", "accuracy"])

# create a shared session between Keras and Tensorflow
policy_sess = tf.Session()
K.set_session(policy_sess)

NUM_LAYERS = 4  # number of layers of the state space
MAX_TRIALS = 10  # maximum number of models generated

MAX_EPOCHS = 5  # maximum number of epochs to train
CHILD_BATCHSIZE = 256  # batch size of the child models
EXPLORATION = 0.1  # high exploration for the first 1000 steps
REGULARIZATION = 1e-3  # regularization strength
CONTROLLER_CELLS = 32  # number of cells in RNN controller
EMBEDDING_DIM = 20  # dimension of the embeddings for each state
ACCURACY_BETA = 0.8  # beta value for the moving average of the accuracy
CLIP_REWARDS = 0.0  # clip rewards in the [-0.05, 0.05] range
RESTORE_CONTROLLER = False  # restore controller to continue training

IMG_CHANNEL = 1
IMG_SIZE = 28
OPT_TIMEPERFORMANCE = 12000000  # set the inference timing requirement


# construct a state space
state_space = StateSpace()

# add states, 在这里添加超参数的值
state_space.add_state(name='kernel', values=[5, 7, 14])
state_space.add_state(name='filters', values=[9, 18, 36])

# print the state space being searched
state_space.print_state_space()

# prepare the training data for the NetworkManager
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float16') / 255.
x_test = x_test.astype('float16') / 255.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

dataset = [x_train, y_train, x_test, y_test]  # pack the dataset for the NetworkManager

previous_acc = 0.0
total_reward = 0.0

with policy_sess.as_default():
    # create the Controller and build the internal policy network
    controller = Controller(policy_sess, NUM_LAYERS, state_space,
                            reg_param=REGULARIZATION,
                            exploration=EXPLORATION,
                            controller_cells=CONTROLLER_CELLS,
                            embedding_dim=EMBEDDING_DIM,
                            restore_controller=RESTORE_CONTROLLER)

# create the Network Manager
manager = NetworkManager(dataset, epochs=MAX_EPOCHS, child_batchsize=CHILD_BATCHSIZE, clip_rewards=CLIP_REWARDS,
                         acc_beta=ACCURACY_BETA)

# get an initial random state space if controller needs to predict an
# action from the initial state
state = state_space.get_random_state_space(NUM_LAYERS)
print("Initial Random State : ", state_space.parse_state_space_list(state))
print()

# clear the previous files
controller.remove_files()

old_acc = 0.7
ite_count = 0
acc_list = []
per_list = []
rew_list = []
old_action = []
old_action_cnt = 0

test_num = 5
for test_id in range(test_num):
    # train for number of trails
    for trial in range(MAX_TRIALS):
        G = nx.DiGraph()
        M_DESIGN_PARA = DESIGN_PARA(IMG_SIZE, IMG_CHANNEL)
        ite_count += 1
        print("===================This is TRIAL:", ite_count, "Let's go====================")
        with policy_sess.as_default():
            K.set_session(policy_sess)
            actions = controller.get_action(state)  # get an action for the previous state

            action_list = state_space.parse_state_space_list(actions)
            if (tuple(old_action) == tuple(action_list)) and old_action_cnt < 1:
                print("[Warning]: Find the same action as previous", old_action_cnt)
                manager.update_moving_acc()
                old_action_cnt += 1
            elif (tuple(old_action) == tuple(action_list)) and old_action_cnt == 1:
                print("[Warning]: Find the same action as previous reaching maximum, generate randomly")
                manager.update_moving_acc()
                # actions = controller.get_rand_action(state)
                old_action_cnt = 0
            elif (tuple(old_action) != tuple(action_list)):
                old_action_cnt = 0
            old_action = state_space.parse_state_space_list(actions)

        # print the action probabilities
        # state_space.print_actions(actions)
        print(" Predicted actions : ",
              state_space.parse_state_space_list(actions))  # generate CNN architecture parameters

        action_to_simulator = state_space.parse_state_space_list(actions)

        start = time.time()
        layersname = M_DESIGN_PARA.get_conv_names(action_to_simulator)
        print("DDDDDDDDDDDDDDDDDDDDDDDDEGBUG", layersname)
        end = time.time()
        print("This is time to explore design", end - start, "\n")

        # 这里就是我们计算的运行时间
        '''start = time.time()
        print('time_performancetime_performancetime_performancetime_performancetime_performance', time_performance)
        # We set performance according to the previous iteration
        end = time.time()
        print("PPPPPPPPPPPPPPPPPPPPP This is time performance", time_performance, "\n",
              "This is time to evaluate time formance", end - start, "\n", "\n", "\n")

        per_list.append(time_performance)'''
        network = Network(useList=True, macs=M_DESIGN_PARA.total_mac(action_to_simulator))
        ga = GeneticAlgorithm()
        ga.set_network(network)
        ga.run()
        best = ga.best_pop
        time_performance = best.time
        print("best: time = " + str(best.time) + " ns = " + str(best.time/pow(10, 6)) + " ms")
        print("area = " + str(best.area) + " um^2 = "+str(best.area/pow(10, 6)) + " mm^2")
        print("energy = " + str(best.energy/pow(10, 12)) + " * 10^12 pJ")
        csv_writer.writerow([trial, best.fit, best.time/pow(10, 6), best.area, best.energy, best.acc])
        print(" Write ok ")
        # build a model, train and get reward and accuracy from the network manager
        reward, previous_acc = manager.get_rewards(model_fn, state_space.parse_state_space_list(actions),
                                                   time_performance,
                                                   OPT_TIMEPERFORMANCE)  # CNN train and return the best accura
        print("Rewards : ", reward, "Accuracy : ", previous_acc)
        acc_list.append(previous_acc)
        rew_list.append(reward)
        print("===============+WWW+=================")
        print("++++++++++Acc History", acc_list)
        print("++++++++++Per History", per_list)
        print("++++++++++Rew History", rew_list)
        print("=====================================")
        # OPT_TIMEPERFORMANCE = min(time_performance*(1+(old_acc-previous_acc)),OPT_TIMEPERFORMANCE)
        old_acc = previous_acc

        with policy_sess.as_default():
            K.set_session(policy_sess)

            total_reward += reward
            print("Total reward : ", total_reward)

            # actions and states are equivalent, save the state and reward
            state = actions
            controller.store_rollout(state, reward)

            # train the controller on the saved state and the discounted rewards
            loss = controller.train_step()
            print("Trial %d: Controller loss : %0.6f" % (trial + 1, loss))

            # write the results of this trial into a file
            with open('train_history.csv', mode='a+') as f:
                data = [previous_acc, reward]
                data.extend(state_space.parse_state_space_list(state))
                writer = csv.writer(f)
                writer.writerow(data)
        print()

data_file.close()
print("Total Reward : ", total_reward)