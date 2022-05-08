#这里是获取乘累加操作个数的地方
import math
import networkx as nx
import sys
import random
import os
import sys
import math
from math import *
import csv
import time
import copy


class DESIGN_PARA:
    def __init__(self, IMG_SIZE, IMG_CHANNEL):
        self.IMG_SIZE = IMG_SIZE  # 图片大小 比如28或32
        self.IMG_CHANNEL = IMG_CHANNEL  # 输入通道 比如1或3

    def total_mac(self, predic_actions):
        total_mac =[]
        for i in range(1, len(predic_actions), 2):
            # 如果是第一个卷积层的话，它的输入为输出图像的宽*输出图像的高*输出图像通道*卷积核宽*高*输入图像通道
            if i == 1:
                mac = self.IMG_SIZE * self.IMG_SIZE * predic_actions[i] * predic_actions[i - 1] * \
                            predic_actions[i - 1] * self.IMG_CHANNEL

            else:
                mac = self.IMG_SIZE * self.IMG_SIZE * predic_actions[i] * predic_actions[i - 1] * \
                            predic_actions[i - 1] * predic_actions[i - 2]

            total_mac.append(mac)
        return total_mac

    def get_design(self, predic_actions):  ## here, layers denote a convolution operation
        layers_size = []
        layer_size = [0] * 5
        # layer_size=[M,N,R,C,K]
        # print(predict_actions)
        for i in range(0, len(predic_actions) - 1, 2):
            # print(i)
            if i == 0:
                layer_size[4] = predic_actions[i]
                layer_size[0] = predic_actions[i + 1]
                layer_size[1] = self.IMG_CHANNEL
                layer_size[2] = self.IMG_SIZE  # RC= 1 + (N-K+Padding)/stride
                layer_size[3] = self.IMG_SIZE  # RC= 1 + (N-K+Padding)/stride
            else:
                layer_size[4] = predic_actions[i]
                layer_size[0] = predic_actions[i + 1]
                layer_size[1] = predic_actions[i - 1]
                layer_size[2] = self.IMG_SIZE  # RC= 1 + (N-K+Padding)/stride
                layer_size[3] = self.IMG_SIZE  # RC= 1 + (N-K+Padding)/stride
            # print(layer_size)
            layers_size.append(copy.deepcopy(layer_size))
            # layers_size.append(layer_size)
        return layers_size


    '''
    Cn denotes the convolution to layer n, layer 1 is the starter(input img). Thus, para N, Tn in C0 is not used.  
    '''

    def get_conv_names(self, predic_actions):
        layersname = ["c1"]  # c1 is regarding as the operation to input img, doing nothing
        for i in range(0, int(len(predic_actions) / 2)):
            temp_name = "c" + str(i + 2)
            layersname.append(temp_name)
        return layersname

    # print(get_conv_names(predic_actions))
