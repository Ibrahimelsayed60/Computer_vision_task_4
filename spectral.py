import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import copy


def global_threshold(image,threshold):
    binary = image > threshold
    for i in range(0,binary.shape[0],1):
        for j in range(0,(binary.shape[1]),1):
            if binary[i][j] == True:
                binary[i][j] = 256
            else:
                binary[i][j]=0
    return binary


def optimal_local_threshold(image,block_size):
    pass