import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import tensorflow as tf
import numpy as np
import math

#=======================================================================================================
# Weight Summation and Weight Scaling FedAvg Functions
#=======================================================================================================

def sum_scaled_weights(scaled_weight_list):
    '''The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        layer_mean = layer_mean / len(layer_mean)
        avg_grad.append(layer_mean)

    return avg_grad

def emwa_model_weights_scaling(weight, scalar):
    '''function responsible for scaling for the model weights
    in case of Exponential Weighted Moving Average'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * np.array(weight[i]))

    return weight_final


def weight_sum(arr1,arr2):
    '''
    This is for the summation of two model weights
    '''
    result_return  = list()
    for i in range(len(arr1)):
        tmp_val = np.sum([arr1[i],arr2[i]], axis=0)
        tmp_val = tmp_val / len(tmp_val)
        result_return.append(tmp_val)
    return result_return
#=============================================================================================================
