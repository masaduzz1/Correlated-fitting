#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
import pylab
import sys
import csv
import pandas as pd
import random

def binner(data,block_length):
    binned_data=[]

    bins=len(data)//block_length

    data=data[0:block_length*bins]
    data=data.reshape(bins,block_length)

    binned_data= np.mean(data, axis=1)

    return binned_data

def partial_average(blocked_data):
    return (np.sum(blocked_data)-blocked_data)/(len(blocked_data)-1)



def jack_error(partial_avg_data):
    return np.std(partial_avg_data)*np.sqrt(len(partial_avg_data)-1)

def unjack(vec):
    thesum = np.sum(vec)
    n = vec.shape[0]
    want = thesum - (n-1) * vec
    return want
