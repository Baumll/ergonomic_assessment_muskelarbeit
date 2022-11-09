import numpy
import numpy.random
import numpy.linalg
from functools import partial
from bocd import *
import pandas as pd
import matplotlib.pyplot as plt

LAMBDA = 100
ALPHA = .1
BETA = 1.
KAPPA = 1.
MU = 0.
DELAY = 7
THRESHOLD = 0.3



def test_BOCD_prune(series):
    """Tests that changepoints are detected while pruning.
    """
    bocd = BOCD(partial(constant_hazard, LAMBDA),
                StudentT(ALPHA, BETA, KAPPA, MU))
    changepoints = []
    for x in series[:DELAY]:
        bocd.update(x)
    for x in series[DELAY:]:
        bocd.update(x)
        if bocd.growth_probs[DELAY] >= THRESHOLD:
            changepoints.append(bocd.t - DELAY + 1)
            bocd.prune(bocd.t - DELAY)

    return changepoints

def BOCD_prune(bocd, data):
    """Tests that changepoints are detected while pruning.
    """
    changepoints = []
    if bocd.t <= DELAY:
        for x in data:
            bocd.update(x)
    else:
        for x in data:
            bocd.update(x)
            if bocd.growth_probs[DELAY] >= THRESHOLD:
                changepoints.append(bocd.t - DELAY + 1)
                bocd.prune(bocd.t-DELAY)
    return changepoints

def show_changepoints(data, changepoints):    
    figure, axis = plt.subplots(3, 3)

    for i in range(len(axis)):
        for j in range(len(axis[i])):
            axis[i,j].plot(data[i*3+j])
            for x in changepoints[i*3+j]:
                axis[i,j].axvline(x,lw=1, color='red')
    plt.show()


if __name__ == '__main__':

    file = pd.read_csv('src/ergonomic_assessment_muskelarbeit/csv_data/rula_data_statischhalten.csv',sep = ' ')

    data = [
        file['neck_angle'].values.tolist(),
        file['trunk_angle'].values.tolist(),
        file['legs_angle'].values.tolist(),
        file['upper_arm_left_angle'].values.tolist(),
        file['upper_arm_right_angle'].values.tolist(),
        file['lower_arm_left_angle'].values.tolist(),
        file['lower_arm_right_angle'].values.tolist(),
        file['wrist_left_angle'].values.tolist(),
        file['wrist_right_angle'].values.tolist()
    ]

    for i in range(len(data)):
        k = [*data[i], *data[i]]
        data[i] = k

    changepoints = []
    for i in range(len(data)):
        changepoints.append([])
        bocd = BOCD(partial(constant_hazard, LAMBDA),
            StudentT(ALPHA, BETA, KAPPA, MU))
        for j in range(0, len(data[i]), 7):
         changepoints[i] += BOCD_prune(bocd,data[i][j:j+7])

    names = [
        'neck_angle',
        'trunk_angle',
        'legs_angle',
        'upper_arm_left_angle',
        'upper_arm_right_angle',
        'lower_arm_left_angle',
        'lower_arm_right_angle',
        'wrist_left_angle',
        'wrist_right_angle'
    ]

    for i in changepoints:
        print(i)
    show_changepoints(data,changepoints)

