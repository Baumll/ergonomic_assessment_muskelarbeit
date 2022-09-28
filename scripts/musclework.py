#!/usr/bin/env python3
from importlib.abc import Finder
from turtle import st
import numpy as np
from sqlalchemy import true
import rospy
from std_msgs.msg import Int8MultiArray, MultiArrayDimension, Float32MultiArray, Int8
import sdt.changepoint as cp
import changefinder
import time
from scipy import signal
from collections import deque
import sys
import recursivesize
from bocd import *

one_minute = 1e+9*60
one_second = 1e+9 #in nanao second

LAMBDA = 100
ALPHA = 0.1
BETA = 1.0
KAPPA = 1.0
MU = 0.0
DELAY = 15

max_time = 10 #How long you have to hold still (in sec)
max_queue = 10 #Max elemts in queue
max_elements = 50 #How many elemts to save
ignore_wrist = true
changepoint_threshold = 0.3 #how likely a change point is detected



past = 5 #How long to look in to past
start_time = time.time_ns()
dt = 0.1 #Ist die Frequenz mit der ich daten erhalte
fft_threashold = 100 #Ab wann frequenzen akzeptiert werden

class Musclework():
    def __init__(self):
        #Paramter
        self.rula_score = []

        
        self.bs = [] #Array of the CPD
        self.cf = [] #change Finder
        self.probabilities = [] #To publish
        self.timestamps = []
        self.last_move_angle = []
        self.last_move_body_part = [0,0] #when was the last change point

        self.angles = []
        self.queue = []
        self.change_points = [] #When was the last changepoint for the angle
        self.angle_names = [
            "upper_arm_left",
            "upper_arm_right",
            "lower_arm_left",
            "lower_arm_right",
            "neck",
            "trunk",
            "legs",
            "wrist_left",
            "wrist_right"]

        #Adds data:
        if ignore_wrist:
            tmp_max = 7
            #Testen Welche einstellungen die besten sind
            self.cf.append(changefinder.ChangeFinder(r=0.1, order=7, smooth=3))
            self.cf.append(changefinder.ChangeFinder(r=0.1, order=7, smooth=4))
            self.cf.append(changefinder.ChangeFinder(r=0.1, order=7, smooth=5))
            self.cf.append(changefinder.ChangeFinder(r=0.1, order=7, smooth=6))
            self.cf.append(changefinder.ChangeFinder(r=0.1, order=7, smooth=7))
            self.cf.append(changefinder.ChangeFinder(r=0.1, order=7, smooth=8))
            self.cf.append(changefinder.ChangeFinder(r=0.1, order=7, smooth=9))
        else:
            tmp_max = 9

        for i in range(tmp_max):
            cpd = cp.BayesOnline()
            #cpd.probabilities = deque(cpd.probabilities, maxlen= max_elements)
            self.bs.append(cpd)
            self.angles.append([])
            self.queue.append([])
            self.change_points.append((time.time_ns()-start_time)/one_second)
            #self.last_move_angle = [-1]*tmp_max
            self.probabilities.append(0)
            
        #Pub sub
        rospy.Subscriber("/ergonomics/joint_angles", Float32MultiArray, self.callback)
        rospy.Subscriber('/ergonomics/rula', Float32MultiArray, self.score_callback)

        self.pub = rospy.Publisher('/musclework/musclework', Int8MultiArray, queue_size=10)
        self.pub_probabilities = rospy.Publisher('/musclework/probabilities', Float32MultiArray, queue_size=10)
        #self.pub_change_points = rospy.Publisher('/musclework/change_points', Int8MultiArray, queue_size=10)

        

    def score_callback(self,data):
        if ignore_wrist:
            data.data = data.data[0:7]
        for i in range(len(data.data)):
            self.rula_score.append(data.data[i])
        self.rula_score = self.rula_score[-max_elements:]

    #Hier kommen die daten rein
    def callback(self,data):
        #Die letzten beiden Elemente werden rausgeworfen
        if ignore_wrist:
            data.data = data.data[0:7]
        
        #rospy.loginfo(data.data)

        #Neues wird hinzu gefügt und altes rausgeschmissen
        tmp_time = (time.time_ns()-start_time)/one_second
        self.timestamps.append(tmp_time)
        self.timestamps = self.timestamps[-max_elements:]

        for i in range(len(data.data)):
            self.queue[i].append(data.data[i])
            
        self.queue[i] = self.queue[i][-max_queue:]
            
    def bocd_cpd(self, bo, data):
        pass

    #Hier werden die daten verarbeitet
    def process(self):
        #Data handle here:
        if len(self.queue[0]) > 0:
            for i in range(len(self.queue)):
                #Add time stamps
                for data_point in self.queue[i]:
                    self.angles[i].append(data_point)
                    

                #löscht alte Elemete wieder
                self.queue[i] = []
                if len(self.angles[i]) >= max_elements:
                    fastFourierTransform(self.angles[i])

                #Bayes complete
                if len(self.angles[i]) > past:
                    self.angles[i] = self.angles[i][-max_elements:]
                    tmp_changepoints = self.bs[i].find_changepoints(self.angles[i], past, changepoint_threshold)

                    if len(tmp_changepoints) > 0:
                        self.change_points[i] = (time.time_ns()-start_time)/one_second#
                        print("Change Points: "+ self.angle_names[i]+ " : " + str(tmp_changepoints))


                    """ #Bayes CPD
                    self.bs[i].update(data_point)
                    prob = self.bs[i].get_probabilities(past) #Past
                    if len(prob) > past:
                        prob[0] = 0
                        lmax = signal.argrelmax(prob)[0]
                        tmp_changepoints = lmax[prob[lmax] >= changepoint_threshold] #Threshhold 
                    """
                    """ #Checken ob es den CP schon gab
                    for j in tmp_changepoints:
                        if not j in self.change_points:
                            self.change_points.append(j)
                            self.change_points[i] = time.time_ns()
                            self.change_points[i] = self.change_points[i][-max_elements:] """
                    #cleanup vom bayes 
                    #self.bs[i].probabilities[-1] = self.bs[i].probabilities[-1][-max_elements:]
                    self.probabilities[i] = self.bs[i].probabilities[-1][-1]*100


                    """ #Change Finder:
                    score = self.cf[i].update(data_point)
                    self.probabilities[i] = score
                    if score > changepoint_threshold:
                        self.changepoints[i] = time.time_ns() """




            #Plus punkt vergeben
            #[Arme, Körper]
            self.last_move_body_part = [0,0] #Wir gehen davon aus der Körper ist dynamisch
            tmp_time = (time.time_ns()-start_time)/one_second
            for i in range(4):
                #Wenn mehr als max_time vergangen ist
                if tmp_time - self.change_points[i] > max_time:
                    if self.rula_score[i] > 1:
                        self.last_move_body_part[0] = 1
            for i in range(4,7):
                #Neck, trunk, legs
                if tmp_time - self.change_points[i] > max_time:
                    if self.rula_score[i] > 1:
                        self.last_move_body_part[1] = 1

        
            #Publish
            self.pub.publish(createInt8MultiArray(self.last_move_body_part))
            #self.pub_change_points.publish(createInt8MultiArray(self.last_move_body_part))
            self.pub_probabilities.publish(createMultiArray(self.probabilities))
            
def fastFourierTransform(data):
    start_time = time.time_ns()
    n = len(data)
    fhat = np.fft.fft(data,n)
    PSD = fhat * np.conj(fhat) / n
    freq = (1/(dt*n)) * np.arange(n)
    L = np.arange(1,np.floor(n/2),dtype='int')
    indices = PSD > fft_threashold
    PSD_clean = PSD * indices
    fhat = indices * fhat
    time_nneded = time.time_ns()- start_time
    print("FFT needed " + str(time_nneded/one_second) + " sec. with data len " + str(n))
    return PSD_clean

def createMultiArray(data):
    msg = Float32MultiArray()
    dim = MultiArrayDimension()
    dim.label = "RULA Scores"
    dim.stride = 1
    dim.size = len(data)
    msg.layout.dim.append(dim)
    msg.data = data
    return msg  

def createInt8MultiArray(data):
    msg = Int8MultiArray()
    dim = MultiArrayDimension()
    dim.label = "RULA Scores"
    dim.stride = 1
    dim.size = len(data)
    msg.layout.dim.append(dim)
    msg.data = data
    return msg

def musclework_start():
    print("Setup Musclework")

    rospy.init_node('Musclework', anonymous=True)
    musclework = Musclework()
    rate = rospy.Rate(10)
    
    while not rospy.is_shutdown():
        try:
            musclework.process()
            rate.sleep()
        except KeyboardInterrupt:
            print("Shutting down")


if __name__ == '__main__':
    try:
        musclework_start()
    except rospy.ROSInterruptException:
        pass