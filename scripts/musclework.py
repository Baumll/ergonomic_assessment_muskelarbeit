#!/usr/bin/env python3
from curses import window
from importlib.abc import Finder
from turtle import st
import numpy as np
from numpy import angle
from sqlalchemy import true
import rospy
from std_msgs.msg import Int8MultiArray, MultiArrayDimension, Float32MultiArray, Int8
import sdt.changepoint as cp
import changefinder
import time
import matplotlib.pyplot as plt

#CPD
from functools import partial
from bocd import *

one_minute = 1e+9*60
one_second = 1e+9 #in nanao second

LAMBDA = 100
ALPHA = 0.1
BETA = 1.0
KAPPA = 1.0
MU = 0.0

max_time = 10 #How long you have to hold still (in sec)
max_queue = 10 #Max elemts in queue
max_elements = 900 #How many elemts to save
ignore_wrist = true
changepoint_threshold = 0.2 #how likely a change point is detected



past = 5 #How long to look in to past
start_time = time.time_ns()
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
        
        #FFT
        self.fft = []

        #Test
        self.callback_calls = 0
        #Adds data:
        if ignore_wrist:
            tmp_max = 7
        else:
            tmp_max = 9

        for i in range(tmp_max):
            cpd = BOCD(partial(constant_hazard, LAMBDA),
                StudentT(ALPHA, BETA, KAPPA, MU))
            self.bs.append(cpd)
            self.angles.append([])
            self.queue.append([])
            self.change_points.append((time.time_ns()-start_time))
            self.probabilities.append(0)
            self.fft.append(-1.0)
            
        #Pub sub
        rospy.Subscriber("/ergonomics/joint_angles", Float32MultiArray, self.callback)
        rospy.Subscriber('/ergonomics/rula', Float32MultiArray, self.score_callback)

        self.pub = rospy.Publisher('/musclework/musclework', Int8MultiArray, queue_size=10)
        self.pub_probabilities = rospy.Publisher('/musclework/probabilities', Float32MultiArray, queue_size=10)
        self.pub_fft = rospy.Publisher('/musclework/fft', Float32MultiArray, queue_size=10)
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
        data.data = list(data.data)
        tmp_time = (time.time_ns()-start_time)
        self.timestamps.append(tmp_time)
        self.timestamps = self.timestamps[-max_elements:]
        #if len(self.timestamps) > 2:
        #    timedelta = (len(self.timestamps))/(self.timestamps[-1]-self.timestamps[0])*one_second
        #    print(str(timedelta) + " aufrufe pro sec")

        for i in range(len(data.data)):
            data.data[i] = data.data[i]-50.0 #
            self.queue[i].append(data.data[i])
            
        self.queue[i] = self.queue[i][-max_queue:]
            

    #Hier werden die daten verarbeitet
    def process(self):
        pro_start_time = time.time_ns()
        #Data handle here:
        if len(self.queue[0]) > 0:
            for i in range(len(self.queue)):
                #Add time stamps
                for data_point in self.queue[i]:
                    self.angles[i].append(data_point)
                    self.angles[i] = self.angles[i][-max_elements:]
                    

                #löscht alte Elemete wieder
                

                #Bayes complete
                delay = len(self.queue[i])
                tmp_changepoints = []
                for point in self.queue[i]:
                    self.bs[i].update(point)
                if self.bs[i].growth_probs[delay] >= changepoint_threshold:
                    tmp_changepoints.append(self.bs[i].t - delay + 1)
                self.probabilities[i] = self.bs[i].growth_probs[delay]*100
                if tmp_changepoints != []:
                    self.change_points[i] = time.time_ns()
                #print(tmp_changepoints)
                #Aufräumen
                self.bs[i].prune(min(max_elements,self.bs[i].t))

                self.queue[i] = []
                if len(self.angles[i]) >= 100:
                    self.fft[i] = self.fastFourierTransform(self.angles[i])
                else:
                    self.fft[i] = -1.0
            
            

            #Plus punkt vergeben
            #[Arme, Körper]
            self.last_move_body_part = [0,0] #Wir gehen davon aus der Körper ist dynamisch
            tmp_time = (time.time_ns()-start_time)
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
            self.pub_fft.publish(createMultiArray(self.fft))
            self.pub_probabilities.publish(createMultiArray(self.probabilities))


            #print("Time Needed: " + str((time.time_ns()- pro_start_time)/one_second))

            
    def fastFourierTransform(self,data):
        dt = (len(self.timestamps))/(self.timestamps[-1]-self.timestamps[0])*one_second
        start_time = time.time_ns()
        n = len(data)
        fhat = np.fft.fft(data,n)
        PSD = fhat * np.conj(fhat) / n
        indices = PSD > fft_threashold
        fhat = indices * fhat
        time_nneded = time.time_ns()- start_time
    
    
        #find Peak
        abs_PSD = np.abs(PSD[1:len(PSD)//2])
        argmax = np.argmax(abs_PSD)

        #print("FFT needed " + str(time_nneded/one_second) + " sec. with data len " + str(n))
        print(str(dt * (argmax+1)/n) + " hz")
        return dt * (argmax+1)/n

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
    rate = rospy.Rate(2)
    
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