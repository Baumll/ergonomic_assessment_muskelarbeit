#!/usr/bin/env python3
from pickletools import uint8
from sqlite3 import Time
from turtle import color
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import rospy
from std_msgs.msg import Int8MultiArray, MultiArrayDimension, Float32MultiArray, Int8
import sdt.changepoint as cp
import time
from scipy import signal
import os
from collections import deque 

one_minute = 1e+9*60

class Musclework():
    def __init__(self):
        #Paramter

        self.max_time = 10 #How long you have to hold still (in min)
        self.last_elements = 100 #How many elemts i store
        self.past = 5 #How long to look in to past
        self.bs = [] #Array of the CPD
        self.timestamps = [] 
        self.lastMove = [0,0] #when was the last change point
        
        self.angles = []
        self.changePoints = []
        self.angleNames = [
            "upper_arm_left",
            "upper_arm_right",
            "lower_arm_left",
            "lower_arm_right",
            "neck",
            "trunk",
            "legs",
            "wrist_left",
            "wrist_right"]

            
        #Pub sub
        rospy.Subscriber("/ergonomics/joint_angles", Float32MultiArray, self.callback)

        self.pub = rospy.Publisher('/musclework', Int8MultiArray, queue_size=10)    

    def callback(self,data):
        #Data handle here:
        rospy.loginfo(data.data)
        if self.bs == []:        
            for i in range(len(data.data)):
                self.bs.append(cp.BayesOnline())
                self.angles.append([])
                self.changePoints.append([])

    


        
        tmp_time = time.time_ns()
        self.timeStamps.append(tmp_time)
        deque(self.timeStamps,maxlen=self.lastElements)
        

        for i in range(len(data.data)):
            self.angles[i].append(data.data[i])
            deque(self.angles[i],maxlen=self.lastElements)
            
            self.bs[i].update(data.data[i])
            deque(self.bs[i].probabilities,maxlen=self.lastElements)

            if(len(self.angles[i]) > self.past ):
                prob = self.bs[i].get_probabilities(self.past) #Past
                prob[0] = 0
                lmax = signal.argrelmax(prob)[0]
                tmp_changepoints = lmax[prob[lmax] >= 0.3] #Threshhold

                #Checken ob es den CP schon gab
                for j in tmp_changepoints:
                    if not j in self.changePoints[i]:
                        self.changePoints[i].append(j)
                        #Die Arme
                        if i <= 3:
                            self.lastMove[0] = tmp_time
                        #Der Körper
                        elif i <= 5:
                            self.lastMove[1] = tmp_time                


                #Plus punkt vergeben
                if len(self.changePoints[i]) > 0:
                    if time.time_ns() - self.timeStamps[self.changePoints[i][-1]] < 1e+9*self.maxTimeSec:

                        #print("ChangePoint!")
                        #Die Arme
                        if i <= 3:
                            self.lastMove[0] = tmp_time
                        #Der Körper
                        elif i <= 5:
                            self.lastMove[1] = tmp_time   

        #publish extra points
        plusScore = [0,0]
        for i in range(len(self.lastMove)):
            
            if self.lastMove[i] + one_minute * self.max_time < tmp_time: 
                plusScore[i] = 1
                #print("Noch "+ str(1e+9) + " sec.")
        
        self.pub.publish(createInt8MultiArray(plusScore))
        #if(plusScore != [0,0]):
            #print("Noch: " + str((1e+9+self.lastMove[0]-tmp_time)/1e+9) + " / " + str((1e+9+self.lastMove[1]-tmp_time)/1e+9) + " sec.")






def plotter(data, changePoints):
    fig = plt.figure(figsize=(9, 3))
    for i in range(len(data)):
        axs[i] = fig.add_subplot(i % np.sqrt(len(data)), np.floor(i/np.sqrt(len(data))), i)
    axs.axhline(y=4, color='grey', linestyle='-',linewidth=.2)
    plt.show()
    quit()

def createInt8MultiArray(self,data):

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
    process = Musclework()
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

    while not rospy.is_shutdown():
        try:
            print(("test"))
        except KeyboardInterrupt:
            print("Shutting down")


if __name__ == '__main__':
    try:
        musclework_start()
    except rospy.ROSInterruptException:
        pass
        
        