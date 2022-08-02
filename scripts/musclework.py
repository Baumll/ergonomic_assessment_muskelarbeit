#!/usr/bin/env python3
from pickletools import uint8
from sqlite3 import Time
from turtle import color
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rospy
from std_msgs.msg import Int8MultiArray, MultiArrayDimension, Float32MultiArray, Int8
import sdt.changepoint as cp
import time
from scipy import signal
import os

class Musclework():
    def __init__(self):
        #Paramter
        self.maxTimeSec = 1
        self.lastElements = 100
        self.past = 5
        self.bs = []
        self.timeStamps = []
        self.score = []
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
        self.pub = rospy.Publisher('/musclework', Int8, queue_size=10)

        #Speichert die daten:
        """     df = pd.DataFrame()
        df["upper_arm_left"] = self.angles[0]
        df["upper_arm_left_cp"] = self.bs[0].probabilities[-1][1:]
        df["upper_arm_right"] = self.angles[1]
        df["upper_arm_right_cp"] = self.bs[1].probabilities[-1][1:]
        df["lower_arm_left"] = self.angles[2]
        df["lower_arm_left_cp"] = self.bs[2].probabilities[-1][1:]
        df["lower_arm_right"] = self.angles[3]
        df["lower_arm_right_cp"] = self.bs[3].probabilities[-1][1:]
        df["neck"] = self.angles[4]
        df["neck_cp"] = self.bs[4].probabilities[-1][1:]
        df["trunk"] = self.angles[5]
        df["trunk_cp"] = self.bs[5].probabilities[-1][1:]
        df["legs"] = self.angles[6]
        df["legs_cp"] = self.bs[6].probabilities[-1][1:]
        df["wrist_left"] = self.angles[7]
        df["wrist_left_cp"] = self.bs[7].probabilities[-1][1:]
        df["wrist_right"] = self.angles[8]
        df["wrist_right_cp"] = self.bs[8].probabilities[-1][1:]

        os.remove("/home/adrian/Documents/Uni/BachelorArbeit/ba_workspace/src/ba_package/csv_data/outCPD.csv")
        df.index.name = 'row'
        df.to_csv("/home/adrian/Documents/Uni/BachelorArbeit/ba_workspace/src/ba_package/csv_data/outCPD.csv")
        print("save") """
        
        

    def callback(self,data):
        #Data handle here:
        rospy.loginfo(data.data)
        if self.bs == []:        
            for i in range(len(data.data)):
                self.bs.append(cp.BayesOnline())
                self.angles.append([])
                self.changePoints.append([])

    
        plusScore = 0
        tmp_time = time.time_ns()
        self.timeStamps.append(tmp_time)
        for i in range(len(data.data)):
            self.angles[i].append(data.data[i])
            
            self.bs[i].update(data.data[i])
            self.bs[i].probabilities = self.bs[i].probabilities[-100:]
            #changepoints = self.bs[i].find_changepoints(self.angles[i][-self.lastElements:], 3, 0.4)
            if(len(self.angles[i]) > self.past ):
                prob = self.bs[i].get_probabilities(self.past) #Past
                prob[0] = 0
                lmax = signal.argrelmax(prob)[0]
                tmp_changepoints = lmax[prob[lmax] >= 0.3] #Threshhold

                #Checken ob es den CP schon gab
                for j in tmp_changepoints:
                    if not j in self.changePoints[i]:
                        self.changePoints[i].append(j)
                        plusScore = 1

                

                #Plus punkt vergeben
                if len(self.changePoints[i]) > 0:
                    if time.time_ns() - self.timeStamps[self.changePoints[i][-1]] < 1e+9*self.maxTimeSec:
                        plusScore = 1

        #publish extra points
        #print("Time Needed: " + str((time.time_ns() - tmp_time)/1e+9) + " sec.")
        self.pub.publish(plusScore)
        if(plusScore > 0):
            print("Score: " + str(plusScore))





def plotter():
    fig = plt.figure(figsize=(9, 3))
    axs = fig.add_subplot(111)
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

if __name__ == '__main__':
    try:
        musclework_start()
    except rospy.ROSInterruptException:
        pass
        
        