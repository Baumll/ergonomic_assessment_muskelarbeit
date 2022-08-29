#!/usr/bin/env python3
from importlib.abc import Finder
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

one_minute = 1e+9*60

max_time = 1 #How long you have to hold still (in min)
max_queue = 10 #Max elemts in queue
max_elements = 251 #How many elemts to save
ignore_wrist = true
changepoint_threshold = 0.5 #how likely a change point is detected

#self.past = 5 #How long to look in to past

class Musclework():
    def __init__(self):
        #Paramter
        self.rula_score = []

        
        #self.bs = [] #Array of the CPD
        self.cf = [] #change Finder
        self.probabilities = []
        self.timestamps = []
        self.last_move_angle = []
        self.last_move_body_part = [0,0] #when was the last change point

        self.angles = []
        self.queue = []
        self.changepoints = []
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
            #self.bs = [cp.BayesOnline()]*tmp_max
            self.angles.append([])
            self.queue.append([])
            self.changepoints.append([])
            #self.last_move_angle = [-1]*tmp_max
            self.probabilities.append([])
            
        #Pub sub
        rospy.Subscriber("/ergonomics/joint_angles", Float32MultiArray, self.callback)
        rospy.Subscriber('/ergonomics/rula', Float32MultiArray, self.score_callback)

        self.pub = rospy.Publisher('/musclework/musclework', Int8MultiArray, queue_size=10)
        self.pub_change_points = rospy.Publisher('/musclework/change_points', Float32MultiArray, queue_size=10)

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
        tmp_time = time.time_ns()
        self.timestamps.append(tmp_time)
        self.timestamps = self.timestamps[-max_elements:]

        for i in range(len(data.data)):
            self.queue[i].append(data.data[0])
            
        self.queue[i] = self.queue[i][-max_queue:]
            


    #Hier werden die daten verarbeitet
    def process(self):
        #Data handle here:
        if len(self.queue[0]) > 0:
            for i in range(len(self.queue)):
                for j in self.queue[i]:
                    self.angles[i].append(j)
                    #self.bs[i].update(j)
                    score = self.cf[i].update(j)
                    self.probabilities[i].append(score)
                    if score > changepoint_threshold:
                        self.changepoints[i].append(self.timestamps[-1])
                
                #löscht alte Elemete wieder
                self.angles[i] = self.angles[i][-max_elements:]
                self.queue[i] = []

                #Plus punkt vergeben
                if len(self.changepoints[i]) > 0:
                    if time.time_ns() - self.changepoints[i][-1] < 1e+9 * max_time:
                        #upper_arm_left
                        if i == 0:
                            if self.changepoints[i][-1] < 1e+9 * max_time:
                                self.last_move_body_part[0] = 1
                        #upper_arm_right
                        elif i == 1:
                            if self.changepoints[i][-1] < 1e+9 * max_time:
                                self.last_move_body_part[0] = 1
                        #lower_arm left and right
                        elif i >= 2 and i < 4:
                            #Wenn der minimale score ist wird statisch ignoriert
                            if self.rula_score[i] > 1:
                                if self.changepoints[i][-1] < 1e+9 * max_time:
                                    self.last_move_body_part[0] = 1
                        #neck
                        elif i == 4:
                            if self.changepoints[i][-1] < 1e+9 * max_time:
                                self.last_move_body_part[1] = 1
                        #trunk
                        elif i == 5:
                            if self.changepoints[i][-1] < 1e+9 * max_time:
                                self.last_move_body_part[1] = 1
                        #legs
                        elif i == 6:
                            if self.changepoints[i][-1] < 1e+9 * max_time:
                                self.last_move_body_part[1] = 1

                        #wrist_left
                        elif i == 7:
                            pass
                        #wrist_right
                        elif i == 8:
                            pass

            #publish extra points
            plusScore = [0,0]
            for i in range(len(self.last_move_body_part)):
                if self.last_move_body_part[i] + one_minute * max_time < self.timestamps[-1]: 
                    plusScore[i] = 1
                    #print("Noch "+ str(1e+9) + " sec.")
        
            self.pub.publish(createInt8MultiArray(plusScore))
            
            #publish the probabilitys
            tmp_probabilities  = []
            for i in self.probabilities:
                tmp_probabilities.append(i[-1])
            self.pub_change_points.publish(createMultiArray(tmp_probabilities))
            

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