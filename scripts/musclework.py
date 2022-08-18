#!/usr/bin/env python3
from sqlalchemy import true
import rospy
from std_msgs.msg import Int8MultiArray, MultiArrayDimension, Float32MultiArray, Int8
import sdt.changepoint as cp
import time
from scipy import signal
from collections import deque 


ignore_wrist = true

class Musclework():
    def __init__(self):
        #Paramter

        self.max_time = 1 #How long you have to hold still (in min)
        self.max_elements = 200 #How many elemts to save
        self.max_queue = 10 #Max elemts in queue
        self.one_minute = 1e+9*60

        self.last_score = []

        self.past = 5 #How long to look in to past
        self.bs = [] #Array of the CPD
        self.timestamps = [] 
        self.lastMove = [0,0] #when was the last change point
        
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
        else:
            tmp_max = 9
        for i in range(tmp_max):
            self.bs.append(cp.BayesOnline())
            self.angles.append([])
            self.queue.append([])
            self.changepoints.append([])
            self.last_score.append([])
            
        #Pub sub
        rospy.Subscriber("/ergonomics/joint_angles", Float32MultiArray, self.callback)
        
        rospy.Subscriber('/ergonomics/rula', Float32MultiArray, self.score_callback)
        self.pub = rospy.Publisher('/musclework', Int8MultiArray, queue_size=10)    

    def score_callback(self,data):
        for i in range(len(data.data)):
            self.last_score[i] = data.data[i]

    #Hier kommen die daten rein
    def callback(self,data):
        #Die letzten beiden Elemente werden rausgeworfen
        if ignore_wrist:
            data.data = data.data[0:7]
        
        rospy.loginfo(data.data)

        #Neues wird hinzu gefügt und altes rausgeschmissen
        tmp_time = time.time_ns()
        self.timestamps.append(tmp_time)
        deque(self.timestamps,maxlen=self.max_elements)

        for i in range(len(data.data)):
            self.queue[i].append(data.data[i])
            deque(self.queue[i],maxlen=self.max_queue)
            


    #Hier werden die daten verarbeitet
    def process(self):
        #Data handle here:
        if len(self.queue[0]) > 0:
            for i in range(len(self.queue)):
                for j in self.queue[i]:
                    self.angles[i].append(j)
                    deque(self.angles[i],maxlen=self.max_elements)

                    self.bs[i].update(j)
                    deque(self.bs[i].probabilities,maxlen=self.max_elements)

                self.queue[i] = []

                if (len(self.angles[i]) > self.past ):
                    prob = self.bs[i].get_probabilities(self.past) #Past
                    prob[0] = 0
                    lmax = signal.argrelmax(prob)[0]
                    tmp_changepoints = lmax[prob[lmax] >= 0.3] #Threshhold

                    #Checken ob es den CP schon gab
                    for j in tmp_changepoints:
                        if not j in self.changepoints[i]:
                            self.changepoints[i].append(j)
                            #Die Arme
                            if i <= 3:
                                self.lastMove[0] = self.timestamps[-1]
                            #Der Körper
                            elif i <= 5:
                                self.lastMove[1] = self.timestamps[-1]               


                    #Plus punkt vergeben
                    if len(self.changepoints[i]) > 0:
                        if time.time_ns() - self.timestamps[self.changepoints[i][-1]] < 1e+9 * self.max_time:

                            print("ChangePoint!")
                            #Die Arme
                            if i <= 3:
                                self.lastMove[0] = self.timestamps[-1]
                            #Der Körper
                            elif i <= 5:
                                self.lastMove[1] = self.timestamps[-1]

            #publish extra points
            plusScore = [0,0]
            for i in range(len(self.lastMove)):
                print(self.lastMove[i])
                if self.lastMove[i] + self.one_minute * self.max_time < self.timestamps[-1]: 
                    plusScore[i] = 1
                    #print("Noch "+ str(1e+9) + " sec.")
        
            self.pub.publish(createInt8MultiArray(plusScore))



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