#!/usr/bin/env python3
import numpy as np
from sqlalchemy import true
import rospy
from std_msgs.msg import Int8MultiArray, MultiArrayDimension, Float32MultiArray
import time
import matplotlib.pyplot as plt
import math
from functools import partial
from bocd import *
import sys
from types import ModuleType, FunctionType
from gc import get_referents
import pandas as pd


one_minute = 1e+9*60
one_second = 1e+9 #in nanao second

#Bayes CPD
LAMBDA = 100
ALPHA = .1
BETA = 1.
KAPPA = 1.
MU = 0.
DELAY = 7
THRESHOLD = 0.3

max_time = 60 #How long you have to hold still (in sec)
max_queue = 10 #Max elemts in queue
max_elements = 900 #How many elemts to save
ignore_wrist = true

BLACKLIST = type, ModuleType, FunctionType

start_time = time.time_ns()
fft_threashold = 0.2 #Ab wann frequenzen akzeptiert werden

class Musclework():
    def __init__(self):
        #Paramter
        self.rula_score = []
        
        self.bs = [] #Array of the CPD
        self.timestamps = []
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
            self.change_points.append(-1.)
            self.fft.append(-1.0)
            
        #Pub sub
        rospy.Subscriber("/ergonomics/joint_angles", Float32MultiArray, self.callback)
        rospy.Subscriber('/ergonomics/rula', Float32MultiArray, self.score_callback)

        self.pub = rospy.Publisher('/musclework/musclework', Int8MultiArray, queue_size=10)

        #init expoerter:
        self.export = exporter(tmp_max)
        

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

        for i in range(len(data.data)):
            self.queue[i].append(data.data[i])
            self.queue[i] = self.queue[i][-max_queue:]
        

        self.timestamps.append(time.time_ns()-start_time)
        self.timestamps = self.timestamps[-max_elements:]
        self.export.add_angels(self.queue)
            

    #Hier werden die daten verarbeitet
    def process(self):
        #Data handle here:
        if len(self.queue[0]) > 0:
            for i in range(len(self.queue)):
                #Add data
                for data_point in self.queue[i]:
                    self.angles[i].append(data_point)
                self.angles[i] = self.angles[i][-max_elements:]
                

                #Bayes complete
                tmp_cp = self.BOCD_prune(self.bs[i], self.queue[i])
                if tmp_cp != []:
                    self.change_points[i] = self.timestamps[-1] #hier wird die zeit des letzten change points genommen
                
                #Aufräumen
                self.queue[i] = []
                
                #FFT
                if len(self.angles[i]) > max_elements/4:
                    self.fft[i] = self.fastFourierTransform(self.angles[i])
                else:
                    self.fft[i] = -1.0
            

            #Plus punkt vergeben
            #[Arme, Körper]
            self.last_move_body_part = [0,0] #Wir gehen davon aus der Körper ist dynamisch
            self.repetive_body_part = [0,0]
            tmp_time = (time.time_ns()-start_time)
            for i in range(4):
                #Wenn mehr als max_time vergangen ist
                if (tmp_time - self.change_points[i]) > max_time*one_second:
                    if self.rula_score[i] > 1:
                        self.last_move_body_part[0] = 1
                #FFT
                if self.fft[i] > fft_threashold: #Hier gerne auch mehr testen
                    self.repetive_body_part[0] = 1

            for i in range(4,7):
                #Neck, trunk, legs
                if (tmp_time - self.change_points[i]) > max_time*one_second:
                    if self.rula_score[i] > 1:
                        self.last_move_body_part[1] = 1
                #FFT
                if self.fft[i] > fft_threashold: #Hier gerne auch mehr testen
                    self.repetive_body_part[1] = 1

            #Zusammen rechnen
            self.end_score = [0,0]
            self.end_score[0] = min(1, self.last_move_body_part[0]+ self.repetive_body_part[0])
            self.end_score[1] = min(1, self.last_move_body_part[1]+ self.repetive_body_part[1])

            #Publish
            self.pub.publish(createInt8MultiArray(self.end_score))
            
    def fastFourierTransform(self,data):
        #Mitlere Frequenz?
        dt = (len(self.timestamps))/(self.timestamps[-1]-self.timestamps[0])*one_second
        n = len(data)
        fhat = np.fft.fft(data,n)
        PSD = fhat * np.conj(fhat) / n
        indices = PSD > fft_threashold
        fhat = indices * fhat
    
        #find Peak
        abs_PSD = np.abs(PSD[1:len(PSD)//2])
        argmax = np.argmax(abs_PSD)

        return dt * (argmax+1)/n

    
    def BOCD_prune(self,bocd, data):
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
        if bocd.t - bocd.t0 > max_elements:
            bocd.prune(bocd.t-DELAY)
        return changepoints


class exporter():
    def __init__(self, lenght):
        self.angles = []
        self.change_points = []
        self.fft = []

        for i in range(lenght):
            self.angles.append([])
            self.change_points.append([])
            self.fft.append([])
        
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
    
    def export_all(self):
        dict = {}
        for i in range(len(self.angles)):
            dict[self.angle_names[i]] = self.angles[i]
        
        for i in range(len(self.fft)):
            dict[self.angle_names[i]+"_fft"] = self.fft[i]
        
        for i in range(len(self.change_points)):
            dict[self.angle_names[i]+"_cp"] = self.change_points[i]

        export = pd.DataFrame.from_dict(dict)
        export.to_csv (r'muscelwork.csv', index = False, header=True)


    def add_angels(self,data):
        for i in range(len(data)):
            self.angles[i] += data[i]
        
        if len(self.angles[0]) > 2000:
            pass
            #show_changepoints(self.angles, self.change_points)

    def add_fft(self,data):
        for i in range(len(data)):
            self.fft[i] += data[i]

    def add_cpd(self,data):
        for i in range(len(data)):
            self.cpd[i] += data[i]

    def show_changepoints(self, data, changepoints):
        y_fig = math.ceil(len(data)/3)
        figure, axis = plt.subplots(3, y_fig)

        for i in range(len(axis)):
            for j in range(len(axis[i])):
                if i*3+j < len(data):
                    axis[i,j].plot(data[i*3+j])
                    for x in changepoints[i*3+j]:
                        axis[i,j].axvline(x,lw=1, color='red')
        plt.show()

def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size

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