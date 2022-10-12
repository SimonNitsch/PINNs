import numpy as np
import scipy.integrate as sint
import matplotlib.pyplot as plt

import os

print("----------------")
try:
    os.add_dll_directory("C:\\Programme\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin")
    print("GPU Support")
except:
    print("No GPU Support")
print("----------------")

import tensorflow as tf
from tensorflow import math


class tunnel:

    def __init__(self,E=1,a=5,b=200,c=0.5,h=1,m=1):
        self.E = E
        self.a = a
        self.b = b
        self.c = c

        self.h = h
        self.m = m
        self.k = np.sqrt(2 * m / h**2 * E)
        self.dict = {}

        

    def potential_numpy(self,x):

        return self.a * (np.tanh(self.b*(x+self.c)) - np.tanh(self.b*(x-self.c)))/2

    
    def potential(self,x):

        return self.a * (math.tanh(self.b*(x+self.c)) - math.tanh(self.b*(x-self.c)))/2


    
    def func(self,x,y): 

        f = 2 * self.m / self.h**2 * (self.potential_numpy(x) - self.E) * y[0]

        return y[1], f

    
    def solve_eq(self,x0,xmax,phi=0,ts=250):

        y0 = np.cos(self.k*phi) + 1j * np.sin(self.k*phi)
        y1 = 1j * self.k * y0

        y = sint.solve_ivp(self.func,(x0,xmax),np.array([y0,y1]),method="DOP853",t_eval=np.linspace(x0,xmax,ts),dense_output=True)
        self.dict.update({"y":y})
        max = np.max(abs(y.y[0,:]))
        self.dict.update({"max":max})


    
    def generate_wave(self,x0,xmax,A,phi,E,ts):


        i = 1

        for a,p,e in zip(A,phi,E):
            self.E = e
            self.k = np.sqrt(2 * self.m / self.h**2 * e)
            self.solve_eq(x0,xmax,phi=p,ts=ts)

            f = lambda x: self.dict["y"].sol(x)[0] / self.dict["max"] * a
            self.dict.update({i:f})
            self.dict.update({str(i):e})
            i+=1

        g = lambda x,t: (tf.constant(np.real(self.sum(x,t,i)),dtype=tf.float64), tf.constant(np.imag(self.sum(x,t,i)),dtype=tf.float64))

        return g


    def sum(self,x,t,i):
        
        x = np.array(x)
        t = np.array(t)

        y = self.dict[1](x) * np.exp(-1j/self.h * t * self.dict["1"])

        for j in range(2,i):
            
            y += self.dict[j](x) * np.exp(-1j/self.h * t * self.dict[str(j)])

        return y








