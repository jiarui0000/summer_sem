# -*- coding: utf-8 -*-
"""
Created on 8-3-2021

J ( \theta) = (theta-3)^4 

Function has no global Lipschitz constant

Here is a simple example where \epsilon can be too large to create covergence.


"""

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

# the random stream also input

def cost_function(theta_value):
    # c  = (theta_value- 3.0)**4
    
    c  = (theta_value - 3.0)**4/(theta_value**2 + 1.0)
    return c

def gradient(theta_value,i):
    # hder = 4.0 * ((theta_value -3.0) )**3 # +np.random.normal(0.0, 0.015 , 1)

    # loca min at x=-2 and global min at x = 3.0 
    hder = 2.0 * ((theta_value -3.0) )**3 * (theta_value**2 + 3.0 * theta_value + 2.0) / ( theta_value**2 + 1.0)**2  #+np.random.normal(0.0, 0.015 , 1)
       
    return hder

# m = number of iterations
# ep = epsilon fixed step size
    
def stochasticapproximation(th0,m,ep, eta):
    theta = np.zeros(m)
    cf = np.zeros(m)
        
    theta[0] = th0
    for i in range(m-1):
        cf[i]= cost_function(theta[i])
        
        # gradient descent
        
        theta[i+1] = theta[i] - ep* gradient(theta[i],i) #/(i+1)
        
        # FD descent
        
        #etai = (   eta/(2*(i+1))  )**2.0
        #cost_plus = cost_function(theta[i]+etai )
        #cost_minus = cost_function(theta[i]-etai )
        #cost_diff = ( cost_plus - cost_minus)/ ( 2.0*etai )
        #theta[i+1] = theta[i] - ep*cost_diff/(i+1)   
        
    plt.plot(np.arange(m),theta)
    plt.legend(['theta']) 
    plt.show()

    plt.plot(np.arange(m),cf)
    plt.legend(['cost function']) 
    plt.show()


    print("Final values  theta & cost", theta[m-1], cost_function(theta[m-1]))    
    
    return theta


def main():
    th0= 2.0
    m  = 100 # iterations
    ep = 0.075  
    eta = 1.0
    
    stochasticapproximation(th0,m,ep,eta)
    
if __name__ == '__main__':
    main()
