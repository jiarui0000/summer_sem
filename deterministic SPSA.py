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
    
    # objective function
       
    # c  = (theta_value[0] - 3.0)**4  + (theta_value[1] + 1.0)**2
    
    c  = 0.5*((theta_value[0]**4 - 16.0*theta_value[0]**2 + 5.0*theta_value[0]) + (theta_value[1]**4 - 16.0*theta_value[1]**2 + 5.0*theta_value[1])) 
    
    # true global min at ( - 2.9035 , - 2.9035 )
    # there is a local minimum at (2.7468, 2.7468)
    
    return c


# m = number of iterations
# ep = epsilon fixed step size
    
def stochasticapproximation(th0,m,ep, eta):
    
    theta = np.zeros((2,m))
    Delta = np.zeros(2)
    cf = np.zeros(m)
     
    theta[:,0] = th0
    for i in range(m-1):
        cf[i]= cost_function(theta[:,i])
        
        # sample Delta vector
        
        if rnd.uniform(0,1) > 0.5 :
           Delta[0]=1
        else:
           Delta[0]=-1  
        
        if rnd.uniform(0,1) > 0.5 :
           Delta[1]=1
        else:
           Delta[1]=-1  
          
        # adjust stepsize
           
        eta_step= eta/((i+1)**0.6) 
        
        # evaluate SPSA proxy for gradient
        
        cost_plus = cost_function(theta[:,i]+eta_step*Delta )
        cost_minus = cost_function(theta[:,i]-eta_step*Delta )
        
        # SA update
        
        theta[0,i+1] = theta[0,i] - (ep/(i+1))*(cost_plus - cost_minus)/(eta_step*Delta[0]) 
        theta[1,i+1] = theta[1,i] - (ep/(i+1))*(cost_plus - cost_minus)/(eta_step*Delta[1]) 
        
        # use truncation to stabalize algorithm
        
        if theta[0,i+1] > 0:
            theta[0,i+1]=min(theta[0,i+1],5)
        else:
            theta[0,i+1]=max(theta[0,i+1],-5)
        
        if theta[1,i+1] > 0:
            theta[1,i+1]=min(theta[0,i+1],5)
        else:
            theta[1,i+1]=max(theta[0,i+1],-5)
        
        
    # ouptu: print results
    
    plt.plot(np.arange(m),theta[0,:],theta[1,:])
    plt.legend(['theta values']) 
    plt.show()

    plt.plot(np.arange(m),cf)
    plt.legend(['cost function']) 
    plt.show()


    print("Final values  theta & cost", theta[:,m-1], cost_function(theta[:,m-1]))    
    
    return theta


def main():
    
    th0 = np.zeros(2)
    
    th0[0]= -2.5
    th0[1] = -2.5
    
    m  = 2000 # iterations
    ep = 0.5  
    eta = 0.2
    
    stochasticapproximation(th0,m,ep,eta)
    
if __name__ == '__main__':
    main()
    