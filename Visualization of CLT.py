111010# -*- coding: utf-8 -*-











"""
    Visualize PDF of various distributions
    
    
    1. Run file (F5)
    2. Fill in asked value in console
    3. Press enter
    4. Repeat step 2 and 3 until the graph is plotted
"""





1000







import numpy as np
import time
import math
from matplotlib import pyplot as plt
from scipy.stats import norm
from matplotlib.patches import Rectangle
from scipy.stats import uniform
from scipy.stats import poisson
import sys
    
def exponential(n, labda, number_of_bins, fill,avg):
    # Drawing n values (i.i.d. with X ~ exp(λ))
    exp = np.zeros(n)
    for i in range(n):
        exp[i] = np.random.exponential(1/labda)
        for j in range(avg):
            exp[i] =  exp[i]+np.random.exponential(1/labda)
        exp[i]=exp[i]/avg
        10
        
    if fill == "Yes":
        bins = np.linspace(0, max(exp), number_of_bins) 
        for i in range(int(10-1)):
            plt.hist(exp[0:(i+1)*(int(n/10))], bins, denisty=True, color='skyblue')
            plt.title("Number of draws = %i" % int((i+1)*(int(n/10))))
            plt.show()
            time.sleep(1)

    # Plotting the histogram of the draws
    bins = np.linspace(0, max(exp), number_of_bins)
    n, bins, patches = plt.hist(exp, bins, density = True)
    def true_value(i):
        return (labda * pow(math.e, (-labda*(i*(max(exp)/(number_of_bins-1))+max(exp/(2*(number_of_bins-1)))))))
    for i in range(number_of_bins-1):
        if n[i] < true_value(i):
            patches[i].set_fc('salmon')
        if n[i] > true_value(i):
            patches[i].set_fc('springgreen')
        if n[i] <= 0.05*n[i]+true_value(i) and n[i] >= -0.05*n[i]+true_value(i):
            patches[i].set_fc('skyblue')

           
    # Plotting the PDF of the exponential distribution
    x = np.linspace(0, max(exp), 1000)
    y = (labda * pow(math.e, (-labda*x)))   # PDF: λe^(-λx)
    plt.plot(x, y, lw=3, color='black')
    
    if(avg>10):
        mu = np.mean(exp)    
        std = np.std(exp)
        x = np.linspace(mu-4*std, mu+4*std, 1000)
        y = norm.pdf(x, mu, std)   
        plt.plot(x, y, lw=3, color='blue', label = 'PDF') 
            
            
def normal(n, mu, sigma, number_of_bins, fill):
    #Drawing n values (i.i.d. with X ~ N(μ,σ^2))
    nor = np.zeros(n)
    for i in range(n):
        nor[i] = np.random.normal(mu, sigma)
        
        
    if fill == "Yes":
        bins = np.linspace(mu-4*sigma, mu+4*sigma, number_of_bins) 
        for i in range(int(10-1)):
            plt.hist(nor[0:(i+1)*(int(n/10))], bins, density=True, color='skyblue')
            plt.title("Number of draws = %i" % int((i+1)*(int(n/10))))
            plt.show()
            time.sleep(2)
    
    # Plotting the histogram of the draws
    bins = np.linspace(mu-4*sigma, mu+4*sigma, number_of_bins)
    n, bins, patches = plt.hist(nor, bins, normed=True)
    def true_value(i):
        return norm.pdf(((mu-4*sigma+i*2*4*sigma/(number_of_bins-1))+2*4*sigma/(2*(number_of_bins-1))), mu, sigma)
    for i in range(number_of_bins-1):
        if n[i] < true_value(i):            # Give bar red color if lower than PDF
            patches[i].set_fc('salmon')     
        if n[i] > true_value(i):            # Give bar green color if higher than PDF
            patches[i].set_fc('springgreen')   
        if n[i] <= 0.05*n[i] + true_value(i) and n[i] >= -0.05*n[i] + true_value(i):
            patches[i].set_fc('skyblue')    # Give bar blue color if approx. equal to PDF
    
    # Plotting the PDF of the normal distribution
    x = np.linspace(mu-4*sigma, mu+4*sigma, 1000)
    y = norm.pdf(x, mu, sigma)              # PDF: 1/(√(2π))*exp(-(x-μ)^2/2σ^2)
    plt.plot(x, y, lw=3, color='black', label = 'PDF')        
   
def uniformm(n, a, b, number_of_bins, fill,avg):
    # Drawing n values (i.i.d. with X ~ U(a,b))
    uni = np.zeros(n)
    for i in range(n):
        uni[i] = np.random.uniform(a,b)  
        for j in range(avg):
            uni[i] = uni[i] +np.random.uniform(a,b)
        uni[i]=uni[i]/avg
     
    if fill == "Yes":
        bins = np.linspace(a-1, b+1, number_of_bins)
        for i in range(int(10-1)):
            plt.hist(uni[0:(i+1)*(int(n/10))], bins, density=True, color='skyblue')
            plt.title("Number of draws = %i" % int((i+1)*(int(n/10))))
            plt.show()
            time.sleep(2)
    
    # Plotting the histogram of the draws
    bins = np.linspace(a, b, number_of_bins)
    n, bins, patches = plt.hist(uni, bins, density=True)
    for i in range(number_of_bins-1):
        if n[i] < 1/(b-a):
            patches[i].set_fc('salmon')
        if n[i] > 1/(b-a):
            patches[i].set_fc('springgreen')
        if n[i] <= 0.05*1/(b-a)+1/(b-a) and n[i] >= -0.05*1/(b-a)+1/(b-a):
            patches[i].set_fc('skyblue')                  
            
    # Plotting the PDF of the uniform distribution
    x = np.linspace(a, b, 1000)
    y = uniform(a, (abs(b)+abs(a)))
    plt.xlim(a-((abs(a)+abs(b))/2), b+((abs(a)+abs(b))/2))
    plt.plot(x, y.pdf(x), lw=3, color='black', label='PDF')
    
       
    if(avg>10):
        mu = np.mean(uni)    
        std = np.std(uni)
        x = np.linspace(mu-4*std, mu+4*std, 1000)
        y = norm.pdf(x, mu, std)   
        plt.plot(x, y, lw=3, color='blue', label = 'PDF') 
            
    
def poissonn(n, labda, number_of_bins, fill):
    # Drawing n values (i.i.d. with X ~ Poisson(λ))
    pois = np.zeros(n)
    for i in range(n):
        pois[i] = np.random.poisson(labda)
    
    if fill == "Yes":
        bins = np.linspace(labda-2*labda, labda+2*labda, number_of_bins) 
        for i in range(int(10-1)):
            plt.hist(pois[0:(i+1)*(int(n/10))], bins, density=True, color='skyblue')
            plt.title("Number of draws = %i" % int((i+1)*(int(n/10))))
            plt.show()
            time.sleep(2)
    
    # Plotting the histogram of the draws
    bins = np.linspace(labda-2*labda, labda+2*labda, num=number_of_bins)
    n, bins, patches = plt.hist(pois, bins, normed=True)
    def true_value(i):
        return pow(math.e, -labda)*(pow(labda, i)/math.factorial(i))
    for i in range(0, 3*labda-1):
        if n[i+labda-1] <= true_value(i):
            patches[i+labda-1].set_fc('salmon')
        if n[i+labda-1] >= true_value(i):
            patches[i+labda-1].set_fc('springgreen')
        if n[i+labda-1] <= (0.05*n[i+labda])+true_value(i) and n[i+labda-1] >= (-0.05*n[i+labda])+true_value(i):
            patches[i+labda-1].set_fc('skyblue') 
            
    # Plotting the PDF of the uniform distribution
    x = np.arange(poisson.ppf(0, labda), poisson.ppf(0.9999, labda))
    plt.plot(x, poisson.pmf(x, labda), lw=3, color='black', label='PMF')

    mu = np.mean(pois)    
    std = np.std(pois)
    x = np.linspace(mu-4*std, mu+4*std, 1000)
    y = norm.pdf(x, mu, std)   
    plt.plot(x, y, lw=3, color='blue', label = 'PDF') 
    
    
def plot(distribution):
    handles = ([Rectangle((0,0),1,1,color=d,ec="k") for d in ['black', 'salmon', 'springgreen', 'skyblue']])
    labels= ["PDF", "Hist < PDF","Hist > PDF", r"Hist $\approx$ PDF"]
    plt.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.05))        
    if distribution == 'Poisson':
        plt.title("PMF of %s Distribution" % distribution)
    else:
        plt.title("PDF of %s Distribution" % distribution)
    plt.show()
    
    
    

    
def main():    
    R  = '\033[31m'
    W  = '\033[0m'

    
    n = int(input("Number of samples = "))                 
    
    fill = 'No'
    
    distribution = input("Choose distribution: (Exponential (E), Normal (N) or Uniform (U))   ") 
    
    
    if distribution == 'Exponential' or distribution == 'E' or distribution == 'e':
        distribution = 'Exponential'
        labda = int(input("λ = "))
        avg = int(input("Batch size = "))
        
             
    if distribution == 'Normal' or distribution == 'N' or distribution == 'n':
        distribution = 'Normal'
        mu = int(input("μ = "))
        sigma2 = int(input("σ^2 = "))
        
    if distribution == 'Uniform' or distribution == 'U' or distribution == 'u':
        distribution = 'Uniform'
        a = int(input("Lower bound = "))
        b = int(input("Upper bound = "))
        if a > b:
            print(R+ "\n\n\n\n\n\n\nThe upper bound may not be lower than the lower bound!\n\n\n\n\n\n\n" +W)
            sys.exit()
        avg = int(input("Sample size = "))
        
    if distribution == 'Poisson' or distribution == 'P' or distribution == 'p':
        distribution = 'Poisson'
        labda = int(input("λ = "))
        number_of_bins = 4*labda

    if distribution != 'Poisson':
        number_of_bins = int(input("Number of bars in the histogram = (Recommended 20-50)    "))    
        
    
#    start_time = time.time()
    
   
    
    if distribution == 'Exponential':
        exponential(n, labda, number_of_bins, fill,avg)
    if distribution == 'Normal':
        normal(n, mu, sigma2, number_of_bins, fill)
    if distribution == 'Uniform':
        uniformm(n, a, b, number_of_bins, fill,avg)
    # if distribution == 'Poisson':
    #    poissonn(n, labda, number_of_bins, fill)
        
        
    plot(distribution)
    
if __name__ == '__main__':
    main()
    
    
    
    


    
    
    
    