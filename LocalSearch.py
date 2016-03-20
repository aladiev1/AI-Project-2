##########################################################################
## File:    Proj2.py                                                    ##
## Author:  Anna Aladiev                                                ## 
## Date:    03/19/2016                                                  ##
## Course:  CMSC 471 - Artificial Intelligence (Spring 2016)            ##
## Section: 02                                                          ##
## E-mail:  aladiev1@umbc.edu                                           ## 
##                                                                      ##
##   This file contains the python code for PROJECT 2 - OPTIMIZATION.   ##
## This program finds the global min of a function by using the hill    ##               
## climbing, hill climbing with random restarts, or simulated           ##
## annealing local search method.                                       ##
##                                                                      ## 
##########################################################################

import math

# FOR GENERATING FUNCTION GRAPHS
import numpy as np   # for coordinate arrays
import matplotlib.pyplot as plt   # for graph plot
from mpl_toolkits.mplot3d import Axes3D   # for 3D plot
from matplotlib import cm   # for colormap 
from matplotlib.ticker import LinearLocator, FormatStrFormatter   # for formatting axes
import random
   
    # BEST MOVE FUNCTION #
def bestMove(function_to_optimize, step_size, xmin, xmax, ymin, ymax, currentX, currentY):

    currentZ = function_to_optimize(currentX, currentY)
    
    bestZval = currentZ
    bestZmove = "current"

    if (currentX + step_size <= xmax):   # don't move if new point beyond xmax
        posXstep = function_to_optimize(currentX + step_size, currentY)
        if (posXstep < bestZval):
            bestZval = posXstep
            bestZmove = "posX"
    
    if (currentX - step_size >= xmin):   # don't move if new point beyond xmin
        negXstep = function_to_optimize(currentX - step_size, currentY) 
        if (negXstep < bestZval):
            bestZval = negXstep
            bestZmove = "negX"
    
    if (currentY + step_size <= ymax):   # don't move if new point beyond ymax
        posYstep = function_to_optimize(currentX, currentY + step_size)
        if (posYstep < bestZval):
            bestZval = posYstep
            bestZmove = "posY"
    
    if (currentY - step_size >= ymin):   # don't move if new point beyond ymin
        negYstep = function_to_optimize(currentX, currentY - step_size)
        if (negYstep < bestZval):
            bestZval = negYstep
            bestZmove = "negY"
       
       
    return (bestZmove)   # return best direction to move forward in


    # HILL-CLIMBING FUNCTION #
def hill_climb(function_to_optimize, step_size, xmin, xmax, ymin, ymax):
        
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    plt.title("HILL CLIMBING")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    currentX = random.uniform(xmin, xmax)   # random number between xmin and xmax chosen as starting point
    currentY = random.uniform(ymin, ymax)   # random number between ymin and ymax chosen as starting point
    
    bestZmove = bestMove(function_to_optimize, step_size, xmin, xmax, ymin, ymax, currentX, currentY)   # choses best next neighbor 

    while (bestZmove != "current"):

        ax.scatter(currentX, currentY, function_to_optimize(currentX, currentY))   # each move, plot new point on graph

        if (bestZmove == "posX"):   # move in the positive x direction if best move
            currentX = currentX + step_size
            bestZmove = bestMove(function_to_optimize, step_size, xmin, xmax, ymin, ymax, currentX, currentY)
            
        if (bestZmove == "negX"):   # move in the negative x direction if best move
            currentX = currentX - step_size
            bestZmove = bestMove(function_to_optimize, step_size, xmin, xmax, ymin, ymax, currentX, currentY)
            
        if (bestZmove == "posY"):   ## move in the positive y direction if best move
            currentY = currentY + step_size
            bestZmove = bestMove(function_to_optimize, step_size, xmin, xmax, ymin, ymax, currentX, currentY)
            
        if (bestZmove == "negY"):   # move in the negative y direction if best move
            currentY = currentY - step_size
            bestZmove = bestMove(function_to_optimize, step_size, xmin, xmax, ymin, ymax, currentX, currentY)
        
    currentZ = function_to_optimize(currentX, currentY)

    plt.show()
    return (currentX, currentY, currentZ)   # return coordinates of global min


    # HILL-CLIMBING WITH RANDOM RESTARTS FUNCTION #
def hill_climb_random_restart(function_to_optimize, step_size, num_restarts, xmin, xmax, ymin, ymax):    

    MAX_RESTARTS = 50
    
    currentX = random.uniform(xmin, xmax)   # random number between xmin and xmax chosen as starting point
    currentY = random.uniform(ymin, ymax)   # random number between ymin and ymax chosen as starting point

    bestZmove = bestMove(function_to_optimize, step_size, xmin, xmax, ymin, ymax, currentX, currentY)   # choses best next neighbor    
        
    while (bestZmove != "current"): 
        
        if (bestZmove == "posX"):   # move in the positive x direction if best move
            currentX = currentX + step_size
            bestZmove = bestMove(function_to_optimize, step_size, xmin, xmax, ymin, ymax, currentX, currentY)
            
        if (bestZmove == "negX"):   # move in the negative x direction if best move
            currentX = currentX - step_size
            bestZmove = bestMove(function_to_optimize, step_size, xmin, xmax, ymin, ymax, currentX, currentY)
            
        if (bestZmove == "posY"):   # move in the positive y direction if best move
            currentY = currentY + step_size
            bestZmove = bestMove(function_to_optimize, step_size, xmin, xmax, ymin, ymax, currentX, currentY)
            
        if (bestZmove == "negY"):   # move in the negative y direction if best move
            currentY = currentY - step_size
            bestZmove = bestMove(function_to_optimize, step_size, xmin, xmax, ymin, ymax, currentX, currentY)
        
    else:
        if (num_restarts != MAX_RESTARTS):   # if current position better than neighbor's, random restart
            num_restarts = num_restarts + 1
            hill_climb_random_restart(function_to_optimize, step_size, num_restarts, xmin, xmax, ymin, ymax)
            
           
    currentZ = function_to_optimize(currentX, currentY)

    return (currentX, currentY, currentZ)   # return coordinates of global min
    

    # ACCEPTANCE PROBABILITY FUNCTION #
def acceptance_probability(oldZ, newZ, temperature):
    
    if (newZ < oldZ):
        accept = 1.0   # new point is closer to goal

    else:
        accept = (math.exp((oldZ - newZ) / temperature))   # else use temperature equation 

    return (accept)
    
   
    # SIMULATED ANNEALING FUNCTION #
def simulated_annealing(function_to_optimize, step_size, max_temp, xmin, xmax, ymin, ymax):
    
    currentTemp = max_temp
   
    currentX = random.uniform(xmin, xmax)   # random number between xmin and xmax chosen as starting point
    currentY = random.uniform(ymin, ymax)   # random number between ymin and ymax chosen as starting point
    
    currentZ = function_to_optimize(currentX, currentY)
    bestZ = currentZ
    
    COOLING_RATE = 0.005
    
    while (currentTemp > 0.001):
        
        newX = random.uniform(xmin, xmax)   # random number between xmin and xmax chosen as new point        
        newY = random.uniform(ymin, ymax)   # random number between ymin and ymax chosen as new point 
        newZ = function_to_optimize(newX, newY)
        
        accept = acceptance_probability(currentZ, newZ, currentTemp)   # decide if new point better than current               
        randomNum = random.random()   # random number between 0 and 1 to compare to        
               
        # move new location if good probability
        if (accept > randomNum):
            currentX = newX
            currentY = newY
            currentZ = function_to_optimize(currentX, currentY)
            
        
        currentTemp = (currentTemp * (1 - COOLING_RATE))   # decrease temperature
        
        
    currentZ = function_to_optimize(currentX, currentY)

    return (currentX, currentY, currentZ)   # return coordinates of global min
    

    # TEST FUNCTION #
# EQUATIONS FOR CHECKING WORK VIA ONLINE GRAPHING DEVISE:
# z = [[sin((x^2)+(3(y^2)))]/[0.1+(r^2)]]+[(x^2)+(5(y^2))]*[[e^(1-(r^2))]/2]
# r = [sqrt((x^2)+(y^2))]
# z = [[sin((x^2)+(3(y^2)))]/[0.1+([sqrt((x^2)+(y^2))]^2)]]+[(x^2)+(5(y^2))]*[[e^(1-([sqrt((x^2)+(y^2))]^2))]/2]

#     sin(x^2 + 3y^2)                  e^(1 - r^2)                          
# z = --------------- + (x^2 + 5y^2) * -----------  ,  r = sqrt(x^2 + y^2)   
#        0.1 + r^2                          2                                                                                                              
def testFunction(testX, testY):
  
    x = float(testX)
    xSquared = math.pow(x, 2)   # x^2
    
    y = float(testY)
    ySquared = math.pow(y, 2)   # y^2
    
    # r = sqrt(x^2 + y^2) => sqrt(xSquared + ySquared)   
    r = math.hypot(x, y)   # euclidian norm
    rSquared = math.pow(r, 2)   # r^2
        

    #       sin(x^2 + 3y^2)   sin(xSquared + 3(ySquared)) 
    # eq1 = -------------- => ---------------------------
    #         0.1 + r^2             0.1 + rSquared
    eq1Numerator = (math.sin(xSquared + (3 * ySquared)))
    eq1Denominator = (0.1 + rSquared)
    eq1 = (eq1Numerator/eq1Denominator)


    # eq2 = (x^2 + 5y^2) => (xSquared + 5(ySquared))
    eq2 = (xSquared + (5 * ySquared))
    
    
    #       e^(1 - r^2)    e^(1 - rSquared)
    # eq3 = ----------- => ----------------
    #            2                2
    eq3Numerator = (math.exp(1 - rSquared))
    eq3Denominator = 2
    eq3 = (eq3Numerator/eq3Denominator)
    
    
    #     sin(x^2 + 3y^2)                  e^(1 - r^2)                          
    # z = --------------- + (x^2 + 5y^2) * -----------        
    #        0.1 + r^2                          2                                                                                                          ##
    z = (eq1 + eq2 * eq3)
    

    return(z)
    

    # FUNCTION GRAPH FUNCTION #
def functionGraph(function_to_optimize, step_size, xmin, xmax, ymin, ymax):
    
 
    FUNCT = plt.figure()
    FUNCTax = FUNCT.add_subplot(1,1,1, projection = '3d')

    plt.title("FUNCTION GRAPH")
    FUNCTax.set_xlabel('X')
    FUNCTax.set_ylabel('Y')
    FUNCTax.set_zlabel('Z')
    
    x = np.arange(xmin, xmax, step_size)   # x values generated from xmin to xmax by step_size
    y = np.arange(ymin, ymax, step_size)   # y values generated from ymin to ymax by step_size
    X, Y = np.meshgrid(x, y)   # coordinate matrices returned from coordinate vectors

    zs = np.array([function_to_optimize(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])   # array of Z values of coordinates
    Z = zs.reshape(X.shape)   # new array shape
    
    FUNCTax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = cm.coolwarm, linewidth = 0, antialiased = False)   # plots points with colormap  

    plt.show()
    
    return (None)
    

def main():    
    
    xMin = -2.5
    xMax = 2.5
    yMin = -2.5
    yMax = 2.5
    stepSize = 0.25    
    
    # creates 3d surface graph of function
    functionGraph(testFunction, stepSize, xMin, xMax, yMin, yMax)
    

    # finds x, y, and z coordinates of global min using hill climbing method
    HCx, HCy, HCz = hill_climb(testFunction, stepSize, xMin, xMax, yMin, yMax)

    ## x, y, and z values rounded to nearest thousandth
    #print ("HILL CLIMBING \n"
    #       'x = %(x).3f \n'
    #       'y = %(y).3f \n'
    #       'z = %(z).3f \n' 
    #       % {    
    #           'x': round(HCx, 3),
    #           'y': round(HCy, 3),
    #           'z': round(HCz, 3),
    #         }
    #      )    
          


    numRestarts = 0
    # finds x, y, and z coordinates of global min using hill climbing with random restarts method
    HCRx, HCRy, HCRz = hill_climb_random_restart(testFunction, stepSize, numRestarts, xMin, xMax, yMin, yMax)

    ## x, y, and z values rounded to nearest thousandth
    #print ("HILL CLIMBING WITH RANDOM RESTARTS \n"
    #       'x = %(x).3f \n'
    #       'y = %(y).3f \n'
    #       'z = %(z).3f \n' 
    #       % {    
    #           'x': round(HCRx, 3),
    #           'y': round(HCRy, 3),
    #           'z': round(HCRz, 3),
    #         }
    #      )   
   


    maxTemp = 1000
    # finds x, y, and z coordinates of global min using simulated annealing 
    SAx, SAy, SAz = simulated_annealing(testFunction, stepSize, maxTemp, xMin, xMax, yMin, yMax)

    # x, y, and z values rounded to nearest thousandth
    #print ("SIMULATED ANNEALING \n"
    #       'x = %(x).3f \n'
    #       'y = %(y).3f \n'
    #       'z = %(z).3f \n' 
    #       % {    
    #           'x': round(SAx, 3),
    #           'y': round(SAy, 3),
    #           'z': round(SAz, 3),
    #         }
    #      )   


main()