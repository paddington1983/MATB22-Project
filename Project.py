# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 20:15:38 2019

@author: Ramon
"""
import matplotlib.pyplot as plt
import numpy as np

from scipy import optimize

from sympy import Symbol, solvers
from mpl_toolkits.mplot3d import Axes3D

from decimal import Decimal


# TASK 1
print("TASK 1")


# the given matrices
A = np.array([[1, 1, 2], [1, 2, 1], [2, 1, 1], [2, 2, 1]])
y = np.array([1, -1, 1, -1])

AtA = np.dot(A.T, A) # np.array([[10, 9, 7], [9, 10, 7], [7, 7, 7]])
Aty = np.dot(A.T, y) # np.array([0, -2, 1])

# solve using numpy
x = np.linalg.solve(AtA, Aty)
print("AtA:", AtA, "Aty:", Aty, "x via numpy:", x, "35x (gives nicer numbers):", x*35)

# Extra: solve using scipy minimum (starting at for example [1, -1, 1])
F = lambda x: np.linalg.norm(np.dot(A, x) - y)
minimum = optimize.fmin(F, np.array([1, -1, 1]))
print("x via scipy optimize:", minimum)

# calculate residuals for various values of a
AtAinv = np.linalg.inv(AtA)
def residual(a):
    y = np.array([1, a, 1, a])
    Aty = np.dot(A.T, y)
    x = np.dot(AtAinv, Aty)
    Ax = np.dot(A, x)
    
    return np.linalg.norm(Ax - y)

a_values = [a for a in range(-3, 8)]
residuals = [residual(a) for a in a_values]
plt.plot(a_values, residuals)
plt.show()

# residual for a=2 should be 0 as then Ax = y, with x = (0, 1, 0), y = (1, 2, 1, 2)
print("residual for a=2 (effectively 0):", residual(2))


#Task 2
print()
print("Task 2. Some information will be written as comments next to the code.")
#The decimal import was needed in order to get the number of significant figures required for these tasks.
#We are given the system of recurrence equations:
# a_n+1 = a_n + 3b_n +2c_n , a_0 = 8,
# b_n+1 = -3a_n + 4b_n +3c_n , b_0 = 3
# c_n+1 = 2a_n + 3b_n + c_n , c_0 = 12
#To compute and analyze values for large n, it can be useful to rewrite such a system in the form:
# Z_n=(A^n)Z_0
#Where:
Z_0 = array([[Decimal(8)], #A vector consisting of initial conditions
           [Decimal(3)],
           [Decimal(12)]])
A=array([[Decimal(1),Decimal(3),Decimal(2)], #A matrix consisting of the coefficients from the 
         [Decimal(-3),Decimal(4),Decimal(3)], #recurrence relations
         [Decimal(2),Decimal(3),Decimal(1)]])
def zn(M):
    List1=[Z_0]
    for i in range(0,M+1):
        w=(dot(A, List1[i]))
        List1.append(w)
        if i >= M:
            return(List1[i])

#Subtask 1: Do the iterates zn converge as n->infinity?
print("Subtask 1: The iterates do not converge. Here are two zn's that will showcase how quickly they change:")
print('zn(100)=', zn(100))
# We know, by comparison to a geometric series, that the iterates could not have converged.
print('and zn(200)=', zn(200), "which obviously differ greatly (just look at the orders of magnitude).")

#Subtask 2
print('Subtask 2')
def Znhat(n):
    Z_0hat = Z_0/(np.linalg.norm(Z_0))
    itList=[Z_0hat]
    for i in range(0,n+1):
        w=(dot(A, itList[i]))
        q=w/np.linalg.norm(w)
        itList.append(q)
        if i >= n:
            return(itList[i])
# This sequence does converge (as comparison to a geometric series would suggest).
# The differences vary little when n increases.
print('This sequence does converge, and changes little when n increases,')
print('for example: Znhat(200)', Znhat(200), 'and')
print('Znhat(500)', Znhat(500), 'have little difference.')
def letsplotnormed(N):  #This just plots the components of the above vector (Zn/||Zn||) to
    Klistx=[] # give visual confirmation.
    Klisty=[]
    Klistz=[]
    for n in range(1,N):
        Klistx.append(Znhat(n)[0][0])
        Klisty.append(Znhat(n)[1][0])
        Klistz.append(Znhat(n)[2][0])
        plot(n,Klistx[n-1], 'bo') #plots a_n in blue
        plot(n,Klisty[n-1], 'go') #plots b_n in green
        plot(n,Klistz[n-1], 'ro') #plots c_n in red
show(letsplotnormed(100))
print("The above graph shows the values of Znhat's entries, which converge rather quickly.")
print("Interestingly, the first and third components converge to the same number.")
print("I do not see a clear limit vector, but the iterates stay pretty close to")
print('Znhat=(0.6882472016116852977216287343,0.2294157338705617659072095784,0.6882472016116852977216287343)')
print("The limit vector is a unit vector (as was ensured when it was normalized).")
#Subtask 3
print('Subtask 3: q_n=ZnT*AZn=ZnT*Zn+1=scalar(Zn,Zn+1).The iterates converge to 1 rather quickly.')
#The q_n's approach 1 (from below).
def ZnTAZ(M): ## q_n=ZnT*A*Zn, since A*Z_n=Z_n+1, this is q_n=scalar(Zn,Z_n+1)
    w=Znhat(M).reshape(1,3)
    next=dot(w,Znhat(M+1))
    last=next[0][0]
    return(last)
def discfuncplot(g,M): #Can plot ZnTAZ(M) for values of k<M. 
    values=[]
    for k in range(0,M):
        values.append(g(k))
        plot(k,values[k], 'o')
show(discfuncplot(ZnTAZ,200))
print('The above graph shows the values of the iterates.')
print('Subtask 4')
def Sub4iteratecount(eps): #Subtask 4. This counts the number of iterates required for a given epsilon.
    Zlim=array([[Decimal('0.6882472016116852977216287343')],
       [Decimal('0.2294157338705617659072095784')],
       [Decimal('0.6882472016116852977216287343')]])
    for i in range(1,999):
        q=np.linalg.norm(Znhat(i)-Zlim)
        if q < eps:
            return(i)
print("There isn't much to say for this part; it takes", Sub4iteratecount(10**(-8)))
print("iterations for ||Znhat-Znhatlimit|| to be less than 10^(-8).")
print('Subtask 5')
lineps=np.linspace(0.1,10**(-16)) #Epsilon range.
def logplotter(f): #Subtask 5. This plots either the above iterate counter, or the one below in a logplot.
    for eps in lineps:
        loglog(eps,f(eps), 'o')
show(logplotter(Sub4iteratecount))
print('Subtask 6')
def Sub6iteratecount(eps): #Subtask 6. This sequence (q_n) seems to converge faster. 
    qnLim=Decimal(1)
    for k in range(1,999):
        diff=abs(ZnTAZ(k)-qnLim)
        if diff < eps:
            return(k)
show(logplotter(Sub6iteratecount))
print('Subtask 7: The second sequence (q_n) seems to converge faster (both from looking at the graph')
print("and from looking at values of particular epsilons (it takes the sequence of q_n's")
print("51 iterations before abs(q_n-qlimit)<10**(-16) as opposed to 121)).")
def normalplot(f): #This is just a plotter (like the logplotter above) with regular axes. 
    for eps in lineps: #I found having regular axes makes it easier to see which sequence converges faster. 
        plot(eps,f(eps),'b*')
show(normalplot(Sub4iteratecount))
show(normalplot(Sub6iteratecount))
print("I found the above graphs (with regular axes) made it easier to see how much faster the")
print("sequence of q_n's converges.")


# TASK 3
print()
print("TASK 3")

# solve for z (x3 in the task) using sympy
x = Symbol('x') #x1
y = Symbol('y') #x2
z = Symbol('z') #x3

print("solution via SymPy:", solvers.solve(2*x**2 - y**2 + 2*z**2 - 10*x*y - 4*x*z + 10*y*z - 1, z))

# the function obtained above via SymPy
f1 = lambda x, y: x - 5*y/2 - sqrt(27*y**2 + 2)/2
f2 = lambda x, y: x - 5*y/2 + sqrt(27*y**2 + 2)/2

# calculate a sample of values for the plot within x, y in [-1, 1]
x, y = np.meshgrid(np.linspace(-1, 1), np.linspace(-1, 1))
z1 = f1(x, y)
z2 = f2(x, y)

# plot them as surfaces (slow) or as contours (fast)
plt.figure(figsize=(6, 8))
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
#ax.plot_surface(x, y, z1)
#ax.plot_surface(x, y, z2)
ax.contour3D(x, y, z1, 50)
ax.contour3D(x, y, z2, 50)

# Extra: show a point at origin
ax.scatter(0, 0, 0, c='black')

# Extra: show points closest to origin
z = (sqrt(3) - 1 + sqrt(2)) / 3
ax.scatter(0, 0, z, c='green')
ax.scatter(0, 0, -z, c='green')

plt.show()

# try using '%matplotlib auto' for interactive plots
# can switch back to noninteractive using '%matplotlib inline'
