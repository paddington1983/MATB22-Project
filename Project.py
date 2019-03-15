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


# TASK 3
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
