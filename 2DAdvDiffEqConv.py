#!/usr/local/bin/python
from math import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import os
import errno

## 2DAdvDiffEq.py
## Remi Carmigniani
## Solves the 2D Advection Diffusion Equation :
## pd_t u[x,y,t] + pd_x u[x,y,t] + alpha* pd_y u[x,y,t] - 1/Pe(pd_xx u[x,y,t]+pd_yy u[x,y,t])  = 0 (x,y) in [0 2pi]^2
## BC periodic
## IC u[x,y,0]=cos(x)+sin(y) 
## Convergence test

## Series of N to try
N = [25, 50, 100]


##Physical parameter
Pe = 4.0
L=2*pi
alpha = 0. #velocity ration between x an y direction chose alpha smaller than 1
## Discretization parameter 
Nx=25
Ny=25
dx=L/float(Nx-1)
dy=L/float(Ny-1)

## time parameters 
t=0
tend = 2*pi
dt =.25*min(dx,dy)**2/(1./Pe + max(dx,dy))
#to make sure tend is reached and stability ok
dt = tend/float(round(tend/dt)+1)

## scheme parameters
# Laplacian terms
lx = 1/Pe*dt/dx/dx
ly=1/Pe*dt/dy/dy
# Advection terms
ax = dt/(2.*dx)
ay = alpha*dt/(2.*dy)

#plot axis
zmin = -1.2 
zmax = 1.2

## Figure numbering 
numb = 0

#dt is such that dt/dx^2 < 0.5 
s = 'The resolution is ' + repr(dx) + 'in the x direction and ' + repr(dy) + 'in the y direction'   + ', and the time step is ' + repr(dt) 
print s

## Initial conditions
def iCond(x,y):
    return .5*cos(x) + .5*cos(y)

uval = iCond(0,0)
u_arr = [[0 for i in xrange(Ny)] for i in xrange(Nx)]


for i in range(0,Nx):
	for j in range(0,Ny):
    		uval = iCond(i*dx,j*dy)
    		u_arr[i][j] = uval

## Construction of the update matrix
A = [[0 for i in xrange(Ny*Nx)] for i in xrange(Nx*Ny)]

for i in range(0,Nx):
	for j in range(0,Ny):      
		A[i*Ny+j][i*Ny+j]=1-2.*(lx+ly)
		A[i*Ny+j][i*Ny+(j+1)%Ny] = ly+ay
		A[i*Ny+j][i*Ny+j-1] = ly-ay  
		A[i*Ny+j][((i+1)%Nx)*Ny+j] = lx-ax
		A[i*Ny+j][(i-1)*Ny+j] = lx+ax



## Create a one row vector from the matrix u_arr
def matrixToVec(u,nx,ny):
	u_vec = [0 for i in xrange(ny*nx)]
	for i in range(0,nx):
		for j in range(0,ny):
			u_vec[i*ny+j] = u[i][j]
		
    	return u_vec

## Reverse
def vecToMatrix(u,nx,ny):
	Umat=[[0 for i in xrange(ny)] for i in xrange(nx)]
	for i in range(0,nx):
		for j in range(0,ny):
			Umat[i][j]=u[i*ny+j]
    	return Umat

#convert u_arr to u_vec
u_vec = matrixToVec(u_arr,Nx,Ny)
##Time loop
while t<=tend :
	t=t+dt
	#update u_vec
	u_vec = np.dot(A,u_vec)
	step=step+1

#Compate to exact solution
def u_exact(x,y,t):
	return 0.5*exp(-1/Pe*t)*cos(x-t)+0.5*exp(-1/Pe*t)*cos(y-alpha*t)
#L2 error using Simpson rule 
def errorL2(u_num):
	error=0
	for i in range(0,Nx):
		for j in range(0,Ny):
			error =  
	


print 'Simulation Completed without error'

		
        	

       
	
	


