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

##
##################################################################################################################
############################			Useful functions		      ############################
##################################################################################################################
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
def iCond(x,y):
    return .5*cos(x) + .5*cos(y)


#plot figure
def plotFig(u_arr,numb,t):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	X = np.arange(0, L+dx, dx)
	Y = np.arange(0, L+dy, dy)
	X, Y = np.meshgrid(X, Y)
	R = np.sqrt(X**2 + Y**2)
	Z = np.sin(R)
	v = np.linspace(-1.2, 1.2, 15, endpoint=True)
	surf = ax.plot_surface(X, Y, u_arr,v, rstride=1, cstride=1,cmap=cm.jet,
 	       linewidth=0, antialiased=False, vmin=zmin, vmax=zmax)
	ax.set_zlim(zmin, zmax)
	ax.set_xlim(0, L)
	ax.set_ylim(0, L)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	
	fig.colorbar(surf, ticks=[-1,-0.8,-0.6, -0.4,-0.2, 0, 0.2,0.4,0.6,0.8, 1])
	fig.suptitle('t = '+ repr(t)) 
	plt.savefig(directory +'/t'+'%0*d' % (3, numb)+'.png')

##################################################################################################################
############################			Useful functions		      ############################
##################################################################################################################

directory='result'
if not os.path.exists(directory):
    os.makedirs(directory)
##PNG Parameter
numbIm = 20

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
dt =0.01*dx*dx
## scheme parameters
# Laplacian terms
lx = 1.*dt/dx/dx
ly=1.*dt/dy/dy
# Adv terms
ax =.5*dt/dx
ay=.5*dt/dy*alpha

#plot axis
zmin = -1.2 
zmax = 1.2

## Figure numbering 
numb = 0

#dt is such that dt/dx^2 < 0.5 
s = 'The resolution is ' + repr(dx) + 'in the x direction and ' + repr(dy) + 'in the y direction'   + ', and the time step is ' + repr(dt) 
print s

## Initial conditions
uval = iCond(0,0)
u_arr = [[0 for i in xrange(Ny)] for i in xrange(Nx)]


for i in range(0,Nx):
	for j in range(0,Ny):
    		uval = iCond(i*dx,j*dy)
    		u_arr[i][j] = uval



## Construction of the update matrix
A = [[0 for i in xrange(Ny*Nx)] for i in xrange(Nx*Ny)]

#for i in range(0,Nx):
#	for j in range(0,Ny):      
#		A[i*Ny+j][i*Ny+j]=1-2.*(lx+ly)
#		A[i*Ny+j][i*Ny+(j+1)%Ny] = ly+ay
#		A[i*Ny+j][i*Ny+(j-1)%Ny] = ly-ay  
#		A[i*Ny+j][((i+1)%Nx)*Ny+j] = lx-ax
#		A[i*Ny+j][((i-1)%Nx)*Ny+j] = lx+ax

#print 'The matrix is generated'


step=0
stepSize = int(tend/dt/numbIm)
plotFig(u_arr,numb,t)
s = 'Figure ' + repr(numb) + ' saved'
print s
#convert u_arr to u_vec
#u_vec = matrixToVec(u_arr,Nx,Ny)
##Time loop
while t<=tend :
	t=t+dt
	#update u_vec
	#u_vec = np.dot(A,u_vec)
	u_old = u_arr
	for i in range(0,Nx):
			for j in range(0,Ny):
			    u_arr[i][j]=u_old[i][j]*(1.-2.*(lx + ly)) + u_old[(i-1)%Nx][j]*(lx + ax)+u_old[(i+1)%Nx][j]*(lx - ax)+u_old[i][(j-1)%Ny]*(ly + ay)+u_old[i][(j+1)%Ny]*(ly - ay)
					    

	step=step+1
	if step%stepSize == 0:
        	step = 0
        	numb = numb+1
        	#u_arr = vecToMatrix(u_vec,Nx,Ny)
        	plotFig(u_arr,numb,t)
        	s = 'Figure ' + repr(numb) + ' saved'
		print s



print 'Simulation Completed without error'

		
        	

       
	
	


