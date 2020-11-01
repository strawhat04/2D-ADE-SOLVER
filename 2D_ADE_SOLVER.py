# 2D convection diffusion solution by FVM, implementing power law scheme
#First order in time, second order in space below Peclet no. 10 and upward scheme onwards

#Notice this code is not suitable for too high Peclet number due to occurace of false diffusion, in that case to use fine mesh to avoid the same

import numpy as np
from matplotlib.collections import cm
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from math import exp, sin, cos, atan, sqrt

def tr(a):
	return np.transpose(a)

#advection flux coeff
def F(rho, u,dcv):
	return rho*u*dcv
#diffusion flux coeff
def D(gamma, dx,cv):
	if gamma==0 or dx==0:
		return 0
	else:
		return gamma*cv/dx
#Power law advection scheme
def funA(F,D):
	if D==0:
		return 0
	else:
		return max(0,(1-0.1*abs(F/(D)))**5 )
#control volume element length 
def dx(i):
	return CV_xGRID[i+1]-CV_xGRID[i]
#Grid element length 
def dxg(i):
	if i==xGRID_nos or i==0:
		return 0
	else:
		return xgrid[i]-xgrid[i-1]
#control volume element width 
def dy(j):
	return CV_yGRID[j+1]-CV_yGRID[j]
#Grid element width
def dyg(j):
	if j==yGRID_nos or j==0:
		return 0
	else:
		return ygrid[j]-ygrid[j-1]


#INPUT & INITIAL CONDITION
	 	
irho=50		#unifrom density of flow
iphi=5		#uniform initial scalar for scalar initiation

runtime=1
flow_u=5	#uniform flow in xdirection
flow_v=5	#uniform flow in ydireciton

G0=100		#uniform diffusion coeff

#MESH GEOMETRY
length=10
width=10

#
sc=0	#source terms
sp=0

dt=0.01	#time step


#MESH ELEMENTS
xELE_nos=100
yELE_nos=100

#IMPLEMENT MESH TYPE
def mesh(xELE_nos,yELE_nos):
	#MESH GEOMETRY

	global xGRID_nos
	global yGRID_nos

	xGRID_nos=xELE_nos+1	# no of x grid points 
	yGRID_nos=yELE_nos+1	# no of y grid points

	global CV_xGRID_nos
	global CV_yGRID_nos

	CV_xELE_nos=xELE_nos+1		# no of x control volume
	CV_yELE_nos=yELE_nos+1		#no of y control volume

	CV_xGRID_nos=CV_xELE_nos+1		# no of x CV grid points
	CV_yGRID_nos=CV_yELE_nos+1		# no of y grid points

	#meshgrid
	global CV_xGRID
	global CV_yGRID

	xcordi=np.linspace(0.0,length, num=xGRID_nos, dtype=float)		#create x grid points
	ycordi=np.linspace(0.0,width, num=yGRID_nos, dtype=float)		#create y grid points	

	#INSERT
	 
	CV_xGRID=np.insert((xcordi[1:]+xcordi[:-1])/2, [0, xGRID_nos-1], [xcordi[0], xcordi[xGRID_nos-1]])		#create control volume grid mid points of grid points including boundary poins 	
	CV_yGRID=np.insert((ycordi[1:]+ycordi[:-1])/2, [0, yGRID_nos-1], [xcordi[0], ycordi[yGRID_nos-1]]) 
	
	return xcordi, ycordi

#IMPLEMENT VELOCITY MODEL HERE
def velocity_model(xELE_nos,yELE_nos):

	u=flow_u*np.ones((xELE_nos+1,yELE_nos+1))
	v=flow_v*np.ones((xELE_nos+1,yELE_nos+1))

	return u,v

#IMPLEMENT DIFFUSION MODEL IN THIS FUNC
def diffusion(CV_xGRID_nos, CV_yGRID_nos):
	gamma=G0*np.ones((CV_xGRID_nos,CV_yGRID_nos))		#create real diffusion coeff grid

	#subtract false diffusion coefficient due to oblique flow
	for j in range(1, yGRID_nos):
		for i in range(1, xGRID_nos):
			if u[i,j]!=0 and v[i,j]!=0:
				theta=atan(v[i,j]/u[i,j])
				gamma[i,j]=gamma[i,j]-rho[i,j]*sqrt(u[i,j]**2+v[i,j]**2)*dxg(i)*dyg(j)*sin(2*theta)/(4*dyg(j)*sin(theta)**3+4*dxg(i)*cos(theta)**3) 
	return gamma

#CREATE GRID POINTS
xgrid, ygrid =mesh(xELE_nos,yELE_nos)

#MODELLING

#assign value of density, velocities and diffusion constant at each control volume face
rho=irho*np.ones((CV_xGRID_nos, CV_yGRID_nos))
u,v=velocity_model(CV_xGRID_nos,CV_yGRID_nos)
gamma=diffusion(CV_xGRID_nos, CV_yGRID_nos)


#INITIATE FUCNTIONS
iniFunc=lambda m: [np.zeros((xGRID_nos,yGRID_nos)) for _ in range(m)]
#ALGEBRIC EQUATION COEFF
ae,aw,au,ad,ap,ap0,b,d,c=iniFunc(9)

#TDMA ALGO TERMS
R=np.zeros(xGRID_nos+2)		#include 2 ghost nodes for including source terms outside domain 
Q=np.zeros(xGRID_nos+2)

phi=iphi*np.ones((int(runtime/dt)+1,xGRID_nos+2, yGRID_nos+2))	 #include 2 ghost nodes for including source terms outside domain 


#COMPUTE ALGEBRIC COEFF	
for i in range(1,xGRID_nos-1):
	for j in range(1,yGRID_nos-1):
		aw[i,j]=D(gamma[i,j], dxg(i),dy(j))*funA(F(rho[i,j],u[i,j], dy(j)),D(gamma[i,j], dxg(i), dy(j))) + max(F(rho[i,j],u[i,j], dy(j)),0)
		
		ae[i,j]=D(gamma[i+1,j], dxg(i+1),dy(j))*funA(F(rho[i+1,j],u[i+1,j],dy(j)),D(gamma[i+1,j], dxg(i+1), dy(j))) + max(-F(rho[i+1,j],u[i+1,j],dy(j)),0)	
		
		ad[i,j]=D(gamma[i,j], dyg(j),dx(i))*funA(F(rho[i,j],v[i,j], dx(i)),D(gamma[i,j], dyg(j), dx(i))) + max(F(rho[i,j],v[i,j], dx(i)),0)
	
		au[i,j]=D(gamma[i,j+1], dyg(j+1), dx(i))*funA(F(rho[i,j+1],v[i,j+1],dx(i)),D(gamma[i,j+1], dyg(j+1), dx(i))) + max(-F(rho[i,j+1],v[i,j+1],dx(i)),0)	
		
		c[i,j]= sp*dx(i)*dy(j)
		ap0[i,j]=rho[i,j]*dx(i)*dy(j)/dt	
		b[i,j]=sc*dx(i)*dy(j) # +ap0[i]*phi[t-1,i]




#INITIAL CONDITION
#phi[0,2:int(xELE_nos/5+1),:]=50


#BOUNDARY CONDITION
#Please note put Dirichlet here
phi[:,:,0]=20


#FORCED BOUNDARY COND for TDMA on x faces
Q[1]=1			
Q[xGRID_nos]=1

#Please put source type BC here
Q[0]=50		#source term at x=0-
R[0]=0

#Please put Neumann BC in term of flux coefficient after integrating ADE 
#Homogenous flux BC at y=0, y=w, x=0
for i in range(0, xGRID_nos):
	au[i,0]=max(-F(rho[i,0],v[i,0],dx(i)),0)

for j in range(0, yGRID_nos):
	ae[0,j]=max(-F(rho[0,j],u[0,j],dy(j)),0)

#COMPUTE FINAL FLUX COEFF OF PRESENT NODE(COMPUTING NODE)
ap=aw+ae+au+ad+ap0-c 

#CREATE RESIDUAL MATRIX TO STORE RESIDUAL VALUE AT EACH GRID POINTS EXCEPT BOUNDARIES
rdue=np.ones((xGRID_nos-2,yGRID_nos-2))


print("Please bear with us for few minutes :)")
#TDMA SOLVER in x direction and Gauss-Siedel in Y direction TO COMPUTE NUMERICAL FLUX 
for t in range(1, len(phi[:,0])):
	rdue.fill(1)
	itera=0
	phi[t,:,:]=phi[t-1,:,:]		#for good initial approximation
	while np.linalg.norm(rdue, 2)>1e-4 :  #np.max(np.abs(rdue))>1e-7 and
		for j in range(0, yGRID_nos):
			for i in range(0, xGRID_nos):
				d[i,j]=b[i,j]+ap0[i,j]*phi[t-1,i+1,j+1]+au[i,j]*phi[t,i+1,j+1+1]+ad[i,j]*phi[t,i+1,j-1+1]
				
				R[i+1]=ae[i,j]/(ap[i,j]-aw[i,j]*R[i])

				Q[i+1]=(d[i,j]+aw[i,j]*Q[i])/(ap[i,j]-aw[i,j]*R[i])

			for i in range(xGRID_nos-1,-1,-1):	
				phi[t,i+1,j+1]=R[i+1]*phi[t,i+2,j+1]+Q[i+1]

		rdue=phi[t,2:-2,2:-2]-np.divide((np.multiply(aw[1:-1,1:-1],phi[t,1:-3,2:-2])+np.multiply(ae[1:-1,1:-1],phi[t,3:-1,2:-2])+np.multiply(au[1:-1,1:-1],phi[t,2:-2,3:-1])+np.multiply(ad[1:-1,1:-1], phi[t,2:-2,1:-3])+b[1:-1,1:-1]+ np.multiply(ap0[1:-1,1:-1],phi[t-1,2:-2,2:-2])),ap[1:-1,1:-1])
		
		itera=itera+1


print("Scalar at t=0:")
print(tr(phi[0,1:-1,1:-1]))
print()
print("Scalar at t=runtime: ")
print(tr(phi[-1,1:-1,1:-1]))

#CREATE MESH GRID TO VISUAL DATA
x,y= np.meshgrid(xgrid,ygrid)
fig,ax=plt.subplots(figsize=(10,5))

#PLOT COLOR MESH
cmin=np.min(phi)		
cmax=np.max(phi)
cs=ax.pcolormesh(x,y,np.transpose(phi[0,1:-1,1:-1]), cmap=cm.RdPu, vmin=cmin,vmax=cmax)

cbar=fig.colorbar(cs)

ax.set_xlabel("Distnace (m)")
ax.set_ylabel("Distance (m)")
ax.set_title("2D Advection-Diffusion FVM Solver")

#Func TO ANIMATE THE PLOT
def update(t):
	for txt in ax.texts:
		txt.set_visible(False)
	ax.pcolormesh(x,y,np.transpose(phi[t,1:-1,1:-1]), cmap=cm.RdPu, vmin=cmin,vmax=cmax)
	ax.text(0,-length/10,'at time= %f sec'%(t*dt), size=10)

sim= animation.FuncAnimation(fig,update, frames=range(0,len(phi[:,0,0])), interval=10, repeat=False)

plt.show()
