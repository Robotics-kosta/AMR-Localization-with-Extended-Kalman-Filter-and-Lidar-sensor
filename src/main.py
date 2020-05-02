#! /usr/bin/env python
import rospy
import tf
import numpy as np
import pandas as pd
import sys
import math as m
from scipy.linalg import block_diag 
from threading import Thread
import time

from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import JointState
from laser_line_extraction.msg import LineSegment,LineSegmentList
from tf.transformations import euler_from_quaternion, quaternion_from_euler


def transitionFunction(X_prev, U, b):
	d_sr,d_sl = U
	_,_,theta_prev = X_prev
	k1 = (d_sl+d_sr)/2
	k2 = (d_sr-d_sl)/2/b
	
	X_hat = X_prev + np.array([k1*m.cos(theta_prev+k2),
							   k1*m.sin(theta_prev+k2),
										k2*2
							  ]).reshape(3,1)
	_,_,theta = X_hat
	Fx = np.array([[1, 0, -k1*m.sin(theta+k2)],
				   [0, 1,  k1*m.cos(theta+k2)],
				   [0, 0,          1         ]
				  ])
	Fu11 = m.cos(theta+k2)/2+m.sin(theta+k2)*k1/2/b
	Fu12 = m.cos(theta+k2)/2-m.sin(theta+k2)*k1/2/b
	Fu21 = m.sin(theta+k2)/2-m.cos(theta+k2)*k1/2/b
	Fu22 = m.sin(theta+k2)/2+m.cos(theta+k2)*k1/2/b
	Fu31 = -1/b
	Fu32 = 1/b
	Fu = np.array([[Fu11, Fu12],
				   [Fu21, Fu22],
				   [Fu31, Fu32]
				  ])
	assert(X_hat.shape==(3,1))
	assert(Fx.shape==(3,3))
	assert(Fu.shape==(3,2)) 
	return X_hat, Fx, Fu
def getApriori(X_prev,P_prev, U, b):
	"""
	From the previous state, current control and previous covariance matrix
	get predictions of the next state and covariance matrix (apriori) 
	"""
	X_hat, Fx, Fu = transitionFunction(X_prev, U, b)

	d_sl,d_sr = U
	k = 0.01
	Q = np.array([[k*abs(d_sr), 0],
		      	  [0, k*abs(d_sl)]
				 ])
	P_hat = np.dot(np.dot(Fx,P_prev),Fx.T) + np.dot(np.dot(Fu,Q),Fu.T)

	assert(P_hat.shape==(3,3))
	return X_hat, P_hat
def measurementFunction(X_hat, M_i):
	"""
	From the predicted apriori position of the robot and the map
	get prediction of the measurement
	"""
	x_hat,y_hat,theta_hat = X_hat
	w_ro, w_alpha = M_i
	z_hat = np.array([w_alpha - theta_hat,
					  w_ro-(x_hat*m.cos(w_alpha)+y_hat*m.sin(w_alpha))
					 ])
	H_hat = np.array([[0,                     0,          -1],
					  [-m.cos(w_alpha), -m.sin(w_alpha),   0]
					 ])
	assert(H_hat.shape==(2,3))
	return z_hat, H_hat
def associateMeasurement(X_hat, P_hat, Z, R, M, g):
	"""
	Pairing what robot sees(measures) with predictions of measurements
	(what robot thinks he should see)
	"""
	# find V and SIGMA_IN
	H_hat = np.empty((M.shape[1],2,3))
	V = np.empty((M.shape[1],Z.shape[1],2))
	SIGMA_IN = np.empty((M.shape[1],Z.shape[1],2,2))
	for i in range(M.shape[1]):
		z_hat, H_hat_i = measurementFunction(X_hat,M[:,i])
		H_hat[i,:,:] = H_hat_i
		for j in range(Z.shape[1]):
			z = Z[:,j].reshape(2,1)
			V[i,j,:] = np.ravel(z-z_hat)
			SIGMA_IN[i,j,:,:] = np.dot(np.dot(H_hat_i,P_hat),H_hat_i.T) + R[j,:,:]
	# associate measuerements using Mahalanobis distance
	Vout=[]
	Hout=[]
	Rout=[]
	for i in range(V.shape[0]):
		for j in range(V.shape[1]):
			dist = np.dot(np.dot(V[i,j,:].T,np.linalg.inv(SIGMA_IN[i,j,:,:])),V[i,j,:])
			#print(dist)
			if(dist < g**2): 
				Vout.append(V[i,j,:])
				Rout.append(R[j,:,:])
				Hout.append(H_hat[i,:,:])
	Vout = np.array(Vout,dtype=float)
	Rout = np.array(Rout,dtype=float)
	Hout = np.array(Hout,dtype=float)
	return Vout,Hout,Rout # Vout=kx2, Hout=kx2x3, Rout=kx2x2
def filterStep(X_hat, P_hat, V, H_hat, R):
	"""
	Apply extended Kalman filter
	"""
	P_hat = P_hat.astype(float)
	R = block_diag(*R) # R=2*kx2*k
	H = np.reshape(H_hat,(-1,3)) # H=2*kx3
	V = np.reshape(V,(-1,1)) # V=2*kx1
	# Kalman Gain
	K = np.dot(np.dot(P_hat,H.T),np.linalg.inv(np.dot(np.dot(H,P_hat),H.T) + R))
	# next state 
	X = X_hat + np.dot(K,V)
	P = np.dot((np.identity(3)-np.dot(K,H)),P_hat)
	return X,P

def callback_laser(data):
	global Z,R
	lines = data.line_segments	
	Z_temp = []
	R_temp = []
	for i,line in enumerate(lines):
		Z_temp.append(np.array([line.angle,line.radius]))
		covariance = np.asarray(line.covariance)
		R_temp.append(covariance.reshape((2,2)))
	if(len(Z_temp) == 0):
		sys.exit("### The robot is stuck ###")
	Z = np.array(Z_temp).T # Z.shape = 2xk
	R = np.array(R_temp) # R.shape = kx2x2
def callback_control(data):
	global U
	U_temp = np.array([data.position[0],data.position[1]])
	U = U_temp
def callback_odom(data):
	global X_odom,P_odom,U,b
	covariance = data.pose.covariance
	P_odom_temp = np.array(covariance)
	idx = [0,1,5,6,7,11,30,31,35]
	P_odom = P_odom_temp[idx].reshape((3,3))
	pose = data.pose.pose.position
	orient = data.pose.pose.orientation
	_,_,theta = euler_from_quaternion([orient.x,orient.y,orient.z,orient.w])
	X_odom = np.array([pose.x,pose.y,theta]).T

	T = 1./30
	v = data.twist.twist.linear.x
	w = data.twist.twist.angular.z
	v_l = v-w*b
	v_r = v+w*b
	U[0] = U[0] + T*v_r
	U[1] = U[1] + T*v_l
def loadMap():
	Map = pd.read_csv('map.csv')
	M = np.zeros((2,len(Map)))
	for i in range(len(Map)):
		M[0,i] = Map.loc[i,'ro'] 
		M[1,i] = Map.loc[i,'theta']
	return M
#=====================================================
#CONTROL THE ROBOT PART
def manual(pub,in1,in2,vel):
	vel.linear.x,vel.linear.y,vel.linear.z = in1,0,0
	vel.angular.x,vel.angular.y,vel.angular.z = 0,0,in2
	pub.publish(vel)
	return
def moveRobot():
	global isPrint
	vel = Twist()
	pub_vel = rospy.Publisher('cmd_vel', Twist, queue_size = 3)
	isPrint = False
	while True:
		print("\nProvide robot velocities ('e' to exit,'p' to start printing): ")
		inp = raw_input()
		if inp == 'e':
			break
		if inp == 'p':
			isPrint = True
			continue
		else:
			isPrint = False
			try:
				in1,in2 = inp.split(' ')
				in1 = float(in1)
				in2 = float(in2)
			except:
				print("##### WRONG INPUTS #####")
				continue
			manual(pub_vel,in1,in2,vel)
#=======================================================
if __name__ == '__main__':
	M = loadMap() # load the map from the .csv file
	b = 0.16 #(in meters)
	g = 0.1

	Z = None
	R = None
	U = np.zeros((2,1))
	P_odom = np.empty((3,3))
	X_odom = np.empty((3,1))

	rospy.init_node('main',anonymous=True)
	rospy.Subscriber("line_segments",LineSegmentList, callback_laser)
	#rospy.Subscriber("joint_states",JointState, callback_control)
	rospy.Subscriber("odom",Odometry, callback_odom)

	X = np.array([[0],[0],[0]])
	P = P_odom
	while Z == None or R == None: # wait for initialization
		continue
	# create thread for moving the robot
	control_thread = Thread(target=moveRobot,args=())
	control_thread.start()
	isPrint = False
	# localization part (EKF)
	while True:
		X_prev = X
		P_prev = P

		X_hat,P_hat = getApriori(X_prev,P_prev, U, b)
		V,H_hat,R_orig = associateMeasurement(X_hat, P_hat, Z, R, M, g)
		X,P = filterStep(X_hat, P_hat, V, H_hat, R_orig)
		
		if isPrint:
			print('---'*20)
			print('POSITION:\n{}\nODOM\n{}'.format(np.ravel(X),X_odom))	
			print('COVARIANCE: \n{}\nODOM\n{}'.format(P,P_odom))
			print('---'*20)
			isPrint=False
		U = np.zeros((2,1)) # reset the control
		if not control_thread.isAlive():	
			break
		time.sleep(0.5) # set the rate
	







