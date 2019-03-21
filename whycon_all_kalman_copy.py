#!/usr/bin/env python

'''
* Author List : Rahul,Vinayaka,Khoushikh,Sriram
* Functions: disarm,arm,whycon_callback,altitude_set_pid,pitch_set_pid,roll_set_pid,pid
* Global Variables:
'''
# Importing the required libraries

from plutodrone.msg import *    
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from std_msgs.msg import Int16
from std_msgs.msg import Int64
from std_msgs.msg import Float64
from pid_tune.msg import PidTune
import rospy
import time
import numpy as np


class Edrone(object):
	"""docstring for Edrone"""
	
	'''
	* Function Name: __init__
	* Logic: Initialization Function
	'''
	def __init__(self):
		
		rospy.init_node('drone_control')	# initializing ros node with name drone_control




############################################################# Kalman stuff ###################################################
		self.A = np.matrix([[1,0,0],[0,1,0],[0,0,1]])  # Process dynamics
		self.B = 0  # Control dynamics
		self.C = np.matrix([[1,0,0],[0,1,0],[0,0,1]])  # Measurement dynamics
		self.current_state_estimate = 0  # Current state estimate
		self.current_prob_estimate = np.matrix([[1,0,0],[0,1,0],[0,0,1]])  # Current probability of state estimate
		self.Q = np.matrix([[0.95,0,0],[0,0.95,0],[0,0,0.005]]) # Process covariance
		self.R = np.matrix([[10,0,0],[0,10,0],[0,0,10]])  # Measurement covariance

################################################################################################################################



		# This corresponds to your current position of drone. This value must be updated each time in your whycon callback
		# [x,y,z,yaw_value]
		self.drone_position = [0.0,0.0,0.0]	

		# [x_setpoint, y_setpoint, z_setpoint, yaw_value_setpoint]
		self.setpoint = [0.0,-1.0,20.0]


		#Declaring a cmd of message type PlutoMsg and initializing values
		self.cmd = PlutoMsg()
		self.cmd.rcRoll = 1500
		self.cmd.rcPitch = 1500
		self.cmd.rcYaw = 1500
		self.cmd.rcThrottle = 1500
		self.cmd.rcAUX1 = 1500
		self.cmd.rcAUX2 = 1500
		self.cmd.rcAUX3 = 1500
		self.cmd.rcAUX4 = 1500
		# self.cmd.plutoIndex = 0
		self.error = [0.0,0.0,0.0]
		self.kalman_estimate = 0.0
		self.normal_estimate = 0.0
		self.alt_error = Float64()          #message type for publishing error values
		self.alt_error.data = self.error[2]

		self.pitch_error = Float64()		 #message type for publishing error values
		self.pitch_error.data = self.error[0]

		self.roll_error = Float64()			 #message type for publishing error values
		self.roll_error.data = self.error[1]


		self.zero_line = Float64()			 #message type for publishing error values
		self.zero_line.data = 0.0

##################################################################################################
		self.kalman_altitude = Float64()
		# self.kalman_altitude.data = self.kalman_estimate
		self.kalman_pitch = Float64()
		self.kalman_roll = Float64()

		self.normal_altitude = Float64()
		self.normal_pitch = Float64()
		self.normal_roll = Float64()
		# self.normal_altitude.data = self.normal_estimate
		self.filtered_position = Pose()	

###################################################################################################3
		# self.Kp = [8,8,40,0] 
		# self.Ki = [0.01,0.01,0,0]
		# self.Kd = [17,17,30,0]
		self.Kp = [8,8,40,0] 
		self.Ki = [0.09,0.09,0,0]
		self.Kd = [17,17,30,0]


		# self.Kp = [0,0,40,0] 
		# self.Ki = [0,0,0.5,0]
		# self.Kd = [0,0,30,0]

		# self.Kp = [0,0,0,0] 
		# self.Ki = [0,0,0,0]
		# self.Kd = [0,0,0,0]


	
		self.prev_errors = [0.0,0.0,0.0]  # to store previous errors

		self.max_values = [1800,1800,1800] # to set maximum and minimum values for pid output.
		self.min_values = [1200,1200,1200]


		########################### SAMPLING TIME#################################################################

		self.sample_time = 0.008 

		###################################################################################################################################


		self.error_sum = [0.0,0.0,0.0] # to store summation of error ( integral of error)

		self.out_pitch = 0  # output values from the pid loop
		self.out_roll = 0
		self.out_throttle = 0
		self.out_yaw = 0
		
		self.proportional = [0,0,0]  # variables required for the pid loop
		self.integral = [0,0,0]
		self.derivative = [0,0,0]


		#----------------------- Publishers------------------------------------------------------------------------------

		# Publishing /drone_command, /alt_error, /pitch_error, /roll_error, /yaw_error
		self.command_pub = rospy.Publisher('/drone_command', PlutoMsg, queue_size=1)

		self.alt_error_pub = rospy.Publisher('/alt_error',Float64, queue_size=1)

		self.pitch_error_pub = rospy.Publisher('/pitch_error',Float64, queue_size=1)

		self.roll_error_pub = rospy.Publisher('/roll_error',Float64, queue_size=1)

		self.zero_line_pub = rospy.Publisher('/zero_line_pub',Float64, queue_size = 1)

		
		#------------------------- Kalman stuff -----------------------------------------
		self.kalman_altitude_pub = rospy.Publisher('/kalman_altitude',Float64, queue_size = 1)
		self.kalman_roll_pub = rospy.Publisher('/kalman_roll',Float64, queue_size = 1)
		self.kalman_pitch_pub = rospy.Publisher('/kalman_pitch',Float64, queue_size = 1)


		self.normal_altitude_pub = rospy.Publisher('/normal_altitude',Float64, queue_size = 1)
		self.normal_pitch_pub = rospy.Publisher('/normal_pitch',Float64, queue_size = 1)
		self.normal_roll_pub = rospy.Publisher('/normal_roll',Float64, queue_size = 1)
		#----------------------------------------------------------------------------------


		# Subscribing to /whycon/poses, /drone_yaw, /pid_tuning_altitude, /pid_tuning_pitch, pid_tuning_roll
		rospy.Subscriber('whycon/poses', PoseArray, self.whycon_callback)
		rospy.Subscriber('/pid_tuning_altitude',PidTune,self.altitude_set_pid)
		#-------------------------Add other ROS Subscribers here----------------------------------------------------

		rospy.Subscriber('/pid_tuning_pitch',PidTune,self.pitch_set_pid)

		rospy.Subscriber('/pid_tuning_roll',PidTune,self.roll_set_pid)

		#------------------------------------------------------------------------------------------------------------

		self.arm() # ARMING THE DRONE


	# Disarming condition of the drone
	def disarm(self):
		self.cmd.rcAUX4 = 1100
		self.command_pub.publish(self.cmd)
		rospy.sleep(1)


	# Arming condition of the drone : Best practise is to disarm and then arm the drone.
	def arm(self):

		self.disarm()

		self.cmd.rcRoll = 1500
		self.cmd.rcYaw = 1500
		self.cmd.rcPitch = 1500
		self.cmd.rcThrottle = 1000
		self.cmd.rcAUX4 = 1500
		self.command_pub.publish(self.cmd)	# Publishing /drone_command
		rospy.sleep(1)


	def whycon_callback(self,msg):
		self.normal_y = msg.poses[0].position.y
		# print(msg)

		# --------------------Set the remaining co-ordinates of the drone from msg----------------------------------------------

		self.normal_x = msg.poses[0].position.x  # or is it poses[1] or poses[2]

		self.normal_z = msg.poses[0].position.z
		
		filtered_position.position.x = data.markers[0].pose.pose.position.x 
		filtered_position.position.y = data.markers[0].pose.pose.position.y 	
		filtered_position.position.z = data.markers[0].pose.pose.position.z 

		self.filtered_position_pub.publish(self.filtered_position)

#-----------------------------------------------------------------------------------------------------------------------------------




	def step(self, control_input, measurement): #measurement[1,2,3]
	# Prediction step
		# predicted_state_estimate = self.A * self.current_state_estimate + self.B * control_input
		predicted_state_estimate = np.dot(self.A,self.current_state_estimate)
		# predicted_prob_estimate = (self.A * self.current_prob_estimate) * self.A + self.Q
		predicted_prob_estimate = np.dot(np.dot(self.A,self.current_prob_estimate),self.A) + self.Q

		# Observation step
		innovation = measurement - np.dot(self.C,predicted_state_estimate)
		innovation_covariance = np.dot(np.dot(self.C,predicted_prob_estimate), self.C) + self.R
		innovation_covariance = np.linalg.inv(innovation_covariance)
		# Update step
		# kalman_gain = predicted_prob_estimate * self.C * 1 / float(innovation_covariance)
		kalman_gain = np.dot(np.dot(predicted_prob_estimate,self.C),innovation_covariance)
		self.current_state_estimate = predicted_state_estimate + np.dot(kalman_gain,innovation)

		# eye(n) = nxn identity matrix.
		# self.current_prob_estimate = (1 - kalman_gain * self.C) * predicted_prob_estimate
		self.current_prob_estimate = np.dot(((np.matrix([[1,0,0],[0,1,0],[0,0,1]]))- np.dot(kalman_gain,self.C)),predicted_prob_estimate)
		# print(self.current_state_estimate,self.altitude)
		return self.current_state_estimate

	


	def update_pos(self):
		whycon_matrix = np.matrix([[self.normal_x],[self.normal_y],[self.normal_z]])
		test_variable = self.step(0,whycon_matrix)
		# self.kalman_estimate = self.drone_position[2]
		# print(test_variable[0,0])
		self.drone_position[0] = test_variable[1,0]
		self.drone_position[1] = test_variable[0,0]
		self.drone_position[2] = test_variable[2,0]

		self.kalman_altitude.data = self.drone_position[2]
		self.normal_altitude.data = self.normal_z

		self.kalman_pitch.data = self.drone_position[0]
		self.normal_pitch.data = self.normal_y

		self.kalman_roll.data = self.drone_position[1]
		self.normal_roll.data = self.normal_x
		# print(self.kalman_altitude.data)

#-----------------------------------------------------------------------------------------------------------------------------------

	# Callback function for /pid_tuning_altitude
	# This function gets executed each time when /tune_pid publishes /pid_tuning_altitude
	def altitude_set_pid(self,alt):
		self.Kp[2] = alt.Kp * 0.01 # This is just for an example. You can change the fraction value accordingly
		self.Ki[2] = alt.Ki * 0.005
		self.Kd[2] = alt.Kd * 0.9
		# print(alt)

	#----------------------------Define callback function like altitide_set_pid to tune pitch, roll and yaw as well--------------

	def pitch_set_pid(self,pitch):
		self.Kp[0] = pitch.Kp * 0.04 
		self.Ki[0] = pitch.Ki * 0.005
		self.Kd[0] = pitch.Kd * 0.02

	def roll_set_pid(self,roll):
		self.Kp[1] = roll.Kp * 0.06 
		self.Ki[1] = roll.Ki * 0.008
		self.Kd[1] = roll.Kd * 0.3

	# def yaw_set_pid(self,yaw):
	# 	self.Kp[3] = yaw.Kp * 0.06 
	# 	self.Ki[3] = yaw.Ki * 0.008
	# 	self.Kd[3] = yaw.Kd * 0.3

	def pid(self):
		
	#-----------------------------Write the PID algorithm here--------------------------------------------------------------

	# Steps:
	# 	1. Compute error in each axis. eg: error[0] = self.drone_position[0] - self.setpoint[0] ,where error[0] corresponds to error in x...
	#	2. Compute the error (for proportional), change in error (for derivative) and sum of errors (for integral) in each axis. Refer Getting_familiar_with_PID.pdf to understand PID equation.
	#	3. Calculate the pid output required for each axis. For eg: calcuate self.out_roll, self.out_pitch, etc.
	#	4. Reduce or add this computed output value on the avg value ie 1500. For eg: self.cmd.rcRoll = 1500 + self.out_roll. LOOK OUT FOR SIGN (+ or -). EXPERIMENT AND FIND THE CORRECT SIGN
	#	5. Don't run the pid continously. Run the pid only at the a sample time. self.sampletime defined above is for this purpose. THIS IS VERY IMPORTANT.
	#	6. Limit the output value and the final command value between the maximum(1800) and minimum(1200)range before publishing. For eg : if self.cmd.rcPitch > self.max_values[1]:
	#																														self.cmd.rcPitch = self.max_values[1]
	#	7. Update previous errors.eg: self.prev_error[1] = error[1] where index 1 corresponds to that of pitch (eg)
	#	8. Add error_sum


####################################### Calculate PID terms #################################################

		# print(self.drone_position)

		# if self.safe_flag == False:
		# 	self.disarm();
		# 	exit(0)

		# if self.drone_position[2] < 15 :
		# 	self.safe_flag = False

		self.update_pos()
		# print(self.drone_position[2])
		self.error = [(self.setpoint[a]-self.drone_position[a]) for a in range(0,3)]  # calculate error array

		self.error_sum = [((self.error_sum[b]+self.error[b])*self.Ki[b]*self.sample_time) for b in range(0,3)] # calculate sum of errors

		self.derivative = [((float(self.error[j]-self.prev_errors[j])/self.sample_time)*self.Kd[j]) for j in range(0,3)] #calculate derivative of errors
 
		self.proportional = [(self.error[n] * self.Kp[n]) for n in range(0,3)] # proportional term in pid equation

		for i in range(len(self.error_sum)):               # limit error_sum to max value
			if self.error_sum[i] > self.max_values[i] :
				self.error_sum[i] = self.max_values[i]

		self.out_pitch = self.proportional[0] + self.error_sum[0] + self.derivative[0]     # pid pitch
		self.out_roll = self.proportional[1] + self.error_sum[1] + self.derivative[1]      # pid roll
		self.out_throttle = self.proportional[2] + self.error_sum[2] + self.derivative[2]  # pid throttle
		# self.out_yaw = self.proportional[3] + self.error_sum[3] + self.derivative[3]       # pid yaw

		self.cmd.rcPitch = 1500 - self.out_pitch			# output Pitch

######################################   Roll sign minus #################################################################################
		self.cmd.rcRoll = 1500 + self.out_roll				# output Roll
####################################################################################################################			
		self.cmd.rcThrottle = 1500 - self.out_throttle		# output Throttle
		# self.cmd.rcYaw = 1500 + self.out_yaw				# output Yaw


####################### Limit output values##############################################################
		if self.cmd.rcPitch > self.max_values[0]:           
			self.cmd.rcPitch = self.max_values[0]
		if self.cmd.rcPitch < self.min_values[0]:
			self.cmd.rcPitch = self.min_values[0]

		if self.cmd.rcRoll > self.max_values[1]:
			self.cmd.rcRoll = self.max_values[1]
		if self.cmd.rcRoll < self.min_values[1]:
			self.cmd.rcRoll = self.min_values[1]

		if self.cmd.rcThrottle > self.max_values[2]:
			self.cmd.rcThrottle = self.max_values[2]
		if self.cmd.rcThrottle < self.min_values[2]:
			self.cmd.rcThrottle = self.min_values[2]

		# if self.cmd.rcYaw > self.max_values[3]:
		# 	self.cmd.rcYaw = self.max_values[3]
		# if self.cmd.rcYaw < self.min_values[3]:
		# 	self.cmd.rcYaw = self.min_values[3]
###########################################################################################################


		self.pitch_error.data = self.error[0] # to publish error values
		self.roll_error.data = self.error[1]
		self.alt_error.data = self.error[2]
		# self.yaw_error.data = self.error[3]
		

		self.command_pub.publish(self.cmd) # publish the commands
		# print(self.cmd)

		self.alt_error_pub.publish(self.alt_error) # publish to plotter
		self.pitch_error_pub.publish(self.pitch_error)
		self.roll_error_pub.publish(self.roll_error)
		# self.yaw_error_pub.publish(self.yaw_error)
		self.zero_line_pub.publish(self.zero_line)

		self.kalman_altitude_pub.publish(self.kalman_altitude)
		self.kalman_pitch_pub.publish(self.kalman_pitch)
		self.kalman_roll_pub.publish(self.kalman_roll)


		self.normal_altitude_pub.publish(self.normal_z)
		self.normal_pitch_pub.publish(self.normal_y)
		self.normal_roll_pub.publish(self.normal_x)

		
		self.prev_errors = [self.error[m] for m in range(0,3)] # previous error 
		
		rospy.sleep(self.sample_time) 


if __name__ == '__main__':
	e_drone = Edrone()
	# A = 1  # No process innovation
	# C = 1  # Measurement
	# B = 0  # No control input
	# Q = 0.005  # Process covariance
	# R = 1  # Measurement covariance
	# x = e_drone.drone_position[2] # Initial estimate
	# P = 1  # Initial covariance
	initial_state = np.matrix([[e_drone.normal_y],[e_drone.normal_x],[e_drone.normal_z]])
	e_drone.current_state_estimate = initial_state

	while not rospy.is_shutdown():
		e_drone.pid()
