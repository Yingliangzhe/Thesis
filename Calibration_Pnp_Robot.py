import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})


def robotAngleToRotationMatrix(angle):

	Rotation_matrix_set = []

	for i in range(len(angle)):
		c_x = math.cos(angle[i][0]*math.pi/180)

		s_x = math.sin(angle[i][0]*math.pi/180)

		c_y = math.cos(angle[i][1]*math.pi/180)

		s_y = math.sin(angle[i][1]*math.pi/180)

		c_z = math.cos(angle[i][2]*math.pi/180)

		s_z = math.sin(angle[i][2]*math.pi/180)


		Rx = np.array([[1,   0,   0],
	                   [0, c_x,-s_x],
	                   [0, s_x, c_x]])

		Ry = np.array([[c_y , 0, s_y],
		               [   0, 1,   0],
		               [-s_y, 0, c_y]])

		Rz = np.array([[c_z, -s_z, 0],
		               [s_z,  c_z, 0],
		               [  0,    0, 1]])

		R_matrix = np.dot(np.dot(Rx,Ry),Rz)

		Rotation_matrix_set.append(R_matrix)

	return Rotation_matrix_set

def robotRotationVector(rotation_matrix):
    robot_rotation_vector_set = []

    for i in range(len(rotation_matrix)):
        rotation_vec = cv2.Rodrigues(rotation_matrix[i])[0]
        robot_rotation_vector_set.append(rotation_vec)

    robot_rotation_vector_set = np.array(robot_rotation_vector_set,dtype=np.float32)

    return robot_rotation_vector_set

def relativRotationVector(Rotation_vector_set):

	'''
	this function uses rotation vector in object coordinate,
	to calculate the rotation matrix between two rotations,
	A_R_B = A_R_C . C_R_B = (C_R_A)T . C_R_B
	then back transform into rotation vector
	'''

	relativ_rotation_vector_set = []

	for i in range(len(Rotation_vector_set)-1):
		Ra = cv2.Rodrigues(Rotation_vector_set[i])[0]
		Rb = cv2.Rodrigues(Rotation_vector_set[i+1])[0]

		Rab = np.dot(np.transpose(Ra),Rb)
		Rab_vec = cv2.Rodrigues(Rab)
		Rab_vec = Rab_vec[0]
		relativ_rotation_vector_set.append(Rab_vec)

	return relativ_rotation_vector_set


def RotationMatrixToAngle(Rotation_matrix_set):
	Rotation_euler_angle = []

	for i in range(len(Rotation_matrix_set)):
		R = Rotation_matrix_set[i]
		sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

		singular = sy < 1e-6

		if not singular:
			x = math.atan2(R[2, 1], R[2, 2])
			y = math.atan2(-R[2, 0], sy)
			z = math.atan2(R[1, 0], R[0, 0])
		else:
			x = math.atan2(-R[1, 2], R[1, 1])
			y = math.atan2(-R[2, 0], sy)
			z = 0

		angle = np.array([x,y,z])

		Rotation_euler_angle.append(angle)
	return Rotation_euler_angle


def RotationVectorToMatrix(Rotation_vector_set):
	Rotation_matrix_set = []

	for i in range(len(Rotation_vector_set)):
		R_matrix = cv2.Rodrigues(Rotation_vector_set[i])[0]

		Rotation_matrix_set.append(R_matrix)

	return Rotation_matrix_set


def TranslationAndRotationDiffPlot(rotation_angle_set_robot_diff,rotation_angle_set_pnp_diff,translation_diff_robot,translation_diff_pnp):

	axis_x = list(range(0,8))

	# ---------------------
	# Rotation
	# x Achse
	angle_x_robot_diff = rotation_angle_set_robot_diff[:,0]
	angle_x_pnp_diff = rotation_angle_set_pnp_diff[:,0]
	angle_x_diff = angle_x_pnp_diff - angle_x_robot_diff
	print('angle x mean: '+ str(np.mean(angle_x_diff)))
	print('angle x std: ' + str(np.std(angle_x_diff)))


	plt.figure('Chessboard Angle Axis X')
	plt.subplot(1,2,1)
	plt.title('Rotationsänderung um Achse-X')
	plt.plot(axis_x, angle_x_robot_diff, 'r-o', label="Roboterbahn")
	plt.plot(axis_x, angle_x_pnp_diff, 'b-+', label='SolvePnP')
	plt.xlabel('Liniesegment ID')
	plt.ylabel('Größe des Winkeländerung(°)')
	plt.legend(loc="upper left")

	plt.subplot(1,2,2)
	plt.title('Betragsdifferenz der Rotationsänderung \n für Roboterbahnpunkt \n und SolvePnP um Achse-X')
	plt.plot(axis_x,(angle_x_diff),'g-*',label='solvePnP vs. Roboterbahn')
	plt.xlabel('Liniesegment ID')
	plt.ylabel('Größe der Differenz der Winkeländerung(°)')
	plt.legend(loc='upper left')

	# y Achse
	angle_y_robot_diff = rotation_angle_set_robot_diff[:, 1]
	angle_y_pnp_diff = rotation_angle_set_pnp_diff[:, 1]
	angle_y_diff = angle_y_pnp_diff - angle_y_robot_diff
	print('angle y mean: ' + str(np.mean(angle_y_diff)))
	print('angle y std: ' + str(np.std(angle_y_diff)))

	plt.figure('Chessboard Angle Axis Y')
	plt.subplot(1, 2, 1)
	plt.title('Rotationsänderung um Achse-Y')
	plt.plot(axis_x, angle_y_robot_diff, 'r-o', label="Roboterbahn")
	plt.plot(axis_x, angle_y_pnp_diff, 'b-+', label='SolvePnP')
	plt.xlabel('Liniesegment ID')
	plt.ylabel('Größe des Winkeländerung(°)')
	plt.legend(loc="upper left")

	plt.subplot(1, 2, 2)
	plt.title('Betragsdifferenz der Rotationsänderung \n für Roboterbahnpunkt \n und SolvePnP um Achse-Y')
	plt.plot(axis_x, (angle_y_diff), 'g-*', label='SolvePnP vs. Roboterbahn')
	plt.xlabel('Liniesegment ID')
	plt.ylabel('Größe der Differenz der Winkeländerung(°)')
	plt.legend(loc='upper left')

	# z Achse
	angle_z_robot_diff = rotation_angle_set_robot_diff[:, 2]
	angle_z_pnp_diff = rotation_angle_set_pnp_diff[:, 2]
	angle_z_diff = angle_z_pnp_diff - angle_z_robot_diff
	print('angle z mean: ' + str(np.mean(angle_z_diff)))
	print('angle z std: ' + str(np.std(angle_z_diff)))

	plt.figure('Chessboard Angle Axis Z')
	plt.subplot(1, 2, 1)
	plt.title('Rotationsänderung um Achse-Z')
	plt.plot(axis_x, angle_z_robot_diff, 'r-o', label="Roboterbahn")
	plt.plot(axis_x, angle_z_pnp_diff, 'b-+', label='SolvePnP')
	plt.xlabel('Liniesegment ID')
	plt.ylabel('Größe des Winkeländerung(°)')
	plt.legend(loc="upper left")

	plt.subplot(1, 2, 2)
	plt.title('Betragsdifferenz der Rotationsänderung \n für Roboterbahnpunkt \n und SolvePnP um Achse-Z')
	plt.plot(axis_x, (angle_z_diff), 'g-*', label='SolvePnP vs. Roboterbahn')
	plt.xlabel('Liniesegment ID')
	plt.ylabel('Größe der Differenz der Winkeländerung(°)')
	plt.legend(loc='upper left')

	# --------------------
	# Translation
	# --------------------

	# x Achse
	trans_x_robot_diff = translation_diff_robot[:, 0]
	trans_x_pnp_diff = translation_diff_pnp[:, 0]
	trans_x_diff = trans_x_pnp_diff - trans_x_robot_diff
	print('trans x mean: ' + str(np.mean(trans_x_diff)))
	print('trans x std: ' + str(np.std(trans_x_diff)))

	plt.figure('Chessboard Translation Axis X')
	plt.subplot(1,2,1)
	plt.title('Translationsänderung entlang Achse-X')
	plt.plot(axis_x, trans_x_robot_diff, 'r-o', label='Roboterbahn')
	plt.plot(axis_x, trans_x_pnp_diff, 'b-+', label='SolvePnP')
	plt.xlabel('Liniesegment ID')
	plt.ylabel('Größe der Translationsänderung(mm)')
	plt.legend(loc="upper left")

	plt.subplot(1,2,2)
	plt.title('Betragsdifferenz der Translationsänderung \n für Roboterbahnpunkt \n und SolvePnP entlang Achse-X')
	plt.plot(axis_x,(trans_x_diff),'g-*',label='SolvePnP vs. Roboterbahn')
	plt.xlabel('Liniesegment ID')
	plt.ylabel('Größe der Differenz der Translationsänderung(mm)')
	plt.legend(loc="upper left")

	# y Achse
	trans_y_robot_diff = translation_diff_robot[:, 1]
	trans_y_pnp_diff = translation_diff_pnp[:, 1]
	trans_y_diff = trans_y_pnp_diff - trans_y_robot_diff
	print('trans y mean: ' + str(np.mean(trans_y_diff)))
	print('trans y std: ' + str(np.std(trans_y_diff)))

	plt.figure('Chessboard Translation Axis Y')
	plt.subplot(1, 2, 1)
	plt.title('Translationsänderung entlang Achse-Y')
	plt.plot(axis_x, trans_y_robot_diff, 'r-o', label='Roboterbahn')
	plt.plot(axis_x, trans_y_pnp_diff, 'b-+', label='SolvePnP')
	plt.xlabel('Liniesegment ID')
	plt.ylabel('Größe der Translationsänderung(mm)')
	plt.legend(loc="upper left")

	plt.subplot(1, 2, 2)
	plt.title('Betragsdifferenz der Translationsänderung \n für Roboterbahnpunkt \n und SolvePnP entlang Achse-Y')
	plt.plot(axis_x, (trans_y_diff), 'g-*', label='SolvePnP vs. Roboterbahn')
	plt.xlabel('Liniesegment ID')
	plt.ylabel('Größe der Differenz der Translationsänderung(mm)')
	plt.legend(loc="upper left")

	# z-Achse
	trans_z_robot_diff = translation_diff_robot[:, 2]
	trans_z_pnp_diff = translation_diff_pnp[:, 2]
	trans_z_diff = trans_z_pnp_diff - trans_z_robot_diff
	print('trans z mean: ' + str(np.mean(trans_z_diff)))
	print('trans z std: ' + str(np.std(trans_z_diff)))

	plt.figure('Chessboard Translation Axis Z')
	plt.subplot(1, 2, 1)
	plt.title('Translationsänderung entlang Achse-Z')
	plt.plot(axis_x, trans_z_robot_diff, 'r-o', label='Roboterbahn')
	plt.plot(axis_x, trans_z_pnp_diff, 'b-+', label='SolvePnP')
	plt.xlabel('Liniesegment ID')
	plt.ylabel('Größe der Translationsänderung(mm)')
	plt.legend(loc="upper left")

	plt.subplot(1, 2, 2)
	plt.title('Betragsdifferenz der Translationsänderung \n für Roboterbahnpunkt \n und SolvePnP entlang Achse-Z')
	plt.plot(axis_x, (trans_z_diff), 'g-*', label='SolvePnP vs. Roboterbahn')
	plt.xlabel('Liniesegment ID')
	plt.ylabel('Größe der Differenz der Translationsänderung(mm)')
	plt.legend(loc="upper left")




if __name__ == "__main__":
	pPoint_trans = [[7.79, 2.4, 492],
	                [18.17, 2.4, 492],
	                [37.42, 2.4, 492],
	                [2.07, 20.26, 492],
	                [18.17, 20.26, 492],
	                [38.32, 20.26, 492],
	                [1.66, 42.56, 492],
	                [18.17, 42.56, 492],
	                [37.41, 42.56, 492]]

	pPoint_trans = np.array(pPoint_trans, dtype=np.float32)

	pPoint_rot = [[180, 0, 0],
	              [180, 0, 0],
	              [180, 0, 0],
	              [180, 0, 0],
	              [180, 0, 0],
	              [180, 0, 0],
	              [180, 0, 0],
	              [180, 0, 0],
	              [180, 0, 0]]

	pPoint_rot = np.array(pPoint_rot, dtype=np.float32)

	P_translation_vector_C_pnp = np.load('calib_pnp_tvec.npy')
	P_rotation_vector_C_pnp = np.load('calib_pnp_rotvec.npy')

	# 1. robot path
	# angle --> rotation matrix
	# Rotation angle
	rotation_matrix_set_robot = robotAngleToRotationMatrix(pPoint_rot)
	rotation_angle_set_robot = RotationMatrixToAngle(rotation_matrix_set_robot)
	rotation_angle_set_robot = np.array(rotation_angle_set_robot,dtype=np.float32)
	rotation_angle_set_robot_diff = np.diff(rotation_angle_set_robot,axis=0)
	# Translation
	translation_diff_robot = np.diff(pPoint_trans, axis=0)


	# 2. points from solvepnp
	# Rotation Angle
	rotation_matrix_set_pnp = RotationVectorToMatrix(P_rotation_vector_C_pnp)
	rotation_angle_set_pnp = RotationMatrixToAngle(rotation_matrix_set_pnp)
	rotation_angle_set_pnp = np.array(rotation_angle_set_pnp,dtype=np.float32)
	rotation_angle_set_pnp_diff = np.diff(rotation_angle_set_pnp,axis=0)
	# Translation
	translation_diff_pnp = np.diff(P_translation_vector_C_pnp, axis=0)

	'''
		plot the result
	'''
	rotation_angle_set_robot_diff = np.around(rotation_angle_set_robot_diff,decimals=2)
	rotation_angle_set_pnp_diff = np.around(rotation_angle_set_pnp_diff,decimals=2)
	translation_diff_robot = np.around(translation_diff_robot,decimals=2)
	translation_diff_pnp = np.around(translation_diff_pnp,decimals=2)

	TranslationAndRotationDiffPlot(rotation_angle_set_robot_diff,rotation_angle_set_pnp_diff,translation_diff_robot,translation_diff_pnp)

	plt.show()
	pass


