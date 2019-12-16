import cv2
import numpy as np
from createChessboardPoint import *
import glob
import math
from matplotlib import pyplot as plt



def rVecTorMatrix(rotationVector):
	rotation_matrix = cv2.Rodrigues(rotationVector)
	rotation_matrix = rotation_matrix[0]

	return rotation_matrix


def cameraPoseInObjectCoordinate(translation_vector,rotation_vector):
	'''
		this function uses rotation vector in object coordinate,
		to calculate the rotation matrix between two rotations,
		A_R_B = A_R_C . C_R_B = (C_R_A)T . C_R_B
		then back transform into rotation vector
		'''

	# vector form
	camera_pose_translation = []
	camera_pose_rotation = []

	for i in range(len(translation_vector)):
		# Translation vector and Rotation matrix of object coordinate respect to camera coordinate
		# Object pose in camera coordinate
		C_translation_vector_P = translation_vector[i]
		C_rotation_matrix_P = rVecTorMatrix(rotation_vector[i])

		# Translation vector and Rotation matrix of camera coordinate respect to object coordinate
		# camera pose in object coordinate
		P_translation_vector_C = np.array(np.zeros((3, 1)))
		P_rotation_matrix_C = np.array(np.zeros((3, 3)))

		P_rotation_matrix_C = np.linalg.inv(C_rotation_matrix_P)
		P_translation_vector_C = -(np.dot(P_rotation_matrix_C, C_translation_vector_P))

		P_rotation_vector_C = cv2.Rodrigues(P_rotation_matrix_C)[0]

		camera_pose_translation.append(P_translation_vector_C)
		camera_pose_rotation.append(P_rotation_vector_C)

	return camera_pose_translation, camera_pose_rotation

def rMatrixToVector(camera_pose_rotation_matrix):
	camera_pose_rotation_vector_set = []
	for i in range(len(camera_pose_rotation_matrix)):
		camera_pose_rotation_vector = cv2.Rodrigues(camera_pose_rotation_matrix[i])
		camera_pose_rotation_vector_set.append(camera_pose_rotation_vector)

'''
def convertRMatrixToAngle(camera_pose_rotation):
	camera_pose_angle = []

	for i in range(len(camera_pose_rotation)):
		sy = math.sqrt(camera_pose_rotation[i][0, 0] * camera_pose_rotation[i][0, 0] + camera_pose_rotation[i][1, 0] * camera_pose_rotation[i][1, 0])
		rx = math.atan2(camera_pose_rotation[i][2, 1], camera_pose_rotation[i][2, 2]) / (math.pi) * 180
		ry = math.atan2(-camera_pose_rotation[i][2, 0], sy) / (math.pi) * 180
		rz = math.atan2(camera_pose_rotation[i][1, 0], camera_pose_rotation[i][0, 0]) / (math.pi) * 180

		rotation_angle = np.array([rx, ry, rz])
		camera_pose_angle.append(rotation_angle)

	return camera_pose_angle
'''

def robotAngleToRotationMatrix(angle):
	Rotation_matrix_set = []

	for i in range(len(angle)):
		c_x = math.cos(angle[i][0] * math.pi / 180)
		s_x = math.sin(angle[i][0] * math.pi / 180)
		c_y = math.cos(angle[i][1] * math.pi / 180)
		s_y = math.sin(angle[i][1] * math.pi / 180)
		c_z = math.cos(angle[i][2] * math.pi / 180)
		s_z = math.sin(angle[i][2] * math.pi / 180)

		Rx = np.array([[1, 0, 0],
		               [0, c_x, -s_x],
		               [0, s_x, c_x]])

		Ry = np.array([[c_y, 0, s_y],
		               [0, 1, 0],
		               [-s_y, 0, c_y]])

		Rz = np.array([[c_z, -s_z, 0],
		               [s_z, c_z, 0],
		               [0, 0, 1]])

		R_matrix = np.dot(np.dot(Rx, Ry), Rz)

		Rotation_matrix_set.append(R_matrix)

	return Rotation_matrix_set


def relativRotationVector(Rotation_matrix_set):
	relativ_rotation_vector_set = []

	for i in range(len(Rotation_matrix_set) - 1):
		Rab = np.dot(np.transpose(Rotation_matrix_set[i]), Rotation_matrix_set[i + 1])
		Rab_vec = cv2.Rodrigues(Rab)
		Rab_vec = Rab_vec[0]
		relativ_rotation_vector_set.append(Rab_vec)

	return relativ_rotation_vector_set


def AxisAngleRotation(relativ_rotation_vector_set):
	theta_unitVec_set = []

	for i in range(len(relativ_rotation_vector_set)):
		theta = math.sqrt(relativ_rotation_vector_set[i][0] ** 2 + relativ_rotation_vector_set[i][1] ** 2 +
		                  relativ_rotation_vector_set[i][2] ** 2)
		unit_vec = [relativ_rotation_vector_set[i][0] / theta,
		            relativ_rotation_vector_set[i][1] / theta,
		            relativ_rotation_vector_set[i][2] / theta]

		theta_unitVec_set.append([theta, unit_vec])

	return theta_unitVec_set


def relativeRotationAxisAngle(angle1):
	Rotation_matrix_set = robotAngleToRotationMatrix(angle1)
	relativ_rotation_vector_set = relativRotationVector(Rotation_matrix_set)
	theta_unitVec_set = AxisAngleRotation(relativ_rotation_vector_set)
	theta_unitVec_set = np.array(theta_unitVec_set)

	return theta_unitVec_set


def relativDifference(camera_pose_translation):
	camera_pose_translation_diff = np.diff(camera_pose_translation, axis=0)
	camera_pose_translation_distance = (np.sqrt(np.sum(np.square(camera_pose_translation_diff), axis=1)))
	return camera_pose_translation_distance



u_axis = 9
v_axis = 6

# like the camera calibration, we need the variable to store the object points in 3D coordinates and 2D coordinates
object_points_calib = []
image_points_calib = []

# this variable is defined to store the image points from original image(before calibration)
image_points_solvepnp = []
object_points_solvepnp = []

rvec_pnp = []
tvec_pnp = []

rvec_calib = []
tvec_calib = []
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
image_number = 10

mtx = np.load('mtx_f12.npy')
dist = np.load('dist_f12.npy')
mtx_ideal = np.load('mtx_f12_ideal.npy')
dist_ideal = np.load('dist_f12_ideal.npy')



# after knowing the intrisic parameter matrix, we can use the solvepnp algorithm to calculate the extrisic parameter
for counter in range(1,10):
	#img_undist = cv2.imread('calibration_result_GF_'+ str(counter)+'.png')
	img_original = cv2.imread('D:/Diplomarbeit/Bildsatz/Reinraum_f12_Test_farbe_bw_with_robotpath/chessboard/' + str(counter) + '.bmp')
	h, w = img_original.shape[:2]
	new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
	img_undist = cv2.undistort(img_original, mtx, dist, None, new_camera_matrix)
	# now the images are undistorted
	# these undistorted images are used to calculate the pose
	print(img_undist.shape[::-1])

	# find the chessboard corner after undistorted, this time the corner position must be different from the previous
	ret, corners = cv2.findChessboardCorners(img_undist,(u_axis, v_axis), None)
	corners = np.squeeze(corners,axis=1)
	corners = np.array(corners)


	if (ret == True) and counter <= 9:
		objp = createChessboardPoint(corners)
		object_points_calib.append(objp)

		chessboard_copy = img_undist.copy()
		chessboard_copy = cv2.cvtColor(chessboard_copy, cv2.COLOR_BGR2GRAY)
		corners_subpixel = cv2.cornerSubPix(chessboard_copy, corners, (11, 11), (-1, -1), criteria)
		image_points_calib.append(corners_subpixel)
		cv2.drawChessboardCorners(img_undist,(u_axis,v_axis),corners_subpixel,ret)
		cv2.namedWindow('show Corners undist '+str(counter),0)
		cv2.imshow('show Corners undist ' + str(counter),img_undist)
		cv2.waitKey(1)

		# for opencv 3.4.2 we only have solvepnp algorithm for 4 points
		image_points_solvepnp = np.array([corners_subpixel[0],corners_subpixel[8],corners_subpixel[45],corners_subpixel[53]],dtype=np.float32)
		object_points_solvepnp = np.array([objp[0],objp[8],objp[45],objp[53]],dtype=np.float32)
		image_points_solvepnp = image_points_solvepnp.reshape(4,1,2)

		# using solvePnP
		retval,rvec_solvepnp,tvec_solvepnp = cv2.solvePnP(object_points_solvepnp,image_points_solvepnp,mtx_ideal,dist_ideal,flags=cv2.SOLVEPNP_ITERATIVE)

		rvec_pnp.append(rvec_solvepnp)
		tvec_pnp.append(tvec_solvepnp)


# rvecs_new and tvecs_new are list type, needed to be in ndarray form
retvel, mtx_calib_new, dist_calib_new, rvec_calib, tvec_calib = cv2.calibrateCamera(object_points_calib,image_points_calib,chessboard_copy.shape[::-1],None, None)

P_translation_vector_C_pnp, P_rotation_vector_C_pnp = cameraPoseInObjectCoordinate(tvec_pnp,rvec_pnp)
P_translation_vector_C_calib,P_rotation_vector_C_calib = cameraPoseInObjectCoordinate(tvec_calib,rvec_calib)


P_translation_vector_C_pnp = np.array(P_translation_vector_C_pnp).reshape((9,3))
P_rotation_vector_C_pnp = np.array(P_rotation_vector_C_pnp).reshape((9,3))

P_translation_vector_C_calib = np.array(P_translation_vector_C_calib).reshape((9,3))
P_rotation_vector_C_calib = np.array(P_rotation_vector_C_calib).reshape((9,3))

#np.save('calib_pnp_rotvec.npy',P_rotation_vector_C_pnp)
#np.save('calib_pnp_tvec.npy',P_translation_vector_C_pnp)

#np.save('calib_calibration_rotvec.npy',P_rotation_vector_C_calib)
#np.save('calib_calibration_tvec.npy',P_translation_vector_C_calib)


#P_translation_vector_C_calib = np.array(P_translation_vector_C_calib)
#P_translation_vector_C_calib_distance = relativDifference(P_translation_vector_C_calib)


pass


