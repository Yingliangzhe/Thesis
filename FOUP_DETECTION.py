import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import glob
import math
import scipy


# if the circle middle point is known, then we can segment the image into many patches
def segment_image(source_image, circle_y, circle_x, circle_radius, pixels):
	# primitive information of flange
	# y first, x second
	if pixels > 6000000:
		half_flange = 650
		corner_dis_1 = 550
		corner_dis_2 = 274
		circle_radius = circle_radius - 120
	else:
		half_flange = 500
		corner_dis_1 = 450
		corner_dis_2 = 200
		circle_radius = circle_radius - 50


	circle_y = int(circle_y)
	circle_x = int(circle_x)
	circle_radius = int(circle_radius)

	# Top 3 corners segment
	# there is also the problem in segment, the patch origin can be negative, smaller as 0.
	# so here would be a judgement to calculate the case
	if circle_y - half_flange - circle_radius < 0:

		TM_Mask = source_image[0 : circle_y - half_flange + circle_radius,circle_x - circle_radius : circle_x + circle_radius]
		TM_pos = (0, circle_y - half_flange + circle_radius, circle_x - circle_radius,
				  circle_x + circle_radius)

		TL_Mask = source_image[0 : circle_y - half_flange + circle_radius,circle_x - corner_dis_1 - circle_radius:circle_x - corner_dis_1 + circle_radius]
		TL_pos = (0,circle_y - half_flange + circle_radius,circle_x - corner_dis_1 - circle_radius,circle_x - corner_dis_1 + circle_radius)

		TR_Mask = source_image[0 : circle_y - half_flange + circle_radius,circle_x + corner_dis_2 - circle_radius:circle_x + corner_dis_2 + circle_radius]
		TR_pos = (0, circle_y - half_flange + circle_radius,
		          circle_x + corner_dis_2 - circle_radius, circle_x + corner_dis_2 + circle_radius)

	else:

		TM_Mask = source_image[circle_y - half_flange - circle_radius: circle_y - half_flange + circle_radius,
		          circle_x - circle_radius: circle_x + circle_radius]
		TM_pos = (circle_y - half_flange - circle_radius, circle_y - half_flange + circle_radius, circle_x - circle_radius,
				  circle_x + circle_radius)

		TL_Mask = source_image[circle_y - half_flange - circle_radius: circle_y - half_flange + circle_radius,
		          circle_x - corner_dis_1 - circle_radius:circle_x - corner_dis_1 + circle_radius]
		TL_pos = (circle_y - half_flange - circle_radius, circle_y - half_flange + circle_radius,
		          circle_x - corner_dis_1 - circle_radius, circle_x - corner_dis_1 + circle_radius)

		TR_Mask = source_image[circle_y - half_flange - circle_radius: circle_y - half_flange + circle_radius,
		          circle_x + corner_dis_2 - circle_radius:circle_x + corner_dis_2 + circle_radius]
		TR_pos = (circle_y - half_flange - circle_radius, circle_y - half_flange + circle_radius,
		          circle_x + corner_dis_2 - circle_radius, circle_x + corner_dis_2 + circle_radius)

	# Left 2 corners segment
	LM_Mask = source_image[circle_y - circle_radius : circle_y + circle_radius, circle_x - half_flange - circle_radius : circle_x - half_flange + circle_radius]
	LM_pos = (circle_y - circle_radius,circle_y + circle_radius, circle_x - half_flange - circle_radius,circle_x - half_flange + circle_radius)

	LT_Mask = source_image[circle_y - corner_dis_2 - circle_radius : circle_y - corner_dis_2 + circle_radius,circle_x - half_flange - circle_radius : circle_x - half_flange + circle_radius]
	LT_pos = (circle_y - corner_dis_2 - circle_radius,circle_y - corner_dis_2 + circle_radius,circle_x - half_flange - circle_radius,circle_x - half_flange + circle_radius)

	# Right one corner segment
	RM_Mask = source_image[circle_y - circle_radius : circle_y + circle_radius, circle_x + half_flange - circle_radius : circle_x + half_flange + circle_radius]
	RM_pos = (circle_y - circle_radius,circle_y + circle_radius, circle_x + half_flange - circle_radius,circle_x + half_flange + circle_radius)

	# Bottom 2 corners segment
	BM_Mask = source_image[circle_y + half_flange - circle_radius : circle_y + half_flange + circle_radius,circle_x - circle_radius : circle_x + circle_radius]
	BM_pos = (circle_y + half_flange - circle_radius,circle_y + half_flange + circle_radius,circle_x - circle_radius,circle_x + circle_radius)

	BR_Mask = source_image[circle_y + half_flange - circle_radius : circle_y + half_flange + circle_radius,circle_x + corner_dis_1 - circle_radius:circle_x + corner_dis_1 + circle_radius]
	BR_pos = (circle_y + half_flange - circle_radius,circle_y + half_flange + circle_radius,circle_x + corner_dis_1 - circle_radius,circle_x + corner_dis_1 + circle_radius)

	# store the Patchs in a dictionary
	Patchs = {'TM':TM_Mask,'TL':TL_Mask,'TR':TR_Mask,'LM':LM_Mask,'LT':LT_Mask,'RM':RM_Mask,'BM':BM_Mask,'BR':BR_Mask}

	# store the Patch position in a dictionary for later
	Patch_pos = {'TM':TM_pos,'TL':TL_pos,'TR':TR_pos,'LM':LM_pos,'LT':LT_pos,'RM':RM_pos,'BM':BM_pos,'BR':BR_pos}


	return Patchs, Patch_pos

def templateMatchingCircleDetection(img_edge):

	template = cv2.imread('circleTemplate_f12.png')

	template_gamma = exposure.adjust_gamma(template,0.35)
	template_filtered = cv2.bilateralFilter(template_gamma, 7, 75, 75)
	template_edge = cv2.Canny(template_filtered, 25,38, 3)

	cv2.namedWindow('flange',0)
	cv2.imshow('flange',img_edge)
	cv2.imshow('template',template_edge)
	cv2.waitKey(1)

	w,h = template_edge.shape[0:2][::-1]
	res = cv2.matchTemplate(img_edge,template_edge,cv2.TM_CCOEFF_NORMED)

	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	top_left = max_loc
	bottom_right = (top_left[0] + w, top_left[1] + h)
	cv2.rectangle(img_edge, top_left, bottom_right, 255, 2)

	plt.figure()
	#plt.subplot(121), plt.imshow(res, cmap='gray')
	#plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
	#plt.subplot(122), \
	plt.imshow(img_edge, cmap='gray')
	plt.title('Detected Point')
	plt.show()

	middle_y = top_left[1]+h/2
	middle_x = top_left[0]+w/2
	window_radius = max(w/2,h/2)

	return middle_x,middle_y,window_radius


def houghCircleDetection(img,img_edge,methode,dp,minDist,canny_upper,accumulator,minRadius,maxRadius):
	circles1 = cv2.HoughCircles(img_edge,methode,dp,minDist,param1=canny_upper,param2=accumulator,minRadius=minRadius,maxRadius=maxRadius)
	circles = circles1[0,:,:] # convert in 2D
	for i in circles[:]:
		cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)  # 画圆
		cv2.circle(img, (i[0], i[1]), 1, (255, 0, 0), 6)  # 画圆心

	# this step is used to convert the float type into Integer,
	# because the image cut is only in Integer

	circle_x = circles[0][0].astype(int)
	circle_y = circles[0][1].astype(int)
	circle_radius = circles[0][2].astype(int)

	while (1):
		cv2.namedWindow('circle',0)
		cv2.imshow('circle', img)
		k = cv2.waitKey(1) & 0xFF
		if k == 27:  # hit escape to quit
			cv2.destroyAllWindows()
			break


	return circle_x,circle_y,circle_radius

def loadTemplate(file_name_set):
	templates = {}
	corner_list = sorted(glob.glob(file_name_set))

	for img_name in corner_list:
		temp_name = img_name.split('\\')[-1].split('.')[-2]
		img = cv2.imread(img_name)
		single_template = dict(zip([temp_name], [img]))
		templates.update(single_template)

	return templates

def NotchTemplateMatching(Patchs_oringinal,templates):
	# using template matching to locate the clamp corner position for every robot handling flange
	clamp_origin_patch = {}
	clamp_image = {}

	for patch in Patchs_oringinal:
		for temp in templates:
			if patch in temp:
				patch_img = Patchs_oringinal[patch]
				patch_img_gamma = exposure.adjust_gamma(patch_img,0.1)
				# patch_img = cv2.equalizeHist(patch_img)
				#patch_img_filtered = cv2.GaussianBlur(patch_img, (5, 5), 0)
				patch_img_filtered = cv2.bilateralFilter(patch_img_gamma,5,75,75)
				patch_img_edge = cv2.Canny(patch_img_filtered, 13, 25)
				temp_img = templates[temp]
				temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
				temp_img = exposure.adjust_gamma(temp_img,0.2)
				# temp_img = cv2.equalizeHist(temp_img)
				#temp_img_filtered = cv2.GaussianBlur(temp_img, (5, 5), 0)
				temp_img_filtered = cv2.bilateralFilter(temp_img,7,75,75)
				temp_img_edge = cv2.Canny(temp_img_filtered, 14, 31)

				cv2.imshow('patch', patch_img_edge)
				cv2.imshow('template', temp_img_edge)
				cv2.waitKey(1)

				w, h = temp_img_edge.shape[0:2][::-1]
				# Apply template Matching
				# maxLoc return (x,y) format coordinate

				res = cv2.matchTemplate(patch_img_edge, temp_img_edge, cv2.TM_CCORR_NORMED)

				min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

				# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
				top_left = max_loc

				# for later segment options
				bottom_right = (top_left[0] + h, top_left[1] + w)
				# bottom_right = (top_left[0] + w, top_left[1] + h + int(w-h))

				cv2.rectangle(patch_img_edge, top_left, bottom_right, 255, 2)

				plt.figure(temp)
				plt.subplot(221), plt.imshow(res, cmap='gray')
				plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
				plt.subplot(222), plt.imshow(patch_img_edge, cmap='gray')
				plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

				#plt.show()

				# here we store the clamp coordinate in the patchs
				# this position is needed to be stored in the origin of new coordinate
				top_left = list(top_left)
				top_left.reverse()
				top_left = tuple(top_left)
				single_clamp_pos = dict(zip([temp], [top_left]))
				clamp_origin_patch.update(single_clamp_pos)

				# here we store the clamp image into a dict
				single_clamp_image = dict(zip([temp], [patch_img[top_left[0]:top_left[0] + h, top_left[1]:top_left[1] + w]]))
				clamp_image.update(single_clamp_image)

				#plt.figure(temp)
				plt.subplot(212)
				plt.imshow(single_clamp_image[temp], cmap='gray')
				plt.show()

				break
	return clamp_origin_patch,clamp_image

def PatchOriginToInitial(Patchs_oringinal,Patch_pos,y_size,x_size):
	# segment original image into Patches for later template matching use
	# this block is only used to store the origin of the patch coordiante
	patch_origin = {}  # dict to store patch origin coordinate in the original image
	img_empty = np.zeros((y_size, x_size), np.uint8)
	for patch_name in Patchs_oringinal:
		# for every left top corner, min coordinate of every patch
		patch_y_min = Patch_pos[patch_name][0]
		patch_x_min = Patch_pos[patch_name][2]
		patch_y_max = Patch_pos[patch_name][1]
		patch_x_max = Patch_pos[patch_name][3]
		patch_min_pos = (patch_y_min, patch_x_min)  # origin of coordinates of patchs in original image
		# store the single patch position for later dict function update
		single_patch_origin = dict(zip([patch_name], [patch_min_pos]))
		# store all patchs position in this dict, for every patch there is always a patch name
		patch_origin.update(single_patch_origin)
		img_empty[patch_y_min:patch_y_max, patch_x_min:patch_x_max] = Patchs_oringinal[patch_name]

	plt.imshow(img_empty, cmap='gray')
	plt.title('Patch Work in Empty image'), plt.xticks([]), plt.yticks([])
	plt.show()

	return patch_origin
		#print(Patchs_oringinal[patch_name].shape)
		#img_empty[patch_y_min:patch_y_max, patch_x_min:patch_x_max] = Patchs_oringinal[patch_name]

def clampOriginToInitial(clamp_image,clamp_origin_patch,patch_origin,y_size,x_size):
	# back transformation from clamp origin to original image coordinate
	# dict stores a tuple of all position
	clamp_origin_initial = {}
	image_empty = np.zeros((y_size, x_size), np.uint8)
	for clamp_name in clamp_image:
		# use np.array to change tuple into ndarray, and use tuple back into tuple
		single_clamp_origin_initial = dict(zip([clamp_name], [tuple(np.array(clamp_origin_patch[clamp_name]) + np.array(patch_origin[clamp_name]))]))
		clamp_y_min = single_clamp_origin_initial[clamp_name][0]
		clamp_x_min = single_clamp_origin_initial[clamp_name][1]
		clamp_y_max = clamp_y_min + clamp_image[clamp_name].shape[0]
		clamp_x_max = clamp_x_min + clamp_image[clamp_name].shape[1]
		image_empty[clamp_y_min:clamp_y_max, clamp_x_min:clamp_x_max] = clamp_image[clamp_name]
		clamp_origin_initial.update(single_clamp_origin_initial)

	plt.figure('clamp in original image')
	plt.imshow(image_empty, cmap='gray')
	plt.title('grip notch in original image origin')
	plt.show()

	return clamp_origin_initial

def findCrossLine(negative_lines,positive_lines,template_name):
	distance_minus = negative_lines[:, 0]
	distance_plus = positive_lines[:, 0]

	if template_name == 'TM':
		distance_to_ref_nega = max(distance_minus)
		distance_to_ref_posi = min(distance_plus)
	elif template_name == 'LM':
		distance_to_ref_nega = min(distance_minus)
		distance_to_ref_posi = min(distance_plus)
	elif template_name == 'RM':
		distance_to_ref_nega = max(distance_minus)
		distance_to_ref_posi = max(distance_plus)
	elif template_name == 'BM':
		distance_to_ref_nega = min(distance_minus)
		distance_to_ref_posi = max(distance_plus)
	else:
		print('not exactly the lines we want')
		print('position of FOUP can not be calculated')

	distance_nega_index = np.where(distance_minus == distance_to_ref_nega)
	distance_posi_index = np.where(distance_plus == distance_to_ref_posi)

	print('negative lines shape')
	print(negative_lines.shape)

	print('positive lines shape')
	print(positive_lines.shape)

	nega = negative_lines[distance_nega_index[0], 1]
	posi = positive_lines[distance_posi_index[0], 1]

	print('nega line is :')
	print(nega[0])
	print('nega shape is :')
	print(nega.shape)

	print('posi line is :')
	print(posi[0])
	print('posi shape is :')
	print(posi.shape)

	cross_line = [nega[0],posi[0]]

	return cross_line

def findLine(cross_line_dict):
	intersection_point = {}

	for line_name in cross_line_dict:
		line_1 = cross_line_dict[line_name][0]
		line_2 = cross_line_dict[line_name][1]

		# y = a1x + b1
		x_1 = line_1[0][0]
		y_1 = line_1[0][1]
		x_2 = line_1[0][2]
		y_2 = line_1[0][3]

		# y = a2x + b2
		x_3 = line_2[0][0]
		y_3 = line_2[0][1]
		x_4 = line_2[0][2]
		y_4 = line_2[0][3]

		a_1 = abs((y_1 - y_2) / (x_1 - x_2))
		b_1 = y_1 - a_1 * x_1

		a_2 = -abs((y_3 - y_4) / (x_3 - x_4))
		b_2 = y_3 - a_2 * x_3

		p = findIntersection(a_1,b_1,a_2,b_2)
		single_point = dict(zip([line_name], [p]))
		intersection_point.update(single_point)

	return intersection_point

def findIntersection(a_1,b_1,a_2,b_2):
	# this method is not a very robust solution,
	# if the slope is infinity, this method will be failed
	intersec_x = -(b_2 - b_1)/(a_2 - a_1)
	intersec_y = a_1 * intersec_x + b_1

	return (intersec_y,intersec_x)


def intersectionInInitial(clamp_origin_initial,intersection_point):
	# this block is used to
	intersection_point_in_initial = {}
	for clamp_name in intersection_point:
		single_intersection_point = dict(zip([clamp_name],[np.array(clamp_origin_initial[clamp_name])
		                                                   + np.array(intersection_point[clamp_name])]))
		intersection_point_in_initial.update(single_intersection_point)

	return intersection_point_in_initial


def organizeData(intersection_point_in_initial_2d):
	# this function is used to organize the data after image processing
	points_2d = []
	object_points_3d = [(0.0, 65.3, 0.0), (-65.3, 0.0, 0.0), (65.3, 0.0, 0.0), (0.0, -65.3, 0.0)]
	single_2d_points = []

	# the sequence of object points is TM LM RM BM
	for clamp_name in intersection_point_in_initial_2d:
		single_2d_points = list(intersection_point_in_initial[clamp_name])
		single_2d_points.reverse()
		single_2d_points = tuple(single_2d_points)

		points_2d.append(single_2d_points)

	object_points_3d = np.array(object_points_3d,dtype=np.float64)
	points_2d = np.array(points_2d,dtype=np.float32)
	points_2d = points_2d.reshape(4,1,2)

	return points_2d,object_points_3d


def poseEstimation(points_2d,points_3d,mtx,dist,algo_flag):

	#camera_matrix = np.eye(3)
	#dist_coeffs = np.zeros((1,5))
	camera_matrix = mtx
	dist_coeffs = dist

	retval, rvec, tvec = cv2.solvePnP(points_3d, points_2d, camera_matrix, dist_coeffs,flags=algo_flag)


	return rvec,tvec

# convert the rotation vector into matrix
def rVecTorMatrix(rotationVector):
	rotation_matrix = cv2.Rodrigues(rotationVector)
	rotation_matrix = rotation_matrix[0]

	return rotation_matrix

# transform the object pose with respect to camera into camera pose in object coordinate
# the input for this function must be a ndarray
def cameraPoseInObjectCoordinate(translation_vector,rotation_vector):

	# both of these variable are vector
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

		# transformation into object coordinate
		P_rotation_matrix_C = np.linalg.inv(C_rotation_matrix_P)

		P_rotation_vector_C = rVecTorMatrix(P_rotation_matrix_C)
		if P_rotation_vector_C[0]<0:
			P_rotation_vector_C[0] = -P_rotation_vector_C[0]

		P_rotation_matrix_C = cv2.Rodrigues(P_rotation_vector_C)[0]

		P_translation_vector_C = -(np.dot(P_rotation_matrix_C, C_translation_vector_P))

		P_rotation_vector_C = cv2.Rodrigues(P_rotation_matrix_C)[0]

		camera_pose_translation.append(P_translation_vector_C)
		camera_pose_rotation.append(P_rotation_vector_C)

	return camera_pose_translation, camera_pose_rotation

if __name__ == '__main__':

	pixels_0 = 5000000
	pixels_1 = 6000000

	C_camera_pose_translation_P = []
	C_camera_pose_rotation_P = []

	# camera parameter
	mtx = np.load('mtx.npy')
	dist = np.load('dist.npy')
	mtx_ideal = np.load('mtx_f12.npy')
	dist_ideal = np.load('dist_f12.npy')

	for counter in range(1, 10):
		img_original_0 = cv2.imread(
			'D:/Diplomarbeit/Bildsatz/Reinraum_f12_Test_farbe_bw_with_robotpath/foup_mit_licht_total_bw/' + str(
				counter) + '.bmp')
		# img_original_0 = cv2.imread('D:/Diplomarbeit/Bildsatz/Mono/handle_bbgf_12.5_pantex/view_point_2.bmp')
		h, w = img_original_0.shape[:2]
		new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
		img_gray_0 = cv2.cvtColor(img_original_0, cv2.COLOR_BGR2GRAY)

		y_size = img_gray_0.shape[0]
		x_size = img_gray_0.shape[1]

		# ------------------------------------------------------
		# -------------------gamma correction ------------------
		# ------------------------------------------------------
		plt.figure()
		plt.imshow(img_gray_0, cmap='gray')

		img_gray_0_gamma = exposure.adjust_gamma(img_gray_0, 0.35)

		plt.figure()
		plt.imshow(img_gray_0_gamma, cmap='gray')
		plt.show()

		img_edge_0 = cv2.Canny(cv2.bilateralFilter(img_gray_0_gamma, 7, 75, 75), 25, 34)

		plt.figure()
		plt.imshow(img_gray_0, cmap='gray')
		plt.show()

		# --------------------------------------------------
		# middle circle detection using hough transformation
		# --------------------------------------------------
		# maybe this is also needed for hough transformation
		# middle_x_0,middle_y_0,window_radius_0 = houghCircleDetection(img_original_0,img_edge_0,cv2.HOUGH_GRADIENT,1,40,20,30,220,250)#200 230
		# middle_x_1,middle_y_1,window_radius_1 = houghCircleDetection(img_original_1,img_edge_1,cv2.HOUGH_GRADIENT,1,200,20,40,230,250)#1,200,20,40,230,250

		# --------------------------------------------------
		# middle circle detection using template matching
		# --------------------------------------------------
		middle_x_0, middle_y_0, window_radius_0 = templateMatchingCircleDetection(img_edge_0)

		Patchs_oringinal_0, Patch_pos_0 = segment_image(img_gray_0, middle_y_0, middle_x_0, window_radius_0, pixels_0)

		'''
		template matching to roughly locate a corner
		'''

		file_name_set = './flange_gf_template_square/' + '*.png'
		templates = loadTemplate(file_name_set)

		# this variable used to store the position of the clamp position, after template matching segment
		clamp_origin_patch_0, clamp_image_0 = NotchTemplateMatching(Patchs_oringinal_0, templates)

		patch_origin_0 = PatchOriginToInitial(Patchs_oringinal_0, Patch_pos_0, y_size, x_size)

		clamp_origin_initial_0 = clampOriginToInitial(clamp_image_0, clamp_origin_patch_0, patch_origin_0, y_size,
		                                              x_size)

		# ----------------------------------------------------
		# ---------------- LSD line detector -----------------
		# ----------------------------------------------------

		y_size = clamp_image_0['TM'].shape[0]
		x_size = y_size

		lsd_detector = cv2.createLineSegmentDetector(0)

		cross_line_dict = {}
		for template_name in clamp_image_0:
			img = clamp_image_0[template_name]

			img = cv2.equalizeHist(img, None)
			img = exposure.adjust_gamma(img, 0.35)

			print(img.shape)

			lines = lsd_detector.detect(img)[0]
			print(len(lines))

			# -----------------------------------
			#    filter the short segment
			# -----------------------------------

			temp_line = []
			negative_lines = []
			positive_lines = []

			for line_index in range(0, len(lines)):
				print(lines[line_index])
				print(lines[line_index].shape)
				x_0 = lines[line_index][0][0]
				y_0 = lines[line_index][0][1]
				x_1 = lines[line_index][0][2]
				y_1 = lines[line_index][0][3]

				y_difference = y_1 - y_0
				x_difference = x_1 - x_0

				p_0 = np.asarray([y_0, x_0])
				p_1 = np.asarray([y_1, x_1])

				# define the theta angle, so that it is better to distinguish the lines
				theta = np.arctan(y_difference / x_difference) * 180 / np.pi * -1

				if (theta > 30 and theta < 60) or (theta > -60 and theta < -30):
					if math.sqrt((y_1 - y_0) ** 2 + (x_1 - x_0) ** 2) > 30:
						temp_line.append(lines[line_index])

						if theta < 0:
							reference_point = np.asarray([y_size, 0])
							# calculate the distance from (200,0) to line segment
							distance_bottom = np.linalg.norm(
								np.cross(p_1 - reference_point, p_0 - reference_point)) / np.linalg.norm(p_1 - p_0)
							negative_lines.append([distance_bottom, lines[line_index]])
						else:
							reference_point = np.asarray([0, 0])
							distance_top = np.linalg.norm(
								np.cross(p_1 - reference_point, p_0 - reference_point)) / np.linalg.norm(p_1 - p_0)
							positive_lines.append([distance_top, lines[line_index]])

			negative_lines = np.asarray(negative_lines)
			positive_lines = np.asarray(positive_lines)

			print('temp_line is:')
			print(temp_line)
			print('temp line shape is :')
			print(temp_line[0].shape)

			temp_line = np.array(temp_line)

			if template_name == 'TM' or template_name == 'LM' or template_name == 'RM' or template_name == 'BM':
				# if  template_name == 'LM' or template_name == 'RM' or template_name == 'BM':
				cross_line = findCrossLine(negative_lines, positive_lines, template_name)
				single_cross_line = dict(zip([template_name], [cross_line]))
				cross_line_dict.update(single_cross_line)
				cross_line = np.array(cross_line)

				drawn_img_none = lsd_detector.drawSegments(img, cross_line)

				plt.figure(template_name)
				plt.subplot(2, 3, 1), plt.imshow(drawn_img_none, cmap='gray')
				plt.title('line in original')

			else:
				temp_line = np.array(temp_line)
				drawn_img_none = lsd_detector.drawSegments(img, temp_line)

				plt.figure(template_name)
				plt.subplot(2, 3, 1), plt.imshow(drawn_img_none, cmap='gray')
				plt.title('line in original')

			img_empty_none = np.zeros((y_size, x_size), np.uint8)
			if template_name == 'TM' or template_name == 'LM' or template_name == 'RM' or template_name == 'BM':
				# if  template_name == 'LM' or template_name == 'RM' or template_name == 'BM':
				drawn_on_empty_none = lsd_detector.drawSegments(img_empty_none, cross_line)
			else:
				drawn_on_empty_none = lsd_detector.drawSegments(img_empty_none, temp_line)

			edge_image_none = drawn_on_empty_none.copy()
			edge_image_none = cv2.cvtColor(edge_image_none, cv2.COLOR_BGR2GRAY)

			plt.subplot(2, 3, 4), plt.imshow(edge_image_none, cmap='gray')
			plt.title('edge image')

			# --------------------------------------------------------
			# ----------------- histrogram operation -----------------
			# --------------------------------------------------------

			img_hist = cv2.equalizeHist(img, None)
			# img_hist = cv2.GaussianBlur(img_hist,(9,9),0)
			img_hist = exposure.adjust_gamma(img_hist, 0.35)
			lines_hist = lsd_detector.detect(img_hist)[0]

			drawn_img_hist = lsd_detector.drawSegments(img_hist, lines_hist)

			plt.subplot(2, 3, 2), plt.imshow(drawn_img_hist, cmap='gray')  # 默认彩色，另一种彩色bgr
			plt.title('line in equhist')

			img_empty_hist = np.zeros((y_size, x_size), np.uint8)
			drawn_on_empty_hist = lsd_detector.drawSegments(img_empty_hist, lines_hist)
			edge_image_hist = drawn_on_empty_hist.copy()
			edge_image_hist = cv2.cvtColor(edge_image_hist, cv2.COLOR_BGR2GRAY)

			plt.subplot(2, 3, 5), plt.imshow(edge_image_hist, cmap='gray')
			plt.title('edge hist')
			# --------------------------------------------------------
			# ------------------- gamma operation --------------------
			# --------------------------------------------------------
			img_gamma = exposure.adjust_gamma(img, 0.2)

			lines_gamma = lsd_detector.detect(img_gamma)[0]

			drawn_img_gamma = lsd_detector.drawSegments(img_gamma, lines_gamma)

			plt.subplot(2, 3, 3), plt.imshow(drawn_img_gamma, cmap='gray')  # 默认彩色，另一种彩色bgr
			plt.title('line in gamma')

			img_empty_gamma = np.zeros((y_size, x_size), np.uint8)
			drawn_on_empty_gamma = lsd_detector.drawSegments(img_empty_gamma, lines_gamma)
			edge_image_gamma = drawn_on_empty_gamma.copy()
			edge_image_gamma = cv2.cvtColor(edge_image_gamma, cv2.COLOR_BGR2GRAY)

			plt.subplot(2, 3, 6), plt.imshow(edge_image_gamma, cmap='gray')
			plt.title('edge gamma')

		plt.show()

		intersection_point = findLine(cross_line_dict)

		# this intersection_point_in_initial is the container for TM LM RM BM ,4 points
		intersection_point_in_initial = intersectionInInitial(clamp_origin_initial_0, intersection_point)

		# we already have the points in 2d image plane, and the points in 3d world coordinate are alse known,
		# so we can use the solvepnp method to recover the pose of object
		points_2d, points_3d = organizeData(intersection_point_in_initial)

		rotation_vector, translation_vector = poseEstimation(points_2d, points_3d, mtx_ideal, dist_ideal,
		                                                     cv2.SOLVEPNP_ITERATIVE)

		C_camera_pose_translation_P.append(translation_vector)
		C_camera_pose_rotation_P.append(rotation_vector)

		print('image ' + str(counter) + 'complete')

	C_camera_pose_translation_P = np.array(C_camera_pose_translation_P, dtype=np.float32)
	C_camera_pose_rotation_P = np.array(C_camera_pose_rotation_P, dtype=np.float32)

	C_camera_pose_translation_P = np.array(C_camera_pose_translation_P.reshape((9, 3)))
	C_camera_pose_rotation_P = np.array(C_camera_pose_rotation_P.reshape((9, 3)))

	np.save('pnp_foup_tvec_change.npy', C_camera_pose_translation_P)
	np.save('pnp_foup_rvec_change.npy', C_camera_pose_rotation_P)

	pass



