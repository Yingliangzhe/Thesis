import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import glob
from skimage.feature import blob_log
from math import sqrt

def segmentLoadport(img_gray,window_radius,middle_x,middle_y):
	y_distance_1 = 700
	y_distance_2 = 1080
	x_distance = 950
	window_radius = int(window_radius)
	middle_y = int(middle_y)
	middle_x = int(middle_x)

	# comment in Diplomarbeit diary 3.Monat 11.09.2019
	if middle_y-y_distance_1-window_radius < 0:
		TL_pin = img_gray[0:middle_y-y_distance_1+window_radius,middle_x-x_distance-window_radius:middle_x-x_distance+window_radius]
		TL_pin_origin_initial = [0,middle_x-x_distance-window_radius]
	else:
		TL_pin = img_gray[middle_y - y_distance_1 - window_radius:middle_y - y_distance_1 + window_radius,
		         middle_x - x_distance - window_radius:middle_x - x_distance + window_radius]
		TL_pin_origin_initial = [middle_y - y_distance_1 - window_radius, middle_x - x_distance - window_radius]

	plt.figure('TL_pin')
	plt.imshow(TL_pin,cmap='gray')

	if middle_y-y_distance_1-window_radius < 0:
		TR_pin = img_gray[0:middle_y-y_distance_1+window_radius,middle_x+x_distance-window_radius:middle_x+x_distance+window_radius]
		TR_pin_origin_initial = [0,middle_x+x_distance-window_radius]
	else:
		TR_pin = img_gray[middle_y - y_distance_1 - window_radius:middle_y - y_distance_1 + window_radius,
		         middle_x + x_distance - window_radius:middle_x + x_distance + window_radius]
		TR_pin_origin_initial = [middle_y - y_distance_1 - window_radius, middle_x + x_distance - window_radius]

	plt.figure('TR_pin')
	plt.imshow(TR_pin,cmap='gray')

	if middle_y+y_distance_2-window_radius < 0:
		BM_pin = img_gray[0:middle_y + y_distance_2 + window_radius,
		         middle_x - window_radius:middle_x + window_radius]
		BM_pin_origin_initial = [0, middle_x - window_radius]
	else:
		BM_pin = img_gray[middle_y + y_distance_2 - window_radius:middle_y + y_distance_2 + window_radius,
		         middle_x - window_radius:middle_x + window_radius]
		print(BM_pin.shape)
		BM_pin_origin_initial = [middle_y + y_distance_2 - window_radius, middle_x - window_radius]

	plt.figure('BM_pin')
	plt.imshow(BM_pin,cmap='gray')

	plt.show()

	Pins_patch = {'TL_pin':TL_pin,'TR_pin':TR_pin,'BM_pin':BM_pin}

	Pins_patch_origin_initial = {'TL_pin':TL_pin_origin_initial,'TR_pin':TR_pin_origin_initial,'BM_pin':BM_pin_origin_initial}


	return Pins_patch,Pins_patch_origin_initial



def centerTemplateMatching(img_edge,template):

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

	middle_y = top_left[1] + h / 2
	middle_x = top_left[0] + w / 2
	window_radius = max(w / 2, h / 2)

	img_origin_initial = [top_left[1],top_left[0]] # convert to [y,x] format, and store in variable img_origin_initial
	center_window_size_y = h
	center_window_size_x = w

	plt.figure()
	#plt.subplot(121), plt.imshow(res, cmap='gray')
	#plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
	#plt.subplot(122), \
	plt.imshow(img_edge, cmap='gray')
	plt.title('Detected Point')
	plt.show()

	return middle_x,middle_y,window_radius,img_origin_initial,center_window_size_x,center_window_size_y

def loadPinTemplate(file_name_set):
	pin_templates = {}
	pin_list = sorted(glob.glob(file_name_set))

	for pin_name in pin_list:
		temp_pin_name = pin_name.split('\\')[-1].split('.')[-2]
		pin_img = cv2.imread(pin_name)
		single_pin_template = dict(zip([temp_pin_name],[pin_img]))
		pin_templates.update(single_pin_template)

	return pin_templates


def pinTemplateMatching(Pins_patch,pin_templates):
	# using template matching to locate the pin position
	pin_origin_patch = {}
	pin_image = {}

	for pin_patch_name in Pins_patch:
		for pin_name in pin_templates:
			if pin_patch_name in pin_name:

				pin_patch_image = Pins_patch[pin_patch_name]
				pin_patch_image_gamma = exposure.adjust_gamma(pin_patch_image,0.35)
				pin_patch_image_filtered = cv2.bilateralFilter(pin_patch_image_gamma,7,75,75)
				pin_patch_image_edge = cv2.Canny(pin_patch_image_filtered,19,30)

				plt.figure(pin_patch_name)
				plt.imshow(pin_patch_image_edge,cmap='gray')

				pin_template_img = pin_templates[pin_name]
				pin_template_img = cv2.cvtColor(pin_template_img,cv2.COLOR_BGR2GRAY)
				pin_template_img_gamma = exposure.adjust_gamma(pin_template_img,0.35)
				pin_template_img_filtered = cv2.bilateralFilter(pin_template_img_gamma,7,75,75)
				pin_template_img_edge = cv2.Canny(pin_template_img_filtered,19,40)

				plt.figure(pin_name)
				plt.imshow(pin_template_img_edge,cmap='gray')

				w,h = pin_template_img_edge.shape[0:2][::-1]

				res = cv2.matchTemplate(pin_patch_image_edge,pin_template_img_edge,cv2.TM_CCORR_NORMED)

				min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

				top_left = max_loc

				bottom_right = (top_left[0]+h,top_left[1]+w)

				cv2.rectangle(pin_patch_image_edge,top_left,bottom_right,255,2)

				plt.figure(pin_name)
				plt.subplot(221), plt.imshow(res, cmap='gray')
				plt.title('Matching Result')
				plt.subplot(222), plt.imshow(pin_patch_image_edge, cmap='gray')
				plt.title('Detected Point')

				top_left = list(top_left)
				top_left.reverse()
				top_left = tuple(top_left)
				single_pin_pos = dict(zip([pin_name],[top_left]))
				pin_origin_patch.update(single_pin_pos)

				single_clamp_image = dict(
					zip([pin_name], [pin_patch_image[top_left[0]:top_left[0] + h, top_left[1]:top_left[1] + w]]))
				pin_image.update(single_clamp_image)

				plt.subplot(212)
				plt.imshow(single_clamp_image[pin_name], cmap='gray')
				plt.show()

				break

	return pin_origin_patch,pin_image


def exactPinToInitial(pin_image,pin_origin_patch,Pins_patch_origin_initial,y_size,x_size):
	exact_pin_initial = {}
	image_empty = np.zeros((y_size, x_size), np.uint8)
	for pin_name in pin_image:
		pin_image_gray = cv2.cvtColor(pin_image[pin_name],cv2.COLOR_BGR2GRAY)
		# store the pin origin into dictionary
		single_pin_origin_initial = dict(zip([pin_name], [np.array(pin_origin_patch[pin_name]) + np.array(Pins_patch_origin_initial[pin_name])]))
		exact_pin_y_min = single_pin_origin_initial[pin_name][0]
		exact_pin_x_min = single_pin_origin_initial[pin_name][1]
		exact_pin_y_max = exact_pin_y_min + pin_image[pin_name].shape[0]
		exact_pin_x_max = exact_pin_x_min + pin_image[pin_name].shape[1]
		image_empty[exact_pin_y_min:exact_pin_y_max,exact_pin_x_min:exact_pin_x_max] = pin_image_gray
		exact_pin_initial.update(single_pin_origin_initial)

	# this is not only a visulasation of pins in initial image,
	# but also a reference plot to validate, if the circle center position is right
	plt.figure('pin in initial image')
	plt.imshow(image_empty,cmap='gray')
	plt.title('pin position in initial image')
	plt.show()

	return exact_pin_initial



def findLargeEllipse(ellipse_major_set):
	major_index_set = []
	average_major = np.mean(ellipse_major_set)
	for counter in range(len(ellipse_major_set)):
		major = ellipse_major_set[counter]
		if major > average_major:
			major_index = counter
			major_index_set.append(major_index)

	return major_index_set




def ellipsePointSet(hull_set):

	ellipse_point_set = np.array([])
	for ellipse_hull_index in range(len(hull_set)):
		if ellipse_hull_index == 0:
			ellipse_point_set = hull_set[ellipse_hull_index]
		else:
			ellipse_point_set = np.concatenate((ellipse_point_set,hull_set[ellipse_hull_index]),axis=0)

	return ellipse_point_set

def MSERBlobDetection(pin_image):
	mser = cv2.MSER.create(_min_area=400)

	ellipse_center_point_dic = {}
	for pin_name in pin_image:
		img = pin_image[pin_name]
		img_copy = img.copy()
		img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		#img_hist = cv2.equalizeHist(img_gray, None)
		img_gamma = exposure.adjust_gamma(img_gray, 0.35)

		plt.figure()
		plt.imshow(img_gamma,cmap='gray')
		plt.show()

		#single_point = ()

		regions,bboxes = mser.detectRegions(img_gamma)


		hull_set = []

		for box_counter in range(len(bboxes)):
			if bboxes[box_counter][3] > 105:
				hull = cv2.convexHull(regions[box_counter].reshape(-1,1,2))
				hull_set.append(hull)


		#ellipse_index = findLargeEllipse(ellipse_major_set)

		ellipse_point_set = ellipsePointSet(hull_set)

		ellipse = cv2.fitEllipse(ellipse_point_set)

		cv2.ellipse(img_copy,ellipse,(0,0,255),1)

		# ellipse_center_point store the center point of ellipse in format (y,x)
		ellipse_center_point = (ellipse[0][1],ellipse[0][0])
		single_ellipse_center_point = dict(zip([pin_name],[ellipse_center_point]))
		ellipse_center_point_dic.update(single_ellipse_center_point)

		cv2.imshow('img_copy',img_copy)
		cv2.waitKey(1)

		#hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
		#cv2.polylines(img,hulls,1,(0,255,0))
		img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		img_copy = cv2.cvtColor(img_copy,cv2.COLOR_BGR2RGB)

		plt.figure()
		plt.subplot(121)
		plt.imshow(img)

		plt.subplot(122)
		plt.imshow(img_copy)
		plt.show()

	return ellipse_center_point_dic

def pinCenterToInitial(exact_pin_initial,ellipse_center_point):
	pin_center_point_in_initial = {}
	for pin_name in ellipse_center_point:
		single_circle_center_point = dict(zip([pin_name], [np.array(exact_pin_initial[pin_name])
		                                                   + np.array(ellipse_center_point[pin_name])]))
		pin_center_point_in_initial.update(single_circle_center_point)

	return pin_center_point_in_initial


def Threshold(pin_image):
	edges = {}
	for pin_name in pin_image:
		img = pin_image[pin_name]
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img_hist = cv2.equalizeHist(img_gray,None)
		img_gamma = exposure.adjust_gamma(img_hist,0.35)

		dst_mean = cv2.adaptiveThreshold(img_gamma,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
		dst_gaussian = cv2.adaptiveThreshold(img_gamma,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

		img_bilateral = cv2.bilateralFilter(img_gamma,7,75,75)
		_,dst_bin = cv2.threshold(img_bilateral,120,255,cv2.THRESH_BINARY)

		canny_edge = cv2.Canny(dst_bin,0,0)

		plt.figure()

		plt.subplot(231)
		plt.imshow(img_gray,'gray')
		plt.title('original img')

		plt.subplot(232)
		#img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		plt.imshow(img_gamma,cmap='gray')
		plt.title('Gamma correction')

		plt.subplot(233)
		#img_sift_rgb = cv2.cvtColor(dst_mean, cv2.COLOR_BGR2RGB)
		plt.imshow(dst_mean,cmap='gray')
		plt.title('mean methode')

		plt.subplot(234)
		plt.imshow(dst_gaussian,cmap='gray')
		plt.title('gaussian methode')

		plt.subplot(235)
		plt.imshow(dst_bin,cmap='gray')
		plt.title('binary')

		plt.subplot(236)
		plt.imshow(canny_edge,cmap='gray')
		plt.title('canny edge')


		plt.show()

		single_edge = dict(zip([pin_name],[canny_edge]))
		edges.update(single_edge)

	return edges

def houghCircle(edges,pin_image):
	circle_center = {}
	for img_name in edges:
		img = edges[img_name]
		pin_img = pin_image[img_name]
		circles1 = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10,
		                           param1=30, param2=15, minRadius=50, maxRadius=70)
		circles = circles1[0, :, :]
		m = 0
		img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
		#circles = np.uint16(np.around(circles))
		for i in circles:

			# draw the outer circle
			cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 3)
			# draw the center of the circle
			cv2.circle(img, (i[0], i[1]), 2, (255, 0, 0), 3)

			if m == 0:
				break
			'''
			cv2.namedWindow('circle', 0)
			cv2.imshow('circle', pin_img)
			cv2.waitKey(1)
			'''

		plt.figure()
		#plt.imshow(pin_img)
		plt.imshow(img)
		plt.show()
		single_circle = dict(zip([img_name],[circles[0][0:2][::-1]])) # this operation we change the coordinate into (y,x)
		circle_center.update(single_circle)

	return circle_center




def circleCenterPointInInitial(circle_center,exact_pin_initial):
	circle_center_point_in_initial = {}
	for pin_name in circle_center:
		single_circle_center_point = dict(zip([pin_name],[np.array(exact_pin_initial[pin_name])
		                                                   + np.array(circle_center[pin_name])]))
		circle_center_point_in_initial.update(single_circle_center_point)

	return circle_center_point_in_initial


def findSideMiddlePoint(pin_center_point_in_initial):

	left_middle_point = ((pin_center_point_in_initial['BM_pin'][0] - pin_center_point_in_initial['TL_pin'][0])/2,
	                    (pin_center_point_in_initial['BM_pin'][1] - pin_center_point_in_initial['TL_pin'][1])/2)

	right_middle_point = ((pin_center_point_in_initial['BM_pin'][0] - pin_center_point_in_initial['TR_pin'][0])/2,
	                      (pin_center_point_in_initial['TR_pin'][1] - pin_center_point_in_initial['BM_pin'][1])/2)

	return left_middle_point,right_middle_point

def middlePoint(p2,p4,pin_center_point_in_initial):

	for pin_name in pin_center_point_in_initial:
		if pin_name == 'TR_pin':
			p1 = pin_center_point_in_initial[pin_name]

		elif pin_name == 'TL_pin':
			p3 = pin_center_point_in_initial[pin_name]




	tolerance = 10 ** (-6)

	x1 = p1[1]
	x2 = p2[1]
	x3 = p3[1]
	x4 = p4[1]

	y1 = p1[0]
	y2 = p2[0]
	y3 = p3[0]
	y4 = p4[0]

	a = x2 - x1
	b = x3 - x4
	c = y2 - y1
	d = y3 - y4
	g = x3 - x1
	h = y3 - y1

	f = a * d - b * c

	if abs(f) < tolerance:
		print('inverse matrix cannot be computed')
		if f > 0:
			f = tolerance
		else:
			f = -tolerance

	t = (d * g - b * h) / f
	s = (a * h - c * g) / f

	if t < 0 or t > 1:
		print('two lines do not intersect')

	if s < 0 or t > 1:
		print('two lines do not intersect')

	print('x1 = ', x1, 'y1 = ', y1)
	print('x2 = ', x2, 'y2 = ', y2)
	print('x3 = ', x3, 'y3 = ', y3)
	print('x4 = ', x4, 'y4 = ', y4)

	print('t = ', t, 's = ', s)

	intersection_point = (y1 + t * (y2 - y1), x1 + t * (x2 - x1))

	print('intersection point is ', intersection_point)

	pin_center_point_in_initial.update({'MM': np.array(intersection_point)})

	return pin_center_point_in_initial



if __name__ == '__main__':
	mtx = np.load('mtx.npy')
	dist = np.load('dist.npy')

	img_original = cv2.imread('D:/Diplomarbeit/Bildsatz/TestGF_1/TestGF_anJialiang/Bilder_GF/LOADPORT/view_point/5.png')
	h, w = img_original.shape[:2]
	new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
	img_original = cv2.undistort(img_original, mtx, dist, None, new_camera_matrix)
	img_gray_0 = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

	img = img_original.copy()

	img_gamma = exposure.adjust_gamma(img, 0.35)
	img_gray = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2GRAY)

	plt.figure()
	plt.subplot(1, 2, 1)
	plt.imshow(img_gray, cmap='gray')
	plt.subplot(1, 2, 2)
	plt.imshow(img_gamma, cmap='gray')
	plt.show()

	img_edge = cv2.Canny(cv2.bilateralFilter(img_gray, 5, 75, 75), 20, 50)

	y_size = img.shape[0]
	x_size = img.shape[1]

	# center_template = cv2.imread('circleTemplate.png')
	center_template = cv2.imread('Loadport_template/center.png')
	center_template_gray = cv2.cvtColor(center_template, cv2.COLOR_BGR2GRAY)

	middle_x, middle_y, window_radius, center_origin_initial, center_window_size_x, center_window_size_y = centerTemplateMatching(
		img_edge, center_template_gray)

	# center_img = img_gray[300:1000,500:1000]
	center_y_min = center_origin_initial[0]
	center_y_max = center_origin_initial[0] + center_window_size_y
	center_x_min = center_origin_initial[1]
	center_x_max = center_origin_initial[1] + center_window_size_x
	center_img = img_gray[center_y_min:center_y_max, center_x_min:center_x_max]

	plt.figure()
	plt.imshow(center_img, cmap='gray')
	plt.show()

	Pins_patch, Pins_patch_origin_initial = segmentLoadport(img, window_radius, middle_x, middle_y)

	file_name_set = './Loadport_template/' + '*.png'
	pin_templates = loadPinTemplate(file_name_set)

	pin_origin_patch, pin_image = pinTemplateMatching(Pins_patch, pin_templates)

	exact_pin_initial = exactPinToInitial(pin_image, pin_origin_patch, Pins_patch_origin_initial, y_size, x_size)

	# ----------------------------------
	# --------- MSER Detector ----------
	# ----------------------------------

	ellipse_center_point_dic = MSERBlobDetection(pin_image)
	pin_center_point_in_initial = pinCenterToInitial(exact_pin_initial, ellipse_center_point_dic)

	left_middle_point, right_middle_point = findSideMiddlePoint(pin_center_point_in_initial)

	pin_center_point_in_initial = middlePoint(right_middle_point, left_middle_point, pin_center_point_in_initial)

	# ----------------------------------
	# ------- adaptive Threshold -------
	# ----------------------------------
	edges = Threshold(pin_image)


	# ----------------------------------------------------
	# ------ hough circle after canny edge operation -----
	# ----------------------------------------------------
	circle_center = houghCircle(edges, pin_image)

	pass


