import numpy as np

def createChessboardPoint(corners):
	first = 0
	fifth = 4
	threshold = 140
	last = len(corners) - 1
	u_axis = 9
	v_axis = 6

	x, y = np.meshgrid(range(0, u_axis, 1), range(v_axis - 1, -1, -1))
	object_points = np.hstack((x.reshape(54, 1), y.reshape(54, 1), np.zeros((54, 1)))).astype(np.float32)
	object_points_image = object_points * 15

	return object_points_image