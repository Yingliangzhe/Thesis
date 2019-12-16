import cv2
import  numpy as np
from createChessboardPoint import *
import glob
import math
from matplotlib import pyplot as plt

u_axis = 9
v_axis = 6

if __name__ == '__main__':

    # 1 -----------
    # camera calibration --> camera matrix, distEfficient
    # Arrays to store object points and image points from all the images
    object_points = []
    image_points = []
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    file_name_set = 'D:/Diplomarbeit/Bildsatz/TestGF_1/chessboard/' + '*.bmp'
    chessboard_list = sorted(glob.glob(file_name_set))

    for counter in range(1,len(chessboard_list)+1):
        print('load the original image:')
        chessboard = cv2.imread('D:/Diplomarbeit/Bildsatz/Reinraum_f12_testbild/chessboard_6/' + str(counter) + '.bmp')
        ret, corners = cv2.findChessboardCorners(chessboard, (u_axis, v_axis), None)
        # this instruction makes the shape of corners into (54,2)
        corners = np.squeeze(corners, axis=1)
        np.array(corners)
        print(corners)

        if ret == True and counter <= len(chessboard_list):
            chessboard_copy = chessboard.copy()
            chessboard_copy = cv2.cvtColor(chessboard_copy, cv2.COLOR_BGR2GRAY)
            objp = createChessboardPoint(corners)
            object_points.append(objp)
            corners_subpixel = cv2.cornerSubPix(chessboard_copy, corners, (11, 11), (-1, -1), criteria)
            image_points.append(corners_subpixel)
            # draw and display the corners
            cv2.drawChessboardCorners(chessboard, (u_axis, v_axis), corners_subpixel, ret)
            cv2.namedWindow('show Corners' + str(counter),0)
            cv2.imshow('show Corners' + str(counter), chessboard)
            cv2.waitKey(1)

    cv2.destroyAllWindows()

    # Calibrate Camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points,image_points,chessboard_copy.shape[::-1],None,None)

    total_error = 0
    for i in range(len(object_points)):
        imgpoints2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], mtx, dist)
        print(imgpoints2.shape)
        imgpoints2 = imgpoints2.reshape(54, 2)
        print(imgpoints2.shape)

        error = cv2.norm(image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error = total_error + error

    print(total_error)
    if total_error < 1:
        print('camera calibration is good')
        np.save('mtx_f12.npy',mtx)
        np.save('dist_f12',dist)
        #np.save('mtx_new', mtx_new)
        #np.save('dist_new', dist_new)

    # 2--------------
    # this time we have the camera parameter, based on this we can calculate more accrate result of extern parameter
    object_points_new = []
    image_points_new = []
    # after knowing the intrisic parameter matrix, we can use the solvepnp to calculate the
    # undistort function
    for counter in range(1,len(chessboard_list)+1):

        img_original = cv2.imread('D:/Diplomarbeit/Bildsatz/Reinraum_f12_testbild/chessboard_6/' + str(counter) + '.bmp')
        h, w = img_original.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))
        img_undistort = cv2.undistort(img_original,mtx,dist,None,new_camera_matrix)
        cv2.imwrite('calibration_result_'+str(counter)+'.png',img_undistort)

        # find the chessboard corner after undistorted, this time the corner must be different from the previous
        ret, corners = cv2.findChessboardCorners(img_undistort, (u_axis, v_axis), None)
        corners = np.squeeze(corners, axis=1)
        np.array(corners)

        if ret == True and counter <= len(chessboard_list):
            chessboard_copy = img_undistort.copy()
            chessboard_copy = cv2.cvtColor(chessboard_copy, cv2.COLOR_BGR2GRAY)
            objp = createChessboardPoint(corners)
            object_points_new.append(objp)
            corners_subpixel = cv2.cornerSubPix(chessboard_copy, corners, (11, 11), (-1, -1), criteria)
            image_points_new.append(corners_subpixel)
            # draw and display the corners
            cv2.drawChessboardCorners(img_undistort, (u_axis, v_axis), corners_subpixel, ret)
            cv2.namedWindow('show Corners' + str(counter),0)
            cv2.imshow('show Corners' + str(counter), img_undistort)
            cv2.waitKey(1)

            corners_for_solvepnp = np.array([corners_subpixel[0],corners_subpixel[8],corners_subpixel[45],corners_subpixel[53]])
            object_points_for_solvepnp = np.array([objp[0],objp[8],objp[45],objp[53]])

    # find the extern parameter
    ret, mtx_new, dist_new, rvecs_new, tvecs_new = cv2.calibrateCamera(object_points_new, image_points_new, chessboard_copy.shape[::-1], None,
                                                       None)


    object_points_new_array = np.array(object_points_new,dtype=np.float64)

    image_points_new_array = np.array(image_points_new,dtype=np.float32)


    image_points_array = np.array(image_points)
    total_error = 0
    for i in range(len(object_points)):
        imgpoints2, _ = cv2.projectPoints(object_points[i],rvecs_new[i], tvecs_new[i], mtx_new, dist_new)
        print(imgpoints2.shape)
        imgpoints2 = imgpoints2.reshape(54,2)
        print(imgpoints2.shape)

        error = cv2.norm(image_points_new[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error = total_error + error

    print(total_error)
    if total_error < 1:
        print('camera calibration is good')
        #np.save('mtx_f12.npy',mtx)
        #np.save('dist_f12',dist)
        #np.save('mtx_new',mtx_new)
        #np.save('dist_new',dist_new)


    # after undistorting, the translation and rotation need to be calculated
    distance_set = []
    for index in range(len(tvecs) - 1):
        x_dis_square = abs(tvecs_new[index][0] - tvecs_new[index + 1][0]) ** 2
        y_dis_square = abs(tvecs_new[index][1] - tvecs_new[index + 1][1]) ** 2
        z_dis_square = abs(tvecs_new[index][2] - tvecs_new[index + 1][2]) ** 2
        dis_square_sum = x_dis_square + y_dis_square + z_dis_square
        distance = math.sqrt(dis_square_sum)
        distance_set.append(distance)
        if index == len(tvecs) - 2:
            break




