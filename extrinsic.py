import cv2 as cv
import numpy as np
import pickle
import glob
import json

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config['folder_path']

folder_path = read_config('config.json')

# Load calibration parameters from "calibration.pkl"
cameraMatrix, dist = pickle.load(open("calibration.pkl", "rb"))

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (10,7)
frameSize = (640,320)


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 15
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


images = glob.glob(folder_path+'\\captured_image.jpg')

for image in images:

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)


cv.destroyAllWindows()
print(objpoints)

print('bruh')
print(imgpoints)
# Estimate camera pose using SolvePnP
success, rotation_vector, translation_vector = cv.solvePnP(objp, corners2, cameraMatrix,
                                                            dist, flags=cv.SOLVEPNP_ITERATIVE)


# Display camera pose
print("Rotation Vector:")
print(rotation_vector)
print("Translation Vector:")
print(translation_vector)

# Save the external Parameters result for later use
pickle.dump((rotation_vector, translation_vector), open( "extrinsicParams.pkl", "wb" ))