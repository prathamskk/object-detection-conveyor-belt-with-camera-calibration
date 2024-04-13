import cv2
import numpy as np
import pickle

# Load calibration parameters from "calibration.pkl"
cameraMatrix, dist = pickle.load(open("calibration.pkl", "rb"))


# Load calibration parameters from "calibration.pkl"
rotation_vector, translation_vector = pickle.load(open("extrinsicParams.pkl", "rb"))



# Define ArUco dictionary
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# Load image from webcam (replace '0' with the appropriate webcam index)
cap = cv2.VideoCapture(0)

ret, frame = cap.read()
cv2.imshow('Img',frame)
cv2.waitKey(0)
# Convert frame to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

chessboardSize = (10,7)
frameSize = (640,320)


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 15
objp = objp * size_of_chessboard_squares_mm


print(objp)
# object_points = np.array([[0, 0, 0],[1, 0, 0],[0, 1, 0],[1, 1, 0]],dtype=np.float32)

img_points, _ = cv2.projectPoints(objp, rotation_vector, translation_vector, cameraMatrix, dist)
print(img_points)

# projection matrix
Lcam=cameraMatrix.dot(np.hstack((cv2.Rodrigues(rotation_vector)[0],translation_vector)))

px=365
py=109
Z=0
X=np.linalg.inv(np.hstack((Lcam[:,0:2],np.array([[-1*px],[-1*py],[-1]])))).dot((-Z*Lcam[:,2]-Lcam[:,3]))

print("X =", X)

# Draw projected marker points
for point in img_points.astype(int):
    cv2.circle(frame, tuple(point[0]), 3, (255, 255, 0), -1)
# Display the image with detected and projected points
cv2.imshow("Detected Marker", frame)
cv2.waitKey(0)



cv2.destroyAllWindows()