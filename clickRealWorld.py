import cv2
import numpy as np
import pickle

# Load calibration parameters from "calibration.pkl"
cameraMatrix, dist = pickle.load(open("calibration.pkl", "rb"))

# Load calibration parameters from "calibration.pkl"
rotation_vector, translation_vector = pickle.load(open("extrinsicParams.pkl", "rb"))





import cv2

# Function to handle mouse click event
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global pixelx, pixely
        pixelx = x
        pixely = y
        
        
        # # Given data
        # uv_point = np.array([[pixelx, pixely]], dtype=np.float64)  # u = 363, v = 222
        # Z_const = 285  # Height Z_const
        # dist_coeffs = dist
        # # Rotation matrix and translation vector
        rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
        rotation_matrix_inv = np.linalg.inv(rotation_matrix)
        camera_matrix_inv = np.linalg.inv(cameraMatrix)
        tvec = translation_vector

        # # Distortion correction
        # uv_point_undistorted = cv2.undistortPoints(uv_point.reshape(1, -1, 2), cameraMatrix, dist_coeffs, P=cameraMatrix)
        # uv_point_undistorted = uv_point_undistorted.reshape(-1, 2)

        # # Compute left side of the equation
        # left_side_mat = np.dot(rotation_matrix_inv, np.dot(camera_matrix_inv, np.hstack((uv_point_undistorted, np.ones((uv_point_undistorted.shape[0], 1)))).T))

        # # Compute right side of the equation
        # right_side_mat = np.dot(rotation_matrix_inv, translation_vector)

        # # Compute scaling factor
        # s = (Z_const + right_side_mat[2]) / left_side_mat[2]

        # # Compute 3D point
        # P = np.dot(rotation_matrix_inv, (s * np.dot(camera_matrix_inv, np.hstack((uv_point_undistorted, np.ones((uv_point_undistorted.shape[0], 1)))).T) - translation_vector))

        # print("Computed 3D Point (P):", P)


        # Given data
        uv_point = np.array([[pixelx, pixely, 1]], dtype=np.float64).T  # u = 363, v = 222, homogeneous coordinates
        Z_const = 1  # Height Z_const

        # Compute left side of the equation
        left_side_mat = np.dot(rotation_matrix_inv, np.dot(camera_matrix_inv, uv_point))

        # Compute right side of the equation
        right_side_mat = np.dot(rotation_matrix_inv, tvec)

        # Compute scaling factor
        s = (Z_const + right_side_mat[2]) / left_side_mat[2]

        # Compute 3D point
        P = np.dot(rotation_matrix_inv, (s * np.dot(camera_matrix_inv, uv_point) - tvec))

        print("Pixel coordinates:", coordinates)
        print("Computed 3D Point (P):", (P).T)

        
        # projection matrix
        Lcam=cameraMatrix.dot(np.hstack((cv2.Rodrigues(rotation_vector)[0],translation_vector)))
        #(187, 132)
        px=pixelx
        py=pixely
        Z=0
        X=np.linalg.inv(np.hstack((Lcam[:,0:2],np.array([[-1*px],[-1*py],[-1]])))).dot((-Z*Lcam[:,2]-Lcam[:,3]))
        print("Computed 3D Point (X):", X)
        
        

# Open webcam
cap = cv2.VideoCapture(1)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Couldn't open webcam")
    exit()

# Create window to display webcam feed
cv2.namedWindow("Webcam")

# Set mouse callback function
cv2.setMouseCallback("Webcam", click_event)
pixelx = 0
pixely = 0

while True:
    # Read frame from webcam
    ret, img = cap.read()
    img = cv2.resize(img, (640, 480))
    
    coordinates = f"({pixelx}, {pixely})"
    
    cv2.circle(img, (pixelx, pixely), 5, (0, 255, 0), -1)
    cv2.putText(img, coordinates, (pixelx, pixely), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    

    # h,  w = img.shape[:2]
    # newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

    # # Undistort
    # img = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)


    # # crop the image
    # x, y, w, h = roi
    # img = img[y:y+h, x:x+w]



    # Display frame
    cv2.imshow("Webcam", img)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF


    # Check if 'esc' key is pressed to exit
    if key == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
