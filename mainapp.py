import tkinter as tk
import cv2
import os
import json
from PIL import Image, ImageTk
import cv2 as cv
import numpy as np
import glob
import pickle
from ultralytics import YOLO
import pandas as pd
import cvzone

def read_config(file_path , key):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config[key]


folder_path = read_config('config.json','folder_path')

chessboardSize_x = read_config('config.json','chessboard_size_x')
chessboardSize_y = read_config('config.json','chessboard_size_y')
chessboardSize = (chessboardSize_x,chessboardSize_y)

size_of_chessboard_squares_mm = read_config('config.json','size_of_chessboard_square_in_mm')



class Dashboard:
    def __init__(self, master):
        self.master = master
        self.master.title("Dashboard")
        self.master.resizable(False, False)
        
        self.calibration_status = "Pending"
        
        # Create a Canvas widget to display the webcam feed or video
        self.canvas = tk.Canvas(self.master, width=640, height=480)
        self.canvas.pack(fill='both', expand=True)
        
        # Display an initial image on the canvas (replace 'initial_image.jpg' with your image)
        initial_image = Image.open('yolo.jpg')  # Replace 'initial_image.jpg' with your image path
        initial_photo = ImageTk.PhotoImage(image=initial_image)
        self.canvas.img = initial_photo
        self.canvas.create_image(0, 0, anchor=tk.NW, image=initial_photo)
        
        # Create a frame to hold the buttons
        self.button_frame = tk.Frame(self.master)
        self.button_frame.pack(fill='x')
        
        self.dashboard_label = tk.Label(self.button_frame, text="Calibration Status : " + self.calibration_status , font=("Arial", 12) ,foreground="red")
        self.dashboard_label.pack(side='right')

        self.webcam_button = tk.Button(self.button_frame, text="Launch Webcam", command=self.launch_webcam)
        self.webcam_button.pack(side='left')
        self.webcam_button.config(state=tk.DISABLED)
        
        self.calibration_button = tk.Button(self.button_frame, text="Launch Calibration Wizard", command=self.launch_calibration)
        self.calibration_button.pack(side='left')

        
        self.check_and_update_calibration()
        

    def launch_calibration(self):
        self.master.withdraw()  # Hide dashboard window
        calibration_window = CalibrationWizard(self.master, self , self.check_and_update_calibration , chessboardSize ,size_of_chessboard_squares_mm)

    def launch_webcam(self):
        self.master.withdraw()  # Hide dashboard window
        webcam_window = WebcamScreen(self.master, self ,chessboardSize ,size_of_chessboard_squares_mm)
        
    
    def check_and_update_calibration(self):
        if os.path.exists("calibration.pkl") and os.path.exists("extrinsicParams.pkl"):
            self.calibration_status = "Done"
            self.dashboard_label.config(text="Calibration Status : " + self.calibration_status , foreground="green")
            self.webcam_button.config(state=tk.NORMAL)
        else:
            self.calibration_status = "Pending"
            self.dashboard_label.config(text="Calibration Status : " + self.calibration_status , foreground="red")
            self.webcam_button.config(state=tk.DISABLED)

class CalibrationWizard:
    def __init__(self, master, dashboard , check_and_update_calibration , chessboardSize , size_of_chessboard_squares_mm):
        
        self.master = master
        self.dashboard = dashboard
        self.check_and_update_calibration = check_and_update_calibration
        
        #calibration settings
        self.chessboardSize = chessboardSize
        self.size_of_chessboard_squares_mm = size_of_chessboard_squares_mm
        
        #logic variables
        self.image_count = 0
        self.extrinsic_capture = False
        self.original_frame = None
        self.frame_with_corners = None
        
        #initialize camera variables
        self.cap = None
        self.is_camera_on = False
        self.frame_count = 0
        self.frame_skip_threshold = 3
        
    
        
        # Create a Toplevel window for the calibration wizard
        self.calibration_window = tk.Toplevel(master)
        self.calibration_window.geometry("+%d+%d" % (master.winfo_rootx() , master.winfo_rooty()))  # Adjust as needed
        self.calibration_window.grab_set()
        self.calibration_window.title("Calibration Wizard")

        # Create a Label widget to display the calibration instructions
        self.image_count_label = tk.Label(self.calibration_window, text="Calibration Image :"+str(self.image_count)+"/20", font=("Arial", 18))
        self.image_count_label.pack(pady=20)
        
        # Create a Canvas widget to display the webcam feed or video
        self.canvas = tk.Canvas(self.calibration_window, width=640, height=480)
        self.canvas.pack(fill='both', expand=True)
        
        # Create a "Capture" button to capture the calibration image
        self.capture_button = tk.Button(self.calibration_window, text="Capture", command=self.capture_image)
        self.capture_button.pack()
        self.capture_button.config(state=tk.DISABLED)
        self.calibration_window.protocol("WM_DELETE_WINDOW", self.on_close) 
        
        # Initialize the webcam feed
        self.start_webcam()
        
    def start_webcam(self):
        
        if not self.is_camera_on:
            self.cap = cv2.VideoCapture(0)  # Use the default webcam (you can change the index if needed)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.is_camera_on = True
            self.video_paused = False
            self.update_canvas()  # Start updating the canvas
            
    def update_canvas(self):
        if self.is_camera_on:
            if not self.video_paused:
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.original_frame = frame.copy()
                          
                
                    chessboardSize = self.chessboardSize
                    frameSize = (640,480)

                    # termination criteria
                    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    # termination criteria
                    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


                    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
                    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
                    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

                    size_of_chessboard_squares_mm = self.size_of_chessboard_squares_mm
                    objp = objp * size_of_chessboard_squares_mm


                    # Arrays to store object points and image points from all the images.
                    objpoints = [] # 3d point in real world space
                    imgpoints = [] # 2d points in image plane.
                    
                    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                    # Find the chess board corners
                    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

                    # If found, add object points, image points (after refining them)
                    if ret == True:
                        self.capture_button.config(state=tk.NORMAL)
                        objpoints.append(objp)
                        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                        imgpoints.append(corners)

                        # Draw and display the corners
                        cv.drawChessboardCorners(frame, chessboardSize, corners2, ret)
                        self.frame_with_corners = frame
                        photo = ImageTk.PhotoImage(image=Image.fromarray(self.frame_with_corners))
                        self.canvas.img = photo
                        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                    else:
                        self.capture_button.config(state=tk.DISABLED)    
                        photo = ImageTk.PhotoImage(image=Image.fromarray(self.original_frame))
                        self.canvas.img = photo
                        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                             
            self.canvas.after(10, self.update_canvas)
                
    def capture_image(self):
        if self.is_camera_on:
            self.video_paused = True
            if self.frame_with_corners is not None:
                if self.extrinsic_capture is not True:
                    detection_screen = DetectionVerificationScreen(self.calibration_window, self.frame_with_corners , self.original_frame , self.callback_response)
                else:
                    self.extrinsic_calibration()
                    
    def callback_response(self, response, frame):
        if response == "Accepted":
            directory = folder_path+"\\images"
            cv2.imwrite(directory+'\\images' + str(self.image_count) + '.png', frame)
            print(response)
            print("image saved!")
            self.image_count += 1
            self.image_count_label.config(text="Calibration Image :"+str(self.image_count)+"/20")
            if self.image_count == 20:
                # Sending 20th image to undistort error screen
                UndistortErrorScreen(self.calibration_window , self.frame_with_corners , self.original_frame , self.calibration_callback , self.chessboardSize , size_of_chessboard_squares_mm)
                self.video_paused = True
        elif response == "Declined":
            print(response)
            print("image not saved!")
        self.video_paused = False
        
    def calibration_callback(self, response):
        if response == "Declined":
            self.video_paused = False
            self.image_count = 0
            self.image_count_label.config(text="Calibration Image :"+str(self.image_count)+"/20")
            self.frame_with_corners = None
            self.original_frame = None
        elif response == "Accepted":
            self.extrinsic_capture = True
            self.image_count_label.config(text="Capture Image for Extrinsic Calibration")
                    
    def extrinsic_calibration(self):
        # Load calibration parameters from "calibration.pkl"
        cameraMatrix, dist = pickle.load(open("calibration.pkl", "rb"))

        ################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

        chessboardSize = self.chessboardSize
        frameSize = (640,480)


        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

        size_of_chessboard_squares_mm = self.size_of_chessboard_squares_mm
        objp = objp * size_of_chessboard_squares_mm

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        img = self.original_frame
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

        cv.destroyAllWindows()

        # Estimate camera pose using SolvePnP
        success, rotation_vector, translation_vector = cv.solvePnP(objp, corners2, cameraMatrix,
                                                                    dist, flags=cv.SOLVEPNP_ITERATIVE)

        # Save the external Parameters result for later use
        pickle.dump((rotation_vector, translation_vector), open( "extrinsicParams.pkl", "wb" ))
        self.check_and_update_calibration()
        self.on_close()


    def on_close(self):
        self.stop_webcam()
        
        self.calibration_window.grab_release()
        self.calibration_window.destroy()
        self.master.deiconify()  # Show dashboard window after completing calibration

    # Function to stop the webcam feed
    def stop_webcam(self):
        if self.cap is not None:
            self.cap.release()
            self.is_camera_on = False
            self.video_paused = False

class DetectionVerificationScreen:
    def __init__(self, master, frame_with_corners , original_frame, callback_response):
        self.frame_with_corners = frame_with_corners
        self.original_frame = original_frame
        self.master = master
        self.callback_response = callback_response
        self.detection_window = tk.Toplevel(master)
        self.detection_window.geometry("+%d+%d" % (master.winfo_rootx() , master.winfo_rooty())) 
        self.detection_window.grab_set()
        self.detection_window.title("Confirm Detection")
        
        # Create a Canvas widget to display the webcam feed or video
        self.canvas = tk.Canvas(self.detection_window, width=640, height=480)
        self.canvas.pack(fill='both', expand=True)
        
        photo = ImageTk.PhotoImage(image=Image.fromarray(self.frame_with_corners))
        self.canvas.img = photo
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)

        
        self.accept_button = tk.Button(self.detection_window, text="Accept", command=self.accept_detection)
        self.accept_button.pack()
        
        self.decline_button = tk.Button(self.detection_window, text="Decline", command=self.decline_detection)
        self.decline_button.pack()
                
        self.detection_window.protocol("WM_DELETE_WINDOW", self.decline_detection) 



    def accept_detection(self):
        self.callback_response("Accepted",self.original_frame)
        self.detection_window.grab_release()
        self.detection_window.destroy()
        
    def decline_detection(self):
        self.callback_response("Declined",self.original_frame)
        self.detection_window.grab_release()
        self.detection_window.destroy()

class UndistortErrorScreen:
    def __init__(self, master, frame_with_corners , original_frame, calibration_callback , chessboardSize , size_of_chessboard_squares_mm):
        self.frame_with_corners = frame_with_corners
        self.original_frame = original_frame
        self.master = master
        self.calibration_callback = calibration_callback
        self.error_calibration_window = tk.Toplevel(master)
        self.error_calibration_window.geometry("+%d+%d" % (master.winfo_rootx(), master.winfo_rooty())) 
        self.error_calibration_window.grab_set()
        self.error_calibration_window.title("Check Undistortion Error")
        
         #calibration settings
        self.chessboardSize = chessboardSize
        self.size_of_chessboard_squares_mm = size_of_chessboard_squares_mm
        
        # Create a Canvas widget to display the webcam feed or video
        self.canvas = tk.Canvas(self.error_calibration_window, width=640, height=480)
        self.canvas.pack(fill='both', expand=True)
        
        self.reprojection_error = self.calibration()
        
        self.error_label = tk.Label(self.error_calibration_window, text="total error: {}".format(self.reprojection_error), font=("Arial", 18))
        self.error_label.pack(pady=20)
        
        self.accept_button = tk.Button(self.error_calibration_window, text="Accept", command=self.accept_detection)
        self.accept_button.pack()
        
        self.decline_button = tk.Button(self.error_calibration_window, text="Decline", command=self.decline_detection)
        self.decline_button.pack()
                
        self.error_calibration_window.protocol("WM_DELETE_WINDOW", self.decline_detection) 



    def accept_detection(self):
        self.calibration_callback("Accepted")
        self.error_calibration_window.grab_release()
        self.error_calibration_window.destroy()
        
    def decline_detection(self):
        print("Declined Calibration")
        self.calibration_callback("Declined")
        self.error_calibration_window.grab_release()
        self.error_calibration_window.destroy()
        
    def calibration(self):
        chessboardSize =  self.chessboardSize 
        frameSize = (640,480)

        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

        size_of_chessboard_squares_mm = self.size_of_chessboard_squares_mm
        objp = objp * size_of_chessboard_squares_mm


        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.


        images = glob.glob(folder_path+'\\images\\*.png')
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
        cv.destroyAllWindows()
        
        ############## CALIBRATION #######################################################

        ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        pickle.dump((cameraMatrix, dist), open( "calibration.pkl", "wb" ))
        
        ############## UNDISTORTION #####################################################

        img = self.frame_with_corners
        h,  w = img.shape[:2]
        newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

        # Undistort
        dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        photo = ImageTk.PhotoImage(image=Image.fromarray(dst))
        self.canvas.img = photo
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
               
        # Reprojection Error
        mean_error = 0

        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
            mean_error += error

        # print( "total error: {}".format(mean_error/len(objpoints)) )
        return mean_error/len(objpoints)

class WebcamScreen:
    
    def __init__(self, master, dashboard , chessboardSize , size_of_chessboard_squares_mm):
        self.master = master
        self.dashboard = dashboard
        self.webcam_window = tk.Toplevel(master)
        self.webcam_window.geometry("+%d+%d" % (master.winfo_rootx(), master.winfo_rooty())) 
        self.webcam_window.grab_set()
        self.webcam_window.title("Webcam Screen")
        
         #calibration settings
        self.chessboardSize = chessboardSize
        self.size_of_chessboard_squares_mm = size_of_chessboard_squares_mm
        
        # Create a Canvas widget to display the webcam feed or video
        self.canvas = tk.Canvas(self.webcam_window, width=640, height=480)
        self.canvas.pack(fill='both', expand=True)

        self.class_list = self.read_classes_from_file('classes.txt')

        self.class_selection = tk.StringVar()
        self.class_selection.set("All")  # Default selection is "All"
        self.class_selection_label = tk.Label(self.webcam_window, text="Select Class:")
        self.class_selection_label.pack(side='left')
        self.class_selection_entry = tk.OptionMenu(self.webcam_window, self.class_selection, "All", *self.class_list)  # Populate dropdown with classes from the text file
        self.class_selection_entry.pack(side='left')

        # Create a frame to hold the buttons
        self.button_frame = tk.Frame(self.webcam_window)
        self.button_frame.pack(fill='x')

        # Create a "Play" button to start the webcam feed
        self.play_button = tk.Button(self.button_frame, text="Play", command=self.start_webcam)
        self.play_button.pack(side='left')

        # Create a "Stop" button to stop the webcam feed
        self.stop_button = tk.Button(self.button_frame, text="Stop", command=self.stop_webcam)
        self.stop_button.pack(side='left')

        # Create a "Pause/Resume" button to pause or resume video
        self.pause_button = tk.Button(self.button_frame, text="Pause/Resume", command=self.pause_resume_video)
        self.pause_button.pack(side='left')

        # Create a "Quit" button to close the application
        self.quit_button = tk.Button(self.button_frame, text="Quit", command=self.on_close)
        self.quit_button.pack(side='left')

        # Display an initial image on the canvas (replace 'initial_image.jpg' with your image)
        initial_image = Image.open('yolo.jpg')  # Replace 'initial_image.jpg' with your image path
        initial_photo = ImageTk.PhotoImage(image=initial_image)
        self.canvas.img = initial_photo
        self.canvas.create_image(0, 0, anchor=tk.NW, image=initial_photo)

        
        self.webcam_window.protocol("WM_DELETE_WINDOW", self.on_close) 
        
                
        # Load calibration parameters from "calibration.pkl"
        self.cameraMatrix, self.dist = pickle.load(open("calibration.pkl", "rb"))

        # Load calibration parameters from "calibration.pkl"
        self.rotation_vector, self.translation_vector = pickle.load(open("extrinsicParams.pkl", "rb"))

        
                
        # Global variables for OpenCV-related objects and flags
        self.cap = None
        self.is_camera_on = False
        self.frame_count = 0
        self.frame_skip_threshold = 3
        self.model = YOLO('barc_model.pt')
        self.video_paused = False

    # Function to read coco.txt
    def read_classes_from_file(self,file_path):
        with open(file_path, 'r') as file:
            classes = [line.strip() for line in file]
        return classes

    # Function to start the webcam feed
    def start_webcam(self):
        if not self.is_camera_on:
            self.cap = cv2.VideoCapture(0)  # Use the default webcam (you can change the index if needed)
            self.is_camera_on = True
            self.video_paused = False
            self.update_canvas()  # Start updating the canvas

    # Function to stop the webcam feed
    def stop_webcam(self):
        if self.cap is not None:
            self.cap.release()
            self.is_camera_on = False
            self.video_paused = False

    # Function to pause or resume the video
    def pause_resume_video(self):
        self.video_paused = not self.video_paused

    # Function to update the Canvas with the webcam frame or video frame
    def update_canvas(self):
        if self.is_camera_on:
            if not self.video_paused:
                ret, frame = self.cap.read()
                if ret:
                    self.frame_count += 1
                    if self.frame_count % self.frame_skip_threshold != 0:
                        self.canvas.after(10, self.update_canvas)
                        return

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (640, 480))
                    selected_class = self.class_selection.get()

                    results = self.model.predict(frame)
                    a = results[0].boxes.data
                    px = pd.DataFrame(a).astype("float")
                    for index, row in px.iterrows():
                        x1 = int(row[0])
                        y1 = int(row[1])
                        x2 = int(row[2])
                        y2 = int(row[3])
                        conf = row[4]
                        d = int(row[5])
                        
                        # Calculating center of bounding box
                        xtotal = x1 + x2
                        xcent = int(xtotal/2)
                        ytotal = y1 + y2
                        ycent = int(ytotal/2)
                        z=0
                        
                        # Calculating Real world Coordinates
                        Lcam=self.cameraMatrix.dot(np.hstack((cv2.Rodrigues(self.rotation_vector)[0],self.translation_vector)))
                        realWorldCoordinates=np.linalg.inv(np.hstack((Lcam[:,0:2],np.array([[-1*xcent],[-1*ycent],[-1]])))).dot((-z*Lcam[:,2]-Lcam[:,3]))
            

                        c = self.class_list[d]
                        if selected_class == "All" or c == selected_class:
                            #draw bounding box of detected object
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            #mark center of the bounding box
                            cv2.circle(frame, tuple([xcent,ycent]), 3, (255, 255, 0), -1)
                            #put real word coordinates above bounding box
                            cvzone.putTextRect(frame, f'{c}, {conf:.2f}, X:{realWorldCoordinates[0]:.1f}mm, Y:{realWorldCoordinates[1]:.1f}mm', (x1, y1), 1, 1)
                            
                    chessboardSize = self.chessboardSize
                    frameSize = (640,480)


                    # termination criteria
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


                    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
                    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
                    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

                    size_of_chessboard_squares_mm = self.size_of_chessboard_squares_mm
                    objp = objp * size_of_chessboard_squares_mm

                    # project points using calibration parameters
                    img_points, _ = cv2.projectPoints(objp, self.rotation_vector, self.translation_vector, self.cameraMatrix, self.dist)   
                                    
                    # Draw projected marker points TEMPORARY
                    for point in img_points.astype(int):
                        cv2.circle(frame, tuple(point[0]), 3, (255, 255, 0), -1)
                        
                        
                    # h,  w = frame.shape[:2]
                    # newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))
                    # # Undistort
                    # frame = cv2.undistort(frame, cameraMatrix, dist, None, newCameraMatrix)
                    # # crop the image
                    # x, y, w, h = roi
                    # frame = frame[y:y+h, x:x+w]




                    photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                    self.canvas.img = photo
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)

            self.canvas.after(10, self.update_canvas)

    def on_close(self):
        self.webcam_window.grab_release()
        self.webcam_window.destroy()
        self.stop_webcam()
        self.master.deiconify()  # Show dashboard window after completing calibration




def main():
    root = tk.Tk()
    app = Dashboard(root)
    root.mainloop()

if __name__ == "__main__":
    main()