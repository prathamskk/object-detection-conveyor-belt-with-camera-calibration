import win32event
import win32api
import sys
from winerror import ERROR_ALREADY_EXISTS
mutex = win32event.CreateMutex(None, False, 'name')
last_error = win32api.GetLastError()
if last_error == ERROR_ALREADY_EXISTS:
   sys.exit(0)


import tkinter as tk
from tkinter.ttk import *
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from tkinter import filedialog
import pandas as pd
from ultralytics import YOLO
import cvzone
import threading
import numpy as np
import pickle

# Load calibration parameters from "calibration.pkl"
cameraMatrix, dist = pickle.load(open("calibration.pkl", "rb"))

# Load calibration parameters from "calibration.pkl"
rotation_vector, translation_vector = pickle.load(open("extrinsicParams.pkl", "rb"))

chessboardSize = (10,7)
frameSize = (640,320)


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 15
objp = objp * size_of_chessboard_squares_mm

# project points using calibration parameters
img_points, _ = cv2.projectPoints(objp, rotation_vector, translation_vector, cameraMatrix, dist)
print(img_points)


# Global variables for OpenCV-related objects and flags
cap = None
is_camera_on = False
frame_count = 0
frame_skip_threshold = 3
model = YOLO('barc_model.pt')
video_paused = False

# Function to read coco.txt
def read_classes_from_file(file_path):
    with open(file_path, 'r') as file:
        classes = [line.strip() for line in file]
    return classes

# Function to start the webcam feed
def start_webcam():
    global cap, is_camera_on, video_paused
    if not is_camera_on:
        cap = cv2.VideoCapture(0)  # Use the default webcam (you can change the index if needed)
        is_camera_on = True
        video_paused = False
        update_canvas()  # Start updating the canvas

# Function to stop the webcam feed
def stop_webcam():
    global cap, is_camera_on, video_paused
    if cap is not None:
        cap.release()
        is_camera_on = False
        video_paused = False

# Function to pause or resume the video
def pause_resume_video():
    global video_paused
    video_paused = not video_paused

# Function to start video playback from a file
def select_file():
    global cap, is_camera_on, video_paused
    if is_camera_on:
        stop_webcam()  # Stop the webcam feed if running
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if file_path:
        cap = cv2.VideoCapture(file_path)
        is_camera_on = True
        video_paused = False
        update_canvas()  # Start updating the canvas with the video

# Function to update the Canvas with the webcam frame or video frame
def update_canvas():
    global is_camera_on, frame_count, video_paused
    if is_camera_on:
        if not video_paused:
            ret, frame = cap.read()
            if ret:
                frame_count += 1
                if frame_count % frame_skip_threshold != 0:
                    canvas.after(10, update_canvas)
                    return

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 320))
                selected_class = class_selection.get()

                results = model.predict(frame)
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
                    Lcam=cameraMatrix.dot(np.hstack((cv2.Rodrigues(rotation_vector)[0],translation_vector)))
                    realWorldCoordinates=np.linalg.inv(np.hstack((Lcam[:,0:2],np.array([[-1*xcent],[-1*ycent],[-1]])))).dot((-z*Lcam[:,2]-Lcam[:,3]))
         


                    c = class_list[d]
                    if selected_class == "All" or c == selected_class:
                        #draw bounding box of detected object
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        #mark center of the bounding box
                        cv2.circle(frame, tuple([xcent,ycent]), 3, (255, 255, 0), -1)
                        #put real word coordinates above bounding box
                        cvzone.putTextRect(frame, f'{c}, {conf:.2f}, X:{realWorldCoordinates[0]:.1f}mm, Y:{realWorldCoordinates[1]:.1f}mm', (x1, y1), 1, 1)
                        
                       
           
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
                canvas.img = photo
                canvas.create_image(0, 0, anchor=tk.NW, image=photo)

        canvas.after(10, update_canvas)

# Function to quit the application
def quit_app():
    stop_webcam()
    root.quit()
    root.destroy()

# Create the main Tkinter window
root = tk.Tk()
root.title("YOLO v8 My App")

# Create a Canvas widget to display the webcam feed or video
canvas = tk.Canvas(root, width=640, height=320)
canvas.pack(fill='both', expand=True)

class_list = read_classes_from_file('classes.txt')

class_selection = tk.StringVar()
class_selection.set("All")  # Default selection is "All"
class_selection_label = tk.Label(root, text="Select Class:")
class_selection_label.pack(side='left')
class_selection_entry = tk.OptionMenu(root, class_selection, "All", *class_list)  # Populate dropdown with classes from the text file
class_selection_entry.pack(side='left')

# Create a frame to hold the buttons
button_frame = tk.Frame(root)
button_frame.pack(fill='x')



# Function to go to a new window when clicked on "Camera Calibration" button
import tkinter as tk
from tkinter import ttk  # for labelframe

# # Import your calibration functions from calibration.py
# from calibration import calibrate_camera, save_calibration_data


def open_calibration_window():
  # Create a new window for calibration
  calibration_window = tk.Tk()
  calibration_window.title("Camera Calibration")

  # Instructions labelframe
  instructions_labelframe = ttk.LabelFrame(calibration_window, text="Instructions")
  instructions_labelframe.pack(padx=10, pady=10)
  instructions_text = """
  1. Print a chessboard pattern on a piece of paper.
  2. Make sure the chessboard is well-lit and flat.
  3. Place the chessboard in front of the camera.
  4. Click the 'Capture Frame' button to capture an image of the chessboard.
  5. Use the mouse to click on the corners of the chessboard in the captured image.
  6. Click the 'Calibrate Camera' button to calculate the camera parameters.
  7. Click the 'Save Calibration' button to save the parameters to a file.
  """
  instructions_label = tk.Label(instructions_labelframe, text=instructions_text)
  instructions_label.pack(padx=10, pady=10)

  # Image frame
  image_frame = ttk.LabelFrame(calibration_window, text="Captured Image")
  image_frame.pack(padx=10, pady=10)
  # Create a label to hold the captured image (initially empty)
  captured_image_label = tk.Label(image_frame)
  captured_image_label.pack()

  # Button frame
  button_frame = tk.Frame(calibration_window)
  button_frame.pack(padx=10, pady=10)

  # Capture frame button
  capture_button = tk.Button(button_frame, text="Capture Frame", command=lambda: capture_frame(captured_image_label))
  capture_button.pack(side='left', padx=10)

  # Calibrate button (disabled initially)
  calibrate_button = tk.Button(button_frame, text="Calibrate Camera", command=lambda: calibrate_camera(), state='disabled')
  calibrate_button.pack(side='left', padx=10)

  # Save button (disabled initially)
  save_button = tk.Button(button_frame, text="Save Calibration", command=lambda: save_calibration(), state='disabled')
  save_button.pack(side='left', padx=10)

  # Logic for capturing frame, calibration, and saving
  def capture_frame(image_label):
    # Simulate capturing a frame (replace with your camera logic)
    image = Image.open("chessboard.jpg")  # Replace with captured image from camera
    photo_image = ImageTk.PhotoImage(image=image)
    image_label.config(image=photo_image)
    image_label.image = photo_image  # Keep a reference to prevent garbage collection

    # Enable calibration button after capturing a frame
    calibrate_button.config(state='normal')

  def calibrate_camera():
    # Call your camera calibration function from calibration.py
    # Assuming calibrate_camera() returns the camera matrix and distortion coefficients
    global camera_matrix, distortion_coefficients  # Declare variables as global to access in save_calibration
    camera_matrix, distortion_coefficients = calibrate_camera(captured_image_label)  # Pass the image label for corner detection (if needed)

    # Enable save button after calibration
    save_button.config(state='normal')

  def save_calibration():
    # Call your function to save calibration data (camera matrix, distortion coefficients)
    # Replace 'calibration.pkl' with your desired file name
    save_calibration_data(camera_matrix, distortion_coefficients, "calibration.pkl")

    # Show a success message
    message_label = tk.Label(calibration_window, text="Calibration data saved successfully!")
    message_label.pack()

  calibration_window.mainloop()






# Create a "Play" button to start the webcam feed
play_button = tk.Button(button_frame, text="Play", command=start_webcam)
play_button.pack(side='left')

# Create a "Stop" button to stop the webcam feed
stop_button = tk.Button(button_frame, text="Stop", command=stop_webcam)
stop_button.pack(side='left')

# Create a "Select File" button to choose a video file
file_button = tk.Button(button_frame, text="Select File", command=select_file)
file_button.pack(side='left')

# Create a "Pause/Resume" button to pause or resume video
pause_button = tk.Button(button_frame, text="Pause/Resume", command=pause_resume_video)
pause_button.pack(side='left')


# Create a "Camera Calibration" button
camera_calibration_button = tk.Button(button_frame, text="Camera Calibration", command=open_calibration_window)
camera_calibration_button.pack(side='left')



# Create a "Quit" button to close the application
quit_button = tk.Button(button_frame, text="Quit", command=quit_app)
quit_button.pack(side='left')

# Display an initial image on the canvas (replace 'initial_image.jpg' with your image)
initial_image = Image.open('yolo.jpg')  # Replace 'initial_image.jpg' with your image path
initial_photo = ImageTk.PhotoImage(image=initial_image)
canvas.img = initial_photo
canvas.create_image(0, 0, anchor=tk.NW, image=initial_photo)

# Start the Tkinter main loop
root.mainloop()
