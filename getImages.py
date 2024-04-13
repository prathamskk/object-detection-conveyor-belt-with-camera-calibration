import cv2
import os
import json

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config['folder_path']


folder_path = read_config('config.json')

print(folder_path)
    
cap = cv2.VideoCapture(0)
directory = folder_path+"\\images"
num = 0
os.chdir(directory) 
while cap.isOpened():

    succes, img = cap.read()
    img = cv2.resize(img, (640, 320))
    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('images' + str(num) + '.png', img)
        print("image saved!")
        num += 1
        print(os.listdir(directory)) 

    cv2.imshow('Img',img)
# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()