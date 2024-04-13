import cv2
import json

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config['folder_path']


folder_path = read_config('config.json')

print(folder_path)
    
    
# Function to handle mouse click event
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        coordinates = f"({x}, {y})"
        print("Pixel coordinates:", coordinates)
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(img, coordinates, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow("Webcam", img)

# Open webcam
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Couldn't open webcam")
    exit()

# Create window to display webcam feed
cv2.namedWindow("Webcam")

# Set mouse callback function
cv2.setMouseCallback("Webcam", click_event)

while True:
    # Read frame from webcam
    ret, img = cap.read()
    img = cv2.resize(img, (640, 320))
    # Display frame
    cv2.imshow("Webcam", img)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # Check if 's' key is pressed to save the image
    if key == ord('s'):
        cv2.imwrite(folder_path+'\\captured_image.jpg', img)
        print("Image saved")

    # Check if 'esc' key is pressed to exit
    elif key == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
