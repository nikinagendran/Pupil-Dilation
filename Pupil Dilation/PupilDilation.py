import cv2
import numpy as np

def nothing(x):
    pass

# Create a window
cv2.namedWindow('Pupil Detection')

# Create trackbars for adjusting threshold values
cv2.createTrackbar('Threshold', 'Pupil Detection', 0, 255, nothing)
cv2.setTrackbarPos('Threshold', 'Pupil Detection', 70)

# Start video capture from webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply a Gaussian blur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # Get the current threshold value from the trackbar
    threshold_value = cv2.getTrackbarPos('Threshold', 'Pupil Detection')
    
    # Apply binary thresholding to the image
    _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        
        # Fit an ellipse to the largest contour
        if len(max_contour) >= 5:
            ellipse = cv2.fitEllipse(max_contour)
            
            # Draw the ellipse on the original frame
            cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
            
            # Draw the center of the ellipse
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            cv2.circle(frame, center, 5, (255, 0, 0), -1)
            
            # Calculate and display the pupil diameter
            diameter = (ellipse[1][0] + ellipse[1][1]) / 2
            cv2.putText(frame, f'Diameter: {diameter:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the original frame with the detected pupil
    cv2.imshow('Pupil Detection', frame)
    
    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
