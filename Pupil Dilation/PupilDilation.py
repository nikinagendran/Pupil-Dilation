import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow('Pupil Detection')

cv2.createTrackbar('Threshold', 'Pupil Detection', 0, 255, nothing)
cv2.setTrackbarPos('Threshold', 'Pupil Detection', 70)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    threshold_value = cv2.getTrackbarPos('Threshold', 'Pupil Detection')
    
    _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        
        if len(max_contour) >= 5:
            ellipse = cv2.fitEllipse(max_contour)
            cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            cv2.circle(frame, center, 5, (255, 0, 0), -1)
            
            diameter = (ellipse[1][0] + ellipse[1][1]) / 2
            cv2.putText(frame, f'Diameter: {diameter:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Pupil Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
