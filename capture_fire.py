# This script captures video, converts it to HSV, creates a mask of the fire, and applies the mask to the frame.
# The frame is displayed in a window.
# Press 'q' to exit the program.

import cv2 as cv
import sys
import numpy as np

cap = cv.VideoCapture(1) # Capture video from the default camera
if not cap.isOpened(): # Check if the camera is opened
    sys.exit("Error: Could not open video stream.\nCheck if the camera is connected and the index is correct.")

# HSV lower and upper bounds for fire detection (broadened for real flame colors)
lower_orange = np.array([10, 100, 180])
upper_orange = np.array([35, 255, 255])
lower_red1 = np.array([0, 150, 150])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 150, 150])
upper_red2 = np.array([180, 255, 255])
lower_blue = np.array([90, 50, 200])
upper_blue = np.array([130, 150, 255])
lower_white = np.array([0, 0, 220]) 
upper_white = np.array([50, 60, 255])

last_frame = None
last_brightness = 0

while True:
    ret, frame = cap.read() # Read the frame from the camera
    if not ret: 
        sys.exit("Error: Could not read frame from camera.")
    
    # Convert the frame to HSV and apply GaussianBlur
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    hsv_blur = cv.GaussianBlur(hsv_frame, (5, 5), 0)

    # Create a combined mask for fire tones (orange, red, blue, white)
    mask_orange = cv.inRange(hsv_blur, lower_orange, upper_orange)
    mask_red1 = cv.inRange(hsv_blur, lower_red1, upper_red1)
    mask_red2 = cv.inRange(hsv_blur, lower_red2, upper_red2)
    mask_blue = cv.inRange(hsv_blur, lower_blue, upper_blue)
    mask_white = cv.inRange(hsv_blur, lower_white, upper_white)
    mask = cv.bitwise_or(mask_orange, mask_red1)
    mask = cv.bitwise_or(mask, mask_red2)
    mask = cv.bitwise_or(mask, mask_blue)
    mask = cv.bitwise_or(mask, mask_white)

    result = cv.bitwise_and(frame, frame, mask=mask) # Apply the mask to the frame

    # Apply morphological filters to the edges
    edges = cv.GaussianBlur(result, (5, 5), 0)
    edges = cv.morphologyEx(edges, cv.MORPH_OPEN, np.ones((3,3), np.uint8))
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, np.ones((3,3), np.uint8))

    # Convert to grayscale and perform Canny edge detection
    edges_gray = cv.cvtColor(edges, cv.COLOR_BGR2GRAY)
    edges_gray_canny = cv.Canny(edges_gray, 60, 160)

    # Brightness flicker check
    brightness = np.mean(cv.cvtColor(frame, cv.COLOR_BGR2GRAY)) # Measures overall brightness fluctuations
    flicker = abs(brightness - last_brightness) 
    last_brightness = brightness

    # Find contours of the fire and draw a rectangle around it
    contours, _ = cv.findContours(edges_gray_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv.contourArea(c)
        if 100 < area and flicker > 2:
            x, y, w, h = cv.boundingRect(c)
            cv.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv.putText(result, "Fire", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the results
    cv.imshow('Raw', cv.resize(frame, (500, 400)))
    cv.imshow('Fire Detection', cv.resize(result, (500, 400)))
    if cv.waitKey(1) & 0xFF == ord('q'): # Press 'q' to exit
        break

# Release the camera and destroy all windows
cap.release()
cv.destroyAllWindows()
