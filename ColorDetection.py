# OpenCV Color Detection
# Brian Huffman

import cv2
import numpy as np

Init_color = (0, 0, 0, 179, 255, 255)
text = 'Object'
textColor = (255, 255, 255)

def set_color(hue, sat, val):

    #Set limit window constants
    HUE_DELTA = 18
    SAT_DELTA = 32
    VAL_DELTA = 40

    #Compute low and high end of HSV parameters
    low_h = hue - HUE_DELTA
    low_s = sat - SAT_DELTA
    low_v = val - VAL_DELTA
    high_h = hue + HUE_DELTA
    high_s = sat + SAT_DELTA
    high_v = val + VAL_DELTA

    #If a low or high end goes beyond it's range anchor it to the limit
    if low_h < 0:
        low_h = 0
    elif low_s < 0:
        low_s = 0
    elif low_v < 0:
        low_v = 0
    elif high_h > 179:
        high_h = 179
    elif high_s > 255:
        high_s = 255
    elif high_v > 255:
        high_v = 255

    #Pass these values into the track bars
    cv2.setTrackbarPos('lowHue', 'Color Detection', low_h)
    cv2.setTrackbarPos('highHue', 'Color Detection', high_h)
    cv2.setTrackbarPos('lowSat', 'Color Detection', low_s)
    cv2.setTrackbarPos('highSat', 'Color Detection', high_s)
    cv2.setTrackbarPos('lowVal', 'Color Detection', low_v)
    cv2.setTrackbarPos('highVal', 'Color Detection', high_v)

    #Determine the color of the selected pixel/object

def pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert the current frame from default BGR to HSV and extract values from matrices
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, sat, val = frame_hsv[y, x, 0], frame_hsv[y, x, 1], frame_hsv[y, x, 2]
        #Use these values to set the color in the main loop
        set_color(hue, sat, val)
        #For debugging purposes this prints pixel coordinates and hsv values
        print([x, y, hue, sat, val])

def trackbar_change(*args):
    #Do nothing on trackbar change
    pass

#On program start create two windows one for the mask and color detection settings and one for the raw video stream
cv2.namedWindow('Color Detection', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Color Detection', 620, 360)
cv2.setWindowTitle('Color Detection', 'Mask Settings')
cv2.namedWindow('Input', cv2.WINDOW_NORMAL)
cv2.setWindowTitle('Input', 'Camera Stream')
#Bind Left Mouse Click Event to our color picking function
cv2.setMouseCallback('Input', pick_color)
#Sliding Value bars to adjust Hue, Saturation, and Value Settings Live
cv2.createTrackbar('lowHue', 'Color Detection', Init_color[0], 179, trackbar_change)
cv2.createTrackbar('highHue', 'Color Detection', Init_color[3], 179, trackbar_change)
cv2.createTrackbar('lowSat', 'Color Detection', Init_color[1], 255, trackbar_change)
cv2.createTrackbar('highSat', 'Color Detection', Init_color[4], 255, trackbar_change)
cv2.createTrackbar('lowVal', 'Color Detection', Init_color[2], 255, trackbar_change)
cv2.createTrackbar('highVal', 'Color Detection', Init_color[5], 255, trackbar_change)

# Initialize webcam
# If 2nd webcam is connected you can access that feed by replacing 0 with 1 (ex: cv2.VideoCapture(1))
vidCapture = cv2.VideoCapture(0)
#Set Capture Sizes
FRAME_WIDTH = 480*2
FRAME_HEIGHT = 620*2
vidCapture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
vidCapture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_HEIGHT)

# Check if the webcam is opened correctly
if not vidCapture.isOpened():
    raise IOError("Cannot open webcam")

#Main Loop
while True:

    # Get the current camera frame
    ret, frame = vidCapture.read()

    # If the camera did not return a frame restart the loop
    if not ret:
        continue

    # Get HSV values from the user interface sliders.
    lowHue = cv2.getTrackbarPos('lowHue', 'Color Detection')
    lowSat = cv2.getTrackbarPos('lowSat', 'Color Detection')
    lowVal = cv2.getTrackbarPos('lowVal', 'Color Detection')
    highHue = cv2.getTrackbarPos('highHue', 'Color Detection')
    highSat = cv2.getTrackbarPos('highSat', 'Color Detection')
    highVal = cv2.getTrackbarPos('highVal', 'Color Detection')

    # Convert the frame to HSV color model.
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # HSV values to define a color range we want to create a mask from
    colorLow = np.array([lowHue, lowSat, lowVal])
    colorHigh = np.array([highHue, highSat, highVal])
    mask = cv2.inRange(frameHSV, colorLow, colorHigh)
    # Shows the mask (black and white image where white represents the color being detected)
    cv2.imshow('Color Detection', mask)
    #Find the contours of the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #If the calculated contours are too small they will equal zero in which
    #case we will not attempt to draw them otherwise the program crashes while
    #adjusting HSV values that the program uses to compute the contours
    if len(contours) != 0:
            
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        #Create a bounding Rectangular box based on the largest contour
        x, y, w, h = cv2.boundingRect(biggest_contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #Place the text for the color currently being detected in the bounding box
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, textColor, 2)

    # Show final output image
    cv2.imshow('Input', frame)
    #Display the frame on screen for 1ms
    key = cv2.waitKey(1)
    # Escape key to terminate
    if key == 27 or 0xFF == 27:
        break

#Close all windows on program termination
cv2.destroyAllWindows()
#Turn off the camera feed on program termination
vidCapture.release()




