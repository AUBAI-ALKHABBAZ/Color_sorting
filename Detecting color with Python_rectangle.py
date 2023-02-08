import cv2
import numpy as np
import os
import datetime
from matplotlib import pyplot as plt




# Create a VideoCapture object and read from input file
# Read until video is completed
#while (True):
    #success, frame =cap.read()
    #cv2.imshow("output" , frame)

cap = cv2.VideoCapture("C:\\Users\\AUBAI\\Desktop\\Open_cv_course_example\\simple_Image_thresholding\\new_pr.mp4")
#cap = cv2.VideoCapture("C:\\Users\\AUBAI\\Desktop\\Digital Logic\\مشروع_تخرجي_أبي\\vedio_new.mp4")
#cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
#out = cv2.VideoWriter('newcolor.mp4',cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (frame_width,frame_height))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
detec = []
Green_boxes = 0
Blue_boxes = 0
offset = 6
c_1 = int(frame_width / 2)
c_2 = int(frame_height/ 2)
def pega_centro(x, y, w, h):

    x1 = int(w / 2)

    y1 = int(h / 2)

    cx = x + x1

    cy = y + y1

    return cx, cy
#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
while (cap.isOpened()):
    # read the frame from the video file
    ret,  imageFrame = cap.read()
    # if the frame was captured successfully
    if ret == True:
        # print the current datetime

        # Reading the video from the
        # webcam in image frames


        # Convert the imageFrame in
        # BGR(RGB color space) to
        # HSV(hue-saturation-value)
        # color space
        hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
        cx = 800
        cy = 250
        pixel_center = hsvFrame [cy, cx]
        hue_value = pixel_center[0]

        color = "Undefined"
        if hue_value < 5:
            color = "no color"
        elif hue_value < 22:
            color = "ORANGE"
        elif hue_value < 33:
            color = "YELLOW"
        elif hue_value < 78:
            color = "GREEN"
        elif hue_value < 131:
            color = "BLUE"
        elif hue_value < 170:
            color = "VIOLET"
        else:
            color = "RED"
        pixel_center_bgr = imageFrame[cy, cx]
        b, g, r = int(pixel_center_bgr[0]), int(pixel_center_bgr[1]), int(pixel_center_bgr[2])
        cv2.circle(imageFrame, (cx, cy), 5, (25, 25, 25), 3)
        # Set range for red color and
        # define mask
        red_lower = np.array([136, 87, 111], np.uint8)
        red_upper = np.array([180, 255, 255], np.uint8)
        red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

        # Set range for green color and
        # define mask
        green_lower = np.array([25, 52, 100], np.uint8)
        green_upper = np.array([102, 255, 255], np.uint8)
        green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

        # Set range for blue color and
        # define mask
        blue_lower = np.array([94, 80, 100], np.uint8)
        blue_upper = np.array([120, 255, 255], np.uint8)
        blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

        # Morphological Transform, Dilation
        # for each color and bitwise_and operator
        # between imageFrame and mask determines
        # to detect only that particular color
        kernal = np.ones((5, 5), "uint8")

        # For red color
        red_mask = cv2.dilate(red_mask, kernal)
        res_red = cv2.bitwise_and(imageFrame, imageFrame,
                                  mask=red_mask)

        # For green color
        green_mask = cv2.dilate(green_mask, kernal)
        res_green = cv2.bitwise_and(imageFrame, imageFrame,
                                    mask=green_mask)

        # For blue color
        blue_mask = cv2.dilate(blue_mask, kernal)
        res_blue = cv2.bitwise_and(imageFrame, imageFrame,
                                   mask=blue_mask)

        cv2.line(imageFrame, (800, 800), (800, 0), (255, 255, 255), 2)
        # Creating contour to track red color
        contours, hierarchy = cv2.findContours(red_mask,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y),
                                           (x + w, y + h),
                                           (0, 0, 0), 5)
                cv2.putText(imageFrame, "Red Color", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (255, 255, 255))
                center = pega_centro(x , y ,w ,h)
                detec.append(center)
                cv2.circle(imageFrame, center, 4, (255, 255, 255), -1)

            # Creating contour to track green color
        contours, hierarchy = cv2.findContours(green_mask,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):

                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y),
                                           (x + w, y + h),
                                           (0, 0, 0), 5)
                cv2.putText(imageFrame, "Green Color", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255))
                center = pega_centro(x, y, w, h)
                detec.append(center)
                cv2.circle(imageFrame, center, 4, (255, 255, 255), -1)
            # Creating contour to track blue color
        contours, hierarchy = cv2.findContours(blue_mask,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y),
                                           (x + w, y + h),
                                           (0, 0, 0), 5)

                cv2.putText(imageFrame, "Blue Color", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (255, 255, 255))
                center = pega_centro(x, y, w, h)
                detec.append(center)
                cv2.circle(imageFrame, center, 4, (255, 255, 255), -1)

        for (x, y) in detec:
            #print(hue_value)
            #pixel_center = hsvFrame[x, y]
            #hue_value = pixel_center[0]
            #cv2.rectangle(imageFrame, (c_1 - 350, 10), (c_1 + 30, 100), (255, 255, 255), -1)
            cv2.putText(imageFrame, "No.Green boxes = "+str(Green_boxes), (c_1 - 300, 50), 0, 1, (0, 0, 0), 2)
            cv2.putText(imageFrame, "No.Blue boxes = "+str(Blue_boxes), (c_1 - 300, 520), 0, 1, (0, 0, 0), 2)
            if x < (900) and x > (860) and hue_value == 74  :
                Green_boxes += 1
                cv2.line(imageFrame, (800, 800), (800, 0), (100, 50,255), 3)
                detec.remove((x, y))
                print("No  box green = : " + str(Green_boxes))
            elif x < (900) and x > (860) and hue_value == 112:
                Blue_boxes += 1
                cv2.line(imageFrame, (800, 800), (800, 0), (100, 50, 255), 3)
                detec.remove((x, y))
                print("No  box Blue = : " + str(Blue_boxes))
    #out.write(imageFrame)
    cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)
    #cv2.imshow("green_mask", green_mask)
        #cv2.imshow("hsvFrame", hsvFrame)

    key = cv2.waitKey(20)
    # if key q is pressed then break
    if key == 113:
        break

    # Closes video file or capturing device.
cap.release()
# finally destroy/close all open windows
cv2.destroyAllWindows()
