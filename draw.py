import numpy as np
import cv2
from collections import deque

class drawings():
	def setValues(x):
	   print("")

	def stackImages(scale,imgArray):
	    rows = len(imgArray)
	    cols = len(imgArray[0])
	    rowsAvailable = isinstance(imgArray[0], list)
	    width = imgArray[0][0].shape[1]
	    height = imgArray[0][0].shape[0]
	    if rowsAvailable:
	        for x in range ( 0, rows):
	            for y in range(0, cols):
	                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
	                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
	                else:
	                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
	                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
	        imageBlank = np.zeros((height, width, 3), np.uint8)
	        hor = [imageBlank]*rows
	        hor_con = [imageBlank]*rows
	        for x in range(0, rows):
	            hor[x] = np.hstack(imgArray[x])
	        ver = np.vstack(hor)
	    else:
	        for x in range(0, rows):
	            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
	                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
	            else:
	                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
	            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
	        hor= np.hstack(imgArray)
	        ver = hor
	    return ver


	Color_detectors=cv2.namedWindow("Color_detectors")
	Color_detectors=cv2.createTrackbar("Upper Hue", "Color_detectors", 153, 180,setValues)
	Color_detectors=cv2.createTrackbar("Upper Saturation", "Color_detectors", 255, 255,setValues)
	Color_detectors=cv2.createTrackbar("Upper Value", "Color_detectors", 255, 255,setValues)
	Color_detectors=cv2.createTrackbar("Lower Hue", "Color_detectors", 64, 180,setValues)
	Color_detectors=cv2.createTrackbar("Lower Saturation", "Color_detectors", 72, 255,setValues)
	Color_detectors=cv2.createTrackbar("Lower Value", "Color_detectors", 49, 255,setValues)

	bpoints = [deque(maxlen=1024)]
	gpoints = [deque(maxlen=1024)]
	rpoints = [deque(maxlen=1024)]
	ypoints = [deque(maxlen=1024)]

	blue_index = 0
	green_index = 0
	red_index = 0
	yellow_index = 0

	kernel = np.ones((5,5),np.uint8)

	colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)] #blue,green,red,yellow
	colorIndex = 0

	paintWindow = np.zeros((480,640,3)) + 255
	paintWindow = cv2.rectangle(paintWindow, (40,1), (140,65), (0,0,0), 2)
	paintWindow = cv2.rectangle(paintWindow, (160,1), (255,65), colors[0], -1)
	paintWindow = cv2.rectangle(paintWindow, (275,1), (370,65), colors[1], -1)
	paintWindow = cv2.rectangle(paintWindow, (390,1), (485,65), colors[2], -1)
	paintWindow = cv2.rectangle(paintWindow, (505,1), (600,65), colors[3], -1)

	cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
	cv2.putText(paintWindow, "", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
	cv2.putText(paintWindow, "", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
	cv2.putText(paintWindow, "", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
	cv2.putText(paintWindow, "", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_AA)
	#cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

	ssss = 0
	cap = cv2.VideoCapture(ssss)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 430)

	while True:
	    if ssss == 0:
	        ret, frame = cap.read()
	        frame = cv2.flip(frame, 1)
	    elif ssss == 1:
	        ret, frame = cap.read()
	    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	    u_hue = cv2.getTrackbarPos("Upper Hue", "Color_detectors")
	    u_saturation = cv2.getTrackbarPos("Upper Saturation", "Color_detectors")
	    u_value = cv2.getTrackbarPos("Upper Value", "Color_detectors")
	    l_hue = cv2.getTrackbarPos("Lower Hue", "Color_detectors")
	    l_saturation = cv2.getTrackbarPos("Lower Saturation", "Color_detectors")
	    l_value = cv2.getTrackbarPos("Lower Value", "Color_detectors")
	    Upper_hsv = np.array([u_hue,u_saturation,u_value])
	    Lower_hsv = np.array([l_hue,l_saturation,l_value])

	    frame = cv2.rectangle(frame, (40,1), (140,65), (0,0,0), 2)
	    frame = cv2.rectangle(frame, (160,1), (255,65), colors[0], -1)
	    frame = cv2.rectangle(frame, (275,1), (370,65), colors[1], -1)
	    frame = cv2.rectangle(frame, (390,1), (485,65), colors[2], -1)
	    frame = cv2.rectangle(frame, (505,1), (600,65), colors[3], -1)
	    cv2.putText(frame, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
	    cv2.putText(frame, "1", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
	    cv2.putText(frame, "2", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
	    cv2.putText(frame, "3", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
	    cv2.putText(frame, "4", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_AA)

	    Mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv)
	    Mask = cv2.erode(Mask, kernel, iterations=1)
	    Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel)
	    Mask = cv2.dilate(Mask, kernel, iterations=1)

	    cnts,_ = cv2.findContours(Mask.copy(), cv2.RETR_EXTERNAL,
	    	cv2.CHAIN_APPROX_SIMPLE)
	    center = None

	    if len(cnts) > 0:
	        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
	        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
	        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
	        M = cv2.moments(cnt)
	        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

	        if center[1] <= 65:
	            if 40 <= center[0] <= 140: # Clear Button
	                bpoints = [deque(maxlen=512)]
	                gpoints = [deque(maxlen=512)]
	                rpoints = [deque(maxlen=512)]
	                ypoints = [deque(maxlen=512)]

	                blue_index = 0
	                green_index = 0
	                red_index = 0
	                yellow_index = 0

	                paintWindow[67:,:,:] = 255
	            elif 160 <= center[0] <= 255:
	                    colorIndex = 0 # Blue
	            elif 275 <= center[0] <= 370:
	                    colorIndex = 1 # Green
	            elif 390 <= center[0] <= 485:
	                    colorIndex = 2 # Red
	            elif 505 <= center[0] <= 600:
	                    colorIndex = 3 # Yellow
	        else :
	            if colorIndex == 0:
	                bpoints[blue_index].appendleft(center)
	            elif colorIndex == 1:
	                gpoints[green_index].appendleft(center)
	            elif colorIndex == 2:
	                rpoints[red_index].appendleft(center)
	            elif colorIndex == 3:
	                ypoints[yellow_index].appendleft(center)
	    else:
	        bpoints.append(deque(maxlen=512))
	        blue_index += 1
	        gpoints.append(deque(maxlen=512))
	        green_index += 1
	        rpoints.append(deque(maxlen=512))
	        red_index += 1
	        ypoints.append(deque(maxlen=512))
	        yellow_index += 1

	    points = [bpoints, gpoints, rpoints, ypoints]
	    for i in range(len(points)):
	        for j in range(len(points[i])):
	            for k in range(1, len(points[i][j])):
	                if points[i][j][k - 1] is None or points[i][j][k] is None:
	                    continue
	                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
	                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

	    imgStack = stackImages(1,([frame,paintWindow]))
	    cv2.imshow("Draw", imgStack)
	    #cv2.imshow("Paint", paintWindow)

	    if cv2.waitKey(1) & 0xFF == ord("q"):
	        break

	cap.release()
	cv2.destroyAllWindows()