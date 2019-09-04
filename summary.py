
#Import necessary packages
import cv2
import imutils
import time
#Read the video 
vs = cv2.VideoCapture("building.mp4")
#Path for the output video
pathOut = "output6.mp4"
#Initialize output and avg as None
out = None
avg = None

start = time.time()
#Keep looping
while True:
    ret, frame = vs.read()

    if frame is None:
        break
    else:
        frame = imutils.resize(frame, width = 500) #Resize the frame to the convenient size
        height, width, layers = frame.shape #Find out the shape of the frame 
        size = (width, height)              #make the same size for the output video as well
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21,21), 0)

    if not out:
        out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'mp4v'), 380, size)
    if avg is None:
        avg = gray.copy().astype("float")
        continue
    #Perform Running average background subtraction
    cv2.accumulateWeighted(gray, avg, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    #Thresholding and dilation for improving the foreground
    thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    #Find out the contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    #cv2.imshow('mask',thresh)

    for c in cnts:
        area = cv2.contourArea(c)
        if area > 1000:
           out.write(frame)
           break # don't check other areas

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(25)

    if key == ord("q"):
        break

end = time.time()
print("time:", end-start)

out.release()
vs.release()
cv2.destroyAllWindows()
