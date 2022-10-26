#! /usr/bin/python
 
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from mqtt_helper import Mqtt
import face_recognition
import imutils
import pickle
import time
import cv2
import random
import json


mqtt = Mqtt()

vs = VideoStream(usePiCamera=True).start()
#vs = VideoStream(usePiCamera=True,resolution=(640, 480),).start()
time.sleep(2.0)
 
# start the FPS counter
fps = FPS().start()
 


while True:
    # grab the frame from the threaded video stream and resize it
    # to 500px (to speedup processing)
    frame = vs.read()
    #frame = imutils.resize(frame, width=500)
    frame = imutils.resize(frame, width=300)
    # Detect the fce boxes
    boxes = face_recognition.face_locations(frame)
    text = "Locking In!!!!!"
    #names = []
                    
    # loop over the recognized faces
    # left = x1, top = y1, right = x2, bottom = y2
    for (top, right, bottom, left), i in enumerate(boxes):
        
        # draw the predicted face name on the image - color is in BGR
        midpoint_x = int((left + right)/2)
        midpoint_y = int((top + bottom)/2)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, text, (left, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
        cv2.circle(frame,(midpoint_x,midpoint_y), 5, (0,0,255), -1)

        #Enable to help with debugging
        #print("Left top co-ordinates - ", left,", ",top)
        #print("Bottom right co-ordinates - ", bottom,", ",right)
        time.sleep(0.5)
        
        msg = {
            'x' : midpoint_x,
            'y' : midpoint_y
        }
        mqtt.publish(f'/face/{i}/', json.dumps(msg))
     

    
    # display the image to our screen
    cv2.imshow("Looking for faces!!!", frame)
    key = cv2.waitKey(1) & 0xFF

    # quit when 'q' key is pressed
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()
 
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

