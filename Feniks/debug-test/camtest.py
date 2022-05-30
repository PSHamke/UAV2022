from easyImage import CameraStream
import os
import time
import cv2
from Timer import Timer



stream = CameraStream([1280, 720])
stream.start()



while True:
    frame = stream.getFrame()
    scopeTimer = Timer()
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    
    print("FPS: ", scopeTimer.FPS())
stream.closeCamera()
