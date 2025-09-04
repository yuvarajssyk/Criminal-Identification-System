# camera.py
import numpy as np
import os
import cv2
import PIL.Image
from PIL import Image
import shutil
class VideoCamera3(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        ff=open("video.txt","r")
        fn=ff.read()
        ff.close()
        
        self.video = cv2.VideoCapture(0)
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter("static/video/"+fn,self.fourcc, 20.0, (640,480))
        self.k=1

        
    
    def __del__(self):
        self.video.release()
        
    
    def get_frame(self):
        success, image = self.video.read()

        if success==True:
            #image = cv2.flip(image,0)
            # write the flipped frame
            self.out.write(image)
        
            
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
